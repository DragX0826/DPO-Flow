# max_flow/train_maxrl.py
"""
MaxFlow: Maximum Likelihood RL for Universal Geometric Drug Design.
Supports distributed multi-GPU training via HuggingFace Accelerate,
mixed-precision (FP16/BF16), SNR-Aware EMA, and WandB logging.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from max_flow.data.preference_dataset import PreferenceDataset, preference_collate_fn
from max_flow.models.backbone import CrossGVP
from max_flow.models.flow_matching import RectifiedFlow
from max_flow.models.max_rl import MaxRL
from max_flow.utils.training import (
    optimize_for_intel, get_optimizer, get_scheduler,
    AverageMeter, DynamicRewardScaler, SNRAwareEMA,
    CSVLogger, SilentStepLogger
)
import warnings
# Suppress Pydantic warnings (including UnsupportedFieldAttributeWarning)
# This must be done as early as possible
warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
from max_flow.config import MaxRLConfig
from accelerate import Accelerator
import json
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def load_partial_weights(model, checkpoint_path, accelerator):
    """
    SOTA Surgery (Phase 65): Loads backbone weights while skipping mismatching heads.
    Essential for transitioning from regression to categorical atom types.
    """
    accelerator.print(f"🔧 Performing weight surgery on {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Skip atom_head (Regression 1 -> Categorical 9)
        if "atom_head" in k:
            accelerator.print(f"   Skipping mismatch layer: {k}")
            continue
        new_state_dict[k] = v
        
    # strict=False allows loading the backbone while leaving the new atom_head randomized
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    accelerator.print(f"✅ Surgery complete. Missing: {missing}, Unexpected: {unexpected}")
    return model

def train_maxrl(args):
    """Main MaxRL training entry point."""

    # ─── 1. Config & Accelerator ───
    config_path = getattr(args, 'config', 'MaxRL_config.json')
    config = MaxRLConfig.load(config_path) if os.path.exists(config_path) else MaxRLConfig()
    config.batch_size = getattr(args, 'batch_size', config.batch_size)
    config.lr = getattr(args, 'lr', config.lr)
    config.beta = getattr(args, 'beta', config.beta)
    config.use_maxrl = getattr(args, 'use_maxrl', config.use_maxrl)

    # Mixed Precision: SOTA preference for bfloat16 if T4/A100/H100
    mixed_precision = config.mixed_precision
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        if mixed_precision == "fp16":
            mixed_precision = "bf16"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="wandb" if args.use_wandb else None
    )
    device = accelerator.device
    accelerator.print(f"[MaxFlow] Device: {device} | Precision: {mixed_precision}")

    if args.use_wandb and accelerator.is_local_main_process:
        accelerator.init_trackers("MaxFlow", config=vars(args))

    # ─── 2. Dataset ───
    index_path = os.path.join(args.data_root, "shards_manifest.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            manifest = json.load(f)
        accelerator.print(f"Loaded {len(manifest)} shards from manifest.")
    else:
        manifest = [f for f in os.listdir(args.data_root) if f.endswith(".pt")]
        accelerator.print(f"No manifest found. Loaded {len(manifest)} .pt files.")

    dataset = PreferenceDataset(manifest, root_dir=args.data_root, use_pt=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=preference_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # ─── 3. Models ───
    def create_model():
        # SOTA Phase 65: node_in_dim updated to 58 (Organic Allowlist Only)
        backbone = CrossGVP(
            node_in_dim=58,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
        return RectifiedFlow(backbone)

    policy_rf = create_model()
    ref_rf = create_model()

    # SOTA Phase 66: Surgical Fine-Tuning Setup
    def surgical_setup(model, checkpoint_path, stage, accelerator):
        """
        Executes weight surgery and freezing strategy.
        Stage 1: Head Warm-up (Backbone Frozen)
        Stage 2: Full Fine-Tuning (All Trainable)
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            load_partial_weights(model, checkpoint_path, accelerator)
        
        if stage == 1:
            accelerator.print("❄️  [Stage 1] Freezing Backbone (GVP+Mamba). Only Atom Head Trainable.")
            for name, param in model.backbone.named_parameters():
                if "atom_head" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    accelerator.print(f"🔥 Atom Head: {name} is HOT")
        else:
            accelerator.print("🔓 [Stage 2] Unfrozen Backbone for Physics Adaptation.")
            for param in model.parameters():
                param.requires_grad = True
        
        return model

    # Apply Surgical Setup
    surgical_setup(policy_rf, args.rf_checkpoint, args.stage, accelerator)

    ref_rf.eval()
    for param in ref_rf.parameters():
        param.requires_grad = False

    MaxRL_wrapper = MaxRL(policy_rf, ref_rf, config=config)

    # ─── 4. Dynamic Reward Scaler & EMA ───
    reward_scaler = DynamicRewardScaler() if config.use_dynamic_scaling else None

    ema_model = None
    if config.use_ema:
        accelerator.print(f"Initializing SNR-Aware EMA (decay={config.ema_decay})")
        ema_model = SNRAwareEMA(policy_rf, decay_base=config.ema_decay)

    if config.compile_model and hasattr(torch, "compile"):
        accelerator.print("Compiling backbone with torch.compile...")
        policy_rf.backbone = torch.compile(policy_rf.backbone, mode="reduce-overhead")

    # ─── 5. Optimizer & Scheduler ───
    # Auto-adjust LR if not specified
    if args.lr == 3e-5: # Default
        config.lr = 1e-3 if args.stage == 1 else 5e-5
        accelerator.print(f"💡 Auto-setting LR for Stage {args.stage}: {config.lr}")

    total_steps = len(dataloader) * args.epochs
    
    # Filter trainable parameters for optimizer
    trainable_params = [p for p in policy_rf.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    accelerator.print(f"🚀 Trainable Parameters: {num_trainable/1e6:.2f}M")

    if getattr(args, 'quant_combat', False):
        from max_flow.optimization.schedule_free import HybridSFOAdamW
        accelerator.print("⚔️ QUANT COMBAT MODE: HybridSFO-AdamW")
        optimizer = HybridSFOAdamW(
            trainable_params, lr=config.lr,
            warmup_steps=int(0.1 * total_steps)
        )
        scheduler = None
    else:
        optimizer = get_optimizer(policy_rf, learning_rate=config.lr) # get_optimizer should filter internaly or we pass filtered
        scheduler = get_scheduler(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
    # SOTA Phase 4: PCGrad & Adaptive Loss
    from max_flow.optimization.adaptive_loss import AdaptiveLossWrapper
    from max_flow.optimization.pcgrad import PCGrad
    
    adaptive_loss_fn = None
    if getattr(args, 'use_adaptive_loss', False):
        task_names = ['MaxRL', 'geom', 'conf', 'vib', 'anchor', 'clip']
        adaptive_loss_fn = AdaptiveLossWrapper(num_tasks=len(task_names), task_names=task_names).to(device)
        # Add adaptive_loss params to optimizer
        optimizer.add_param_group({'params': adaptive_loss_fn.parameters(), 'lr': config.lr})
        accelerator.print("📈 SOTA: Adaptive Loss Weighting enabled.")

    pcgrad_optimizer = None
    if getattr(args, 'use_pcgrad', False):
        pcgrad_optimizer = PCGrad(optimizer)
        accelerator.print("⚔️ SOTA: PCGrad Gradient Surgery enabled.")

    # ─── 6. Prepare with Accelerator ───
    policy_rf, optimizer, dataloader = accelerator.prepare(
        policy_rf, optimizer, dataloader
    )
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    ref_rf = ref_rf.to(device)

    # ─── 7. Training Loop ───
    from max_flow.utils.metrics import MultiObjectiveScorer
    scorer = MultiObjectiveScorer()
    # Setup CSV Logger
    if accelerator.is_main_process:
        csv_logger = CSVLogger("MaxRL_training_stats.csv", ["epoch", "step", "loss", "reward_win", "reward_lose", "lr", "MaxRL", "anchor", "geom", "clip", "conf", "vib"])

    effective_epoch = 0
    for epoch in range(args.epochs):
        effective_epoch = epoch + 1 + args.epoch_offset
        policy_rf.train()
        loss_meter = AverageMeter()
        
        # Silent Logger
        step_logger = SilentStepLogger(
            accelerator, 
            total_steps=len(dataloader), 
            interval=20, 
            desc=f"MaxRL Epoch {effective_epoch}"
        )

        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            batch_win, batch_lose = batch

            # Move to device once
            batch_win = batch_win.to(device)
            batch_lose = batch_lose.to(device)

            # ── Vectorized Reward (Phase 63/SOTA) ──
            with torch.no_grad():
                reward_win, mask_win = scorer.calculate_batch_reward(batch_win)
                reward_lose, mask_lose = scorer.calculate_batch_reward(batch_lose)
                
                # SOTA: Masking & Filtering
                valid_mask = mask_win & mask_lose
                valid_count = valid_mask.sum().item()
                valid_ratio = valid_count / batch_win.num_graphs
                
                if valid_count < 1:  # Skip entire batch if no valid pairs
                    continue
                
                # SOTA: Debug Dump for high failure rates
                if valid_ratio < 0.5 and i % 10 == 0:
                    dump_path = os.path.join(args.data_root, f"debug_batch_epoch{epoch}_step{i}.pt")
                    # Note: args.output_dir might not be set, using data_root or generic path
                    # torch.save({'win': batch_win, 'lose': batch_lose, 'mask': valid_mask}, dump_path)
                    # accelerator.print(f"⚠️ [DEBUG] High reward failure rate ({valid_ratio:.2%}).")
                
                # Keep track of which one is actually better for logging purposes
                # SOTA Hardening: Use torch.where to safely pick from valid indices
                r_max = torch.where(valid_mask, torch.max(reward_win, reward_lose), torch.tensor(float('nan'), device=device))

                if reward_scaler is not None:
                    # Combined stats for global population monitoring
                    # We only update scaler with valid rewards
                    if valid_mask.any():
                        active_rewards = torch.cat([reward_win[valid_mask], reward_lose[valid_mask]])
                        reward_scaler.update(active_rewards)
                    
                    # SOTA: Advantage Normalization (Normalize the difference)
                    # This stabilizes the logits significantly per-batch.
                    reward_win = reward_scaler.normalize(reward_win)
                    reward_lose = reward_scaler.normalize(reward_lose)

            optimizer.zero_grad()
            # SOTA Hardening: Explicitly enable grads to ensure confidence/guidance heads are backprop-active
            with torch.set_grad_enabled(True):
                result = MaxRL_wrapper.loss(
                    batch_win, batch_lose,
                    reward_win=reward_win, reward_lose=reward_lose,
                    valid_mask=valid_mask
                )
            
            # Phase 63: Decomposed loss return
            if isinstance(result, tuple):
                loss, loss_dict = result
            else:
                loss = result
                loss_dict = {}

            with accelerator.accumulate(policy_rf):
                # optimizer.zero_grad() is handled by accelerator.accumulate
                # Phase 63/SOTA: Decomposed loss return
                result = MaxRL_wrapper.loss(
                    batch_win, batch_lose,
                    reward_win=reward_win, reward_lose=reward_lose,
                    valid_mask=valid_mask
                )
                
                loss_total, loss_dict, individual_losses = result
                
                if adaptive_loss_fn is not None:
                    loss_total, weighted_dict = adaptive_loss_fn(individual_losses)
                    loss_dict.update(weighted_dict)

                if torch.isnan(loss_total):
                    if accelerator.sync_gradients:
                        accelerator.skip_step()
                    continue

                if pcgrad_optimizer is not None:
                    # PCGrad handles backward internally
                    pcgrad_optimizer.pcgrad_backward(list(individual_losses.values()))
                else:
                    accelerator.backward(loss_total)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy_rf.parameters(), config.grad_clip_norm)

                if pcgrad_optimizer is not None:
                    pcgrad_optimizer.step()
                else:
                    optimizer.step()
                    
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if ema_model is not None:
                ema_model.update(policy_rf)

            loss_meter.update(loss.item(), batch_win.num_graphs)

            if accelerator.is_local_main_process:
                # Log the better reward (target) and raw mean
                target_r = r_max[valid_mask].mean().item() if valid_count > 0 else 0.0
                rdkit_fail = batch_win.num_graphs - valid_count
                log_data = {
                    "loss": loss_meter.val, 
                    "avg_loss": loss_meter.avg, 
                    "r_mu": target_r,
                    "r_valid": valid_ratio,
                    "n_fail": rdkit_fail
                }
                if loss_dict:
                    log_data["L_MaxRL"] = loss_dict.get('MaxRL', 0.0)
                    log_data["L_clip"] = loss_dict.get('clip', 0.0)
                # Phase 63 Debug: Monitoring for zero signal
                if not hasattr(step_logger, 'zero_reward_streak'):
                    step_logger.zero_reward_streak = 0
                
                if target_r == 0.0:
                    step_logger.zero_reward_streak += 1
                else:
                    step_logger.zero_reward_streak = 0
                
                if step_logger.zero_reward_streak == 10:
                    accelerator.print("\n⚠️ [CRITICAL] Reward signal (r_mu) is consistently 0.0! Possible RDKit reconstruction failure.")
                
                step_logger.log(i + 1, log_data)

                if accelerator.is_main_process:
                    csv_log_entry = {
                        "epoch": epoch + 1,
                        "step": i + 1,
                        "loss": loss.item(),
                        "reward_win": target_r,
                        "reward_lose": (reward_lose.mean().item() if 'reward_lose' in locals() else 0.0),
                        "lr": optimizer.param_groups[0]['lr']
                    }
                    csv_log_entry.update(loss_dict)
                    csv_logger.log(csv_log_entry)

                if args.use_wandb and i % 10 == 0:
                    wandb_data = {
                        "train/loss": loss_meter.avg,
                        "train/reward_mean": target_r,
                        "train/reward_max": r_max.max().item() if 'r_max' in locals() else 0.0,
                        "train/lr": optimizer.param_groups[0]['lr'],
                    }
                    wandb_data.update({f"train/{k}": v for k, v in loss_dict.items()})
                    accelerator.log(wandb_data, step=epoch * len(dataloader) + i)

        accelerator.print(f"Epoch {effective_epoch} — Avg Loss: {loss_meter.avg:.4f}")

        # ─── Save Checkpoints (FIXED: Handles torch.compile wrapping) ───
        if accelerator.is_main_process:
            os.makedirs("checkpoints_MaxRL", exist_ok=True)
            
            # 1. 完整解包 accelerator wrapper
            unwrapped = accelerator.unwrap_model(policy_rf)
            
            # 2. 處理 torch.compile 造成的 _orig_mod 包裹
            if hasattr(unwrapped, "_orig_mod"):
                # 如果模型被編譯，真實參數藏在 _orig_mod 中
                save_state_dict = unwrapped._orig_mod.state_dict()
            elif hasattr(unwrapped, "backbone") and hasattr(unwrapped.backbone, "_orig_mod"):
                # 針對只有 backbone 被編譯的情況
                # 這裡最安全的方式是存整個 unwrapped 的 state_dict，雖然會有 _orig_mod 前綴
                # 但重新載入時 load_state_dict 通常能處理 (或使用 strict=False)
                # 更精確的做法是手動提取，但為求穩健，這裡直接存 unwrapped
                save_state_dict = unwrapped.state_dict()
            else:
                # 標準情況
                save_state_dict = unwrapped.state_dict()

            # 3. 驗證參數大小 (Debug: 確保不是空殼 1.2MB)
            param_count = sum(p.numel() for p in unwrapped.parameters())
            accelerator.print(f"💾 Saving Checkpoint (Epoch {effective_epoch}) | Params: {param_count/1e6:.2f}M")

            torch.save(save_state_dict, f"checkpoints_MaxRL/MaxRL_epoch{effective_epoch}.pt")
            torch.save(save_state_dict, "checkpoints_MaxRL/MaxRL_final_spot.pt")

            if ema_model is not None:
                torch.save(ema_model.state_dict(), "checkpoints_MaxRL/MaxRL_final_ema.pt")

    if args.use_wandb:
        accelerator.end_training()

    accelerator.print("✅ MaxRL Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxFlow Alignment Training")
    parser.add_argument("--rf_checkpoint", type=str, default="checkpoints/rf_last.pt")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--quant_combat", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--config", type=str, default="MaxRL_config.json")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Fine-tuning stage (1: Head, 2: Full)")
    parser.add_argument("--epoch_offset", type=int, default=0, help="Starting epoch number offset for logging/saving")
    parser.add_argument("--use_pcgrad", action="store_true", help="Use PCGrad gradient surgery")
    parser.add_argument("--use_adaptive_loss", action="store_true", help="Use Adaptive Loss weighting")
    parser.add_argument("--use_maxrl", action="store_true", help="Phase 11: Use MaxRL (weighted likelihood) instead of MaxRL")
    args = parser.parse_args()
    
    # Propagate override to config
    if args.use_maxrl:
        # Load config to modify it if loading from file
        config_path = getattr(args, 'config', 'MaxRL_config.json')
        config = MaxRLConfig.load(config_path) if os.path.exists(config_path) else MaxRLConfig()
        config.use_maxrl = True
        # We re-assign to args.config object if possible, but train_MaxRL loads it internally.
        # So we must rely on the fact that train_MaxRL uses args properties to override config fields?
        # Looking at train_MaxRL, it does: config.lr = getattr(args, 'lr', config.lr)
        # We should add similar logic for use_maxrl in train_MaxRL function.
        
    train_maxrl(args)
