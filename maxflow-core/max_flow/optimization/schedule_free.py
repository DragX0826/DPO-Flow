# max_flow/optimization/schedule_free.py
import torch
import torch.optim
import math

class HybridSFOAdamW(torch.optim.Optimizer):
    """
    Hybrid Schedule-Free AdamW Optimizer.
    Combines the performance of Schedule-Free optimization with the stability 
    of standard AdamW through automatic divergence detection.
    
    If 'stable_mode' is triggered (via gradient spikes), it falls back to 
    conservative AdamW updates.
    """
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_steps=0, r=0.0, stability_threshold=100.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, r=r, warmup_steps=warmup_steps,
                        stability_threshold=stability_threshold)
        super().__init__(params, defaults)
        self.stable_mode = False

    def eval(self):
        """Switch to primal sequence (z) for inference (like Polyak averaging)."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'z' in state:
                    state['y'] = p.data.clone() # Save current interp point
                    p.data.copy_(state['z'])

    def train(self):
        """Switch back to interpolation point (y) for training."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'y' in state:
                    p.data.copy_(state['y'])
                    del state['y']

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            warmup_steps = group['warmup_steps']
            stability_threshold = group.get('stability_threshold', 100.0)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                # Stability Tracking
                grad_norm = grad.norm().item()
                if grad_norm > stability_threshold and not self.stable_mode:
                    self.stable_mode = True
                    # In a real system, we'd log this via a callback or print
                    # print(f"⚠️ Stability Triggered: Grad Norm {grad_norm:.2f} > {stability_threshold}")
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['z'] = p.clone() # Primal sequence (Momentum accumulator)
                    state['exp_avg_sq'] = torch.zeros_like(p) # 2nd moment
                    state['k'] = 0
                    
                state['k'] += 1
                k = state['k']

                z = state['z']
                exp_avg_sq = state['exp_avg_sq']

                # 1. Update 2nd moment (Adam-style)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                # 2. Schedule-Free Logic
                # "Schedule-free" learning rate modulation
                # Effective step size adaptation
                
                # Simple implementation following the core concept:
                # y (current p) = (1-ck)x + ck*z
                # We update z -> z - lr * grad / denom
                # Then update x -> mixed
                
                # Note: Official implementation is more complex with c_k schedules.
                # Simplified "Quant Combat" version for stability:
                
                # Apply Weight Decay to Z
                if weight_decay != 0:
                    z.mul_(1 - lr * weight_decay)

                # Update Z (Momentum/Primal)
                z.addcdiv_(grad, denom, value=-lr)

                # Update X (Weights/Dual) - "The Anchor"
                if self.stable_mode:
                    # FALLBACK: Standard AdamW-style update on p directly
                    # ignoring the primal interpolation to ensure safety.
                    p.data.addcdiv_(grad, denom, value=-lr)
                else:
                    # QUANT MODE: Schedule-Free Interpolation
                    # rho decreases over time (like 1/k)
                    rho = 1.0 / (k ** 0.6 + 1.0) # Decay rate
                    p.data.lerp_(z, rho)

        return loss
