"""
Integrated Training with All New Optimizations
Combines: Adaptive Loss Weighting, Smooth PSA Constraint, PCGrad Gradient Surgery
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import numpy as np

# Import all optimization modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.adaptive_loss import AdaptiveLossWeighting, MultiTaskLossWrapper
from models.smooth_psa import SmoothPSARegularizer, PSAConstraintWrapper, PSAAnalyzer
from models.pcgrad import PCGrad, MultiObjectivePCGrad


class IntegratedReflowTrainer:
    """
    Integrated trainer combining all new optimizations.
    
    Features:
    - Adaptive loss weighting (Homoscedastic Uncertainty)
    - Smooth PSA constraint (Soft Gaussian Potential)
    - PCGrad gradient surgery for multi-objective optimization
    - Comprehensive monitoring and analysis
    """
    
    def __init__(self, model, optimizer, config: Dict):
        """
        Initialize integrated trainer.
        
        Args:
            model: Model to train
            optimizer: Base optimizer
            config: Configuration dictionary
        """
        self.model = model
        self.base_optimizer = optimizer
        self.config = config
        
        # Initialize components
        self._init_adaptive_loss()
        self._init_psa_constraint()
        self._init_pcgrad()
        self._init_logging()
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.best_metrics = {}
        
    def _init_adaptive_loss(self):
        """Initialize adaptive loss weighting."""
        # Define loss components
        self.loss_components = {
            'consistency': {'weight': 1.0, 'initial_log_sigma': -1.0},
            'psa': {'weight': 1.0, 'initial_log_sigma': -1.0},
            'quality': {'weight': 1.0, 'initial_log_sigma': -1.0},
            'uncertainty': {'weight': 1.0, 'initial_log_sigma': -1.0}
        }
        
        # Initialize adaptive loss weighting
        self.adaptive_loss = AdaptiveLossWeighting(
            num_losses=len(self.loss_components),
            initial_log_sigma=self.config.get('initial_log_sigma', -1.0)
        )
        
        # Multi-task loss wrapper
        self.multi_task_wrapper = MultiTaskLossWrapper(
            num_tasks=len(self.loss_components)
        )
        
    def _init_psa_constraint(self):
        """Initialize PSA constraint."""
        psa_config = self.config.get('psa_constraint', {})
        
        self.psa_regularizer = SmoothPSARegularizer(
            target_psa=psa_config.get('target_psa', 75.0),
            sigma=psa_config.get('sigma', 15.0),
            weight=psa_config.get('weight', 1.0)
        )
        
        self.psa_wrapper = PSAConstraintWrapper(
            target_psa=psa_config.get('target_psa', 75.0),
            sigma=psa_config.get('sigma', 15.0),
            weight=psa_config.get('weight', 1.0)
        )
        
        self.psa_analyzer = PSAAnalyzer(
            target_psa=psa_config.get('target_psa', 75.0),
            sigma=psa_config.get('sigma', 15.0)
        )
        
    def _init_pcgrad(self):
        """Initialize PCGrad."""
        pcgrad_config = self.config.get('pcgrad', {})
        
        self.pc_grad = PCGrad(
            reduction=pcgrad_config.get('reduction', 'mean')
        )
        
        # Define objectives for PCGrad
        self.objectives = ['consistency', 'psa', 'quality', 'uncertainty']
        
        self.multi_objective_pcgrad = MultiObjectivePCGrad(
            objectives=self.objectives,
            pc_grad=self.pc_grad
        )
        
        # PCGrad optimizer wrapper (removed for now - use base optimizer directly)
        
    def _init_logging(self):
        """Initialize logging."""
        self.logger = logging.getLogger(__name__)
        
        # Create metrics directory
        self.metrics_dir = Path(self.config.get('metrics_dir', 'metrics'))
        self.metrics_dir.mkdir(exist_ok=True)
        
    def train_step(self, batch: Dict) -> Dict:
        """
        Perform one integrated training step.
        
        Args:
            batch: Input batch
            
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Calculate individual losses
        losses = self._calculate_losses(outputs, batch)
        
        # Apply PCGrad to gradients before adaptive weighting
        self._apply_pcgrad_to_model(losses)
        
        # Apply adaptive loss weighting
        weighted_loss, adaptive_metrics = self.adaptive_loss.calculate_total_loss(
            list(losses.values())
        )
        
        # Update model
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        
        # Calculate PSA metrics
        psa_metrics = self._calculate_psa_metrics(outputs, batch)
        
        # Calculate conflict metrics
        conflict_metrics = self._calculate_conflict_metrics()
        
        # Combine all metrics
        metrics = {
            'step': self.step_count,
            'total_loss': weighted_loss.item(),
            **losses,
            **adaptive_metrics,
            **psa_metrics,
            **conflict_metrics
        }
        
        # Update training state
        self.step_count += 1
        
        return metrics
    
    def _calculate_losses(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Calculate individual loss components.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            losses: Dictionary of losses
        """
        losses = {}
        
        # Consistency loss
        losses['consistency'] = self._calculate_consistency_loss(outputs, batch)
        
        # PSA loss
        losses['psa'] = self._calculate_psa_loss(outputs, batch)
        
        # Quality loss
        losses['quality'] = self._calculate_quality_loss(outputs, batch)
        
        # Uncertainty loss
        losses['uncertainty'] = self._calculate_uncertainty_loss(outputs, batch)
        
        return losses
    
    def _calculate_consistency_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Calculate consistency loss."""
        # Implement consistency loss based on your model
        # This is a placeholder
        return torch.tensor(0.0, requires_grad=True)
    
    def _calculate_psa_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Calculate PSA loss."""
        # Extract PSA values from outputs
        psa_values = outputs.get('psa', torch.tensor(75.0))
        
        # Calculate PSA regularization loss
        psa_loss, _ = self.psa_regularizer(psa_values)
        
        return psa_loss
    
    def _calculate_quality_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Calculate quality loss."""
        # Implement quality loss based on your model
        # This is a placeholder
        return torch.tensor(0.0, requires_grad=True)
    
    def _calculate_uncertainty_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Calculate uncertainty loss."""
        # Implement uncertainty loss based on your model
        # This is a placeholder
        return torch.tensor(0.0, requires_grad=True)
    
    def _apply_pcgrad_to_model(self, losses: Dict[str, torch.Tensor]):
        """Apply PCGrad to model gradients."""
        # Ensure all losses require gradients
        losses_with_grad = {}
        for name, loss in losses.items():
            if not loss.requires_grad:
                losses_with_grad[name] = loss.requires_grad_()
            else:
                losses_with_grad[name] = loss
        
        # Calculate gradients for each loss separately
        task_gradients = []  # List of gradient lists, one per task
        
        for loss_name, loss_value in losses_with_grad.items():
            # Zero gradients
            self.base_optimizer.zero_grad()
            
            # Calculate gradient for this loss
            loss_value.backward(retain_graph=True)
            
            # Collect gradients for this task
            task_grad_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    task_grad_list.append(param.grad.clone())
                else:
                    task_grad_list.append(torch.zeros_like(param))
            
            task_gradients.append(task_grad_list)
            
            # Zero gradients for next iteration
            self.base_optimizer.zero_grad()
        
        # Apply PCGrad if we have multiple losses
        if len(task_gradients) > 1:
            # Flatten all gradients for PCGrad processing
            flattened_task_gradients = []
            
            # For each task, flatten all its parameters' gradients
            for task_idx in range(len(task_gradients)):
                task_flat_grads = []
                for param_grad in task_gradients[task_idx]:
                    task_flat_grads.append(param_grad.view(-1))
                
                # Concatenate all flattened gradients for this task
                task_flat_tensor = torch.cat(task_flat_grads)
                flattened_task_gradients.append(task_flat_tensor)
            
            # Apply PCGrad to the flattened gradients
            projected_flat_gradients = self.pc_grad.pc_gradient(flattened_task_gradients)
            
            # Unflatten the projected gradients back to original parameter shapes
            final_gradients = []
            for task_idx in range(len(projected_flat_gradients)):
                task_grads = []
                start_idx = 0
                
                # Unflatten each parameter's gradient
                for param_grad in task_gradients[task_idx]:
                    param_size = param_grad.numel()
                    param_shape = param_grad.shape
                    
                    # Extract the corresponding slice and reshape
                    flat_slice = projected_flat_gradients[task_idx][start_idx:start_idx + param_size]
                    task_grads.append(flat_slice.view(param_shape))
                    start_idx += param_size
                
                final_gradients.append(task_grads)
            
            # Use the first task's projected gradients (or average them if needed)
            # For now, we'll use the first task's gradients
            self.base_optimizer.zero_grad()
            for param, grad in zip(self.model.parameters(), final_gradients[0]):
                param.grad = grad
        else:
            # Single loss - just compute gradient normally
            self.base_optimizer.zero_grad()
            list(losses_with_grad.values())[0].backward()
    
    def _calculate_psa_metrics(self, outputs: Dict, batch: Dict) -> Dict:
        """Calculate PSA metrics."""
        # Extract PSA values
        psa_values = outputs.get('psa', torch.tensor(75.0))
        
        # Calculate PSA metrics
        psa_metrics = self.psa_analyzer.analyze_distribution(psa_values)
        
        # Track PSA progress
        self.psa_analyzer.track_progress(psa_values)
        
        return {f'psa_{k}': v for k, v in psa_metrics.items()}
    
    def _calculate_conflict_metrics(self) -> Dict:
        """Calculate conflict metrics."""
        # Get conflict analysis from PCGrad
        conflict_analysis = self.pc_grad.get_conflict_analysis()
        gradient_analysis = self.pc_grad.get_gradient_analysis()
        
        # Handle empty analysis gracefully
        if not conflict_analysis or not gradient_analysis:
            return {
                'conflicts_total': 0,
                'conflict_rate': 0.0,
                'avg_norm_ratio': 1.0,
                'avg_projection_ratio': 1.0
            }
        
        return {
            'conflicts_total': conflict_analysis.get('total_conflicts', 0),
            'conflict_rate': conflict_analysis.get('conflict_rate', 0.0),
            'avg_norm_ratio': gradient_analysis.get('avg_norm_ratio', 1.0),
            'avg_projection_ratio': gradient_analysis.get('avg_projection_ratio', 1.0)
        }
    
    def validate_step(self, batch: Dict) -> Dict:
        """
        Perform validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            metrics: Validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(batch)
            losses = self._calculate_losses(outputs, batch)
            
            # Calculate weighted loss
            weighted_loss, _ = self.adaptive_loss.calculate_total_loss(
                list(losses.values())
            )
            
            # Calculate PSA metrics
            psa_metrics = self._calculate_psa_metrics(outputs, batch)
            
            metrics = {
                'val_total_loss': weighted_loss.item(),
                **{f'val_{k}': v for k, v in losses.items()},
                **{f'val_{k}': v for k, v in psa_metrics.items()}
            }
        
        return metrics
    
    def save_checkpoint(self, filepath: str, metrics: Dict = None):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            metrics: Current metrics
        """
        checkpoint = {
            'step': self.step_count,
            'epoch': self.epoch_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.base_optimizer.state_dict(),
            'adaptive_loss_state_dict': self.adaptive_loss.state_dict(),
            'config': self.config,
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint
            
        Returns:
            checkpoint_data: Loaded checkpoint data
        """
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.adaptive_loss.load_state_dict(checkpoint['adaptive_loss_state_dict'])
        
        self.step_count = checkpoint['step']
        self.epoch_count = checkpoint['epoch']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        
        return checkpoint
    
    def get_training_summary(self) -> Dict:
        """
        Get comprehensive training summary.
        
        Returns:
            summary: Training summary
        """
        # Get conflict analysis
        conflict_analysis = self.pc_grad.get_conflict_analysis()
        gradient_analysis = self.pc_grad.get_gradient_analysis()
        
        # Get PSA analysis
        psa_progress = self.psa_analyzer.get_progress_metrics()
        
        # Get adaptive loss weights
        current_weights = self.adaptive_loss.get_weights()
        
        summary = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'conflict_analysis': conflict_analysis,
            'gradient_analysis': gradient_analysis,
            'psa_progress': psa_progress,
            'adaptive_weights': current_weights,
            'best_metrics': self.best_metrics
        }
        
        return summary
    
    def plot_training_progress(self):
        """Plot training progress."""
        # This would implement visualization of training progress
        # Including PSA distribution, conflict rates, loss weights, etc.
        pass


class IntegratedTrainingConfig:
    """
    Configuration for integrated training.
    """
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration."""
        return {
            # Adaptive loss configuration
            'adaptive_loss': {
                'initial_log_sigma': -1.0,
                'num_losses': 4
            },
            
            # PSA constraint configuration
            'psa_constraint': {
                'target_psa': 75.0,
                'sigma': 15.0,
                'weight': 1.0
            },
            
            # PCGrad configuration
            'pcgrad': {
                'reduction': 'mean'
            },
            
            # Training configuration
            'training': {
                'max_steps': 100000,
                'validate_every': 1000,
                'save_every': 5000,
                'log_every': 100
            },
            
            # Paths
            'metrics_dir': 'metrics',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs'
        }
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            is_valid: Whether configuration is valid
        """
        required_keys = [
            'adaptive_loss', 'psa_constraint', 'pcgrad', 'training'
        ]
        
        for key in required_keys:
            if key not in config:
                return False
        
        return True


# Utility functions
def create_integrated_trainer(model, optimizer, config: Dict) -> IntegratedReflowTrainer:
    """
    Create integrated trainer with all optimizations.
    
    Args:
        model: Model to train
        optimizer: Base optimizer
        config: Configuration
        
    Returns:
        trainer: Integrated trainer
    """
    return IntegratedReflowTrainer(model, optimizer, config)


def train_with_integrated_optimizations(model, train_loader, val_loader, config: Dict):
    """
    Train model with all integrated optimizations.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        
    Returns:
        trainer: Trained trainer
    """
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
    
    # Create integrated trainer
    trainer = create_integrated_trainer(model, optimizer, config)
    
    # Training loop
    for epoch in range(config.get('num_epochs', 100)):
        # Training phase
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            metrics = trainer.train_step(batch)
            
            # Log metrics
            if batch_idx % config.get('log_every', 100) == 0:
                print(f"Epoch {epoch}, Step {batch_idx}: {metrics}")
        
        # Validation phase
        model.eval()
        val_metrics = []
        for batch in val_loader:
            val_metric = trainer.validate_step(batch)
            val_metrics.append(val_metric)
        
        # Aggregate validation metrics
        avg_val_metrics = {}
        for key in val_metrics[0].keys():
            # Handle tensors that might require gradients
            values = []
            for m in val_metrics:
                val = m[key]
                if torch.is_tensor(val):
                    val = val.detach().cpu()
                    if val.numel() == 1:
                        val = val.item()
                    else:
                        val = val.numpy()
                values.append(val)
            avg_val_metrics[key] = np.mean(values)
        
        print(f"Validation - Epoch {epoch}: {avg_val_metrics}")
        
        # Save checkpoint
        if epoch % config.get('save_every', 10) == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(checkpoint_path, avg_val_metrics)
    
    return trainer
