"""
Adaptive Loss Weighting Implementation
Implements Homoscedastic Uncertainty Weighting for automatic loss balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class AdaptiveLossWeighting(nn.Module):
    """
    Homoscedastic Uncertainty Weighting for automatic loss balancing.
    
    Implements:
    - Learnable uncertainty parameters for each loss term
    - Automatic weighting based on task uncertainty
    - Total loss with uncertainty regularization
    """
    
    def __init__(self, num_losses: int, initial_log_sigma: float = -1.0):
        """
        Initialize adaptive loss weighting.
        
        Args:
            num_losses: Number of loss terms to balance
            initial_log_sigma: Initial log uncertainty value
        """
        super().__init__()
        self.num_losses = num_losses
        self.log_sigma = nn.Parameter(torch.full((num_losses,), initial_log_sigma))
        
    def get_weights(self) -> torch.Tensor:
        """
        Get current loss weights based on learned uncertainties.
        
        Returns:
            weights: Tensor of loss weights
        """
        return 1.0 / (2.0 * torch.exp(self.log_sigma) ** 2)
    
    def get_uncertainty_regularization(self) -> torch.Tensor:
        """
        Get uncertainty regularization term.
        
        Returns:
            regularization: Uncertainty regularization loss
        """
        return torch.sum(self.log_sigma)
    
    def calculate_total_loss(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate total loss with adaptive weighting.
        
        Args:
            losses: List of individual loss terms
            
        Returns:
            total_loss: Weighted total loss
            metrics: Dictionary of loss components
        """
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} losses, got {len(losses)}")
        
        # Ensure all losses are scalars
        scalar_losses = []
        for loss in losses:
            if loss.dim() > 0:
                scalar_losses.append(loss.mean())
            else:
                scalar_losses.append(loss)
        
        # Get weights and regularization
        weights = self.get_weights()
        uncertainty_reg = self.get_uncertainty_regularization()
        
        # Calculate weighted losses
        weighted_losses = []
        for i, loss in enumerate(scalar_losses):
            weighted_loss = weights[i] * loss
            weighted_losses.append(weighted_loss)
        
        # Calculate total loss
        total_loss = torch.sum(torch.stack(weighted_losses)) + uncertainty_reg
        
        # Prepare metrics
        metrics = {
            'total_loss': total_loss.item(),
            'uncertainty_regularization': uncertainty_reg.item(),
            'weights': weights.detach().cpu().numpy().tolist(),
            'log_sigma': self.log_sigma.detach().cpu().numpy().tolist()
        }
        
        for i, loss in enumerate(losses):
            metrics[f'loss_{i}'] = loss.item()
            metrics[f'weighted_loss_{i}'] = weighted_losses[i].item()
        
        return total_loss, metrics
    
    def get_loss_contributions(self, losses: List[torch.Tensor]) -> Dict:
        """
        Get contribution of each loss to the total loss.
        
        Args:
            losses: List of individual loss terms
            
        Returns:
            contributions: Dictionary of loss contributions
        """
        weights = self.get_weights()
        
        contributions = {}
        total_weighted = 0.0
        
        for i, loss in enumerate(losses):
            weighted = weights[i] * loss.item()
            contributions[f'loss_{i}_contribution'] = weighted
            total_weighted += weighted
        
        # Normalize contributions
        for key in contributions:
            contributions[key] /= total_weighted
        
        contributions['total_weighted'] = total_weighted
        
        return contributions


class MultiTaskLossWrapper(nn.Module):
    """
    Wrapper for multi-task learning with adaptive loss weighting.
    
    Automatically balances multiple loss terms based on their uncertainties.
    """
    
    def __init__(self, num_tasks: int, initial_log_sigma: float = -1.0):
        super().__init__()
        self.adaptive_loss = AdaptiveLossWeighting(num_tasks, initial_log_sigma)
        self.num_tasks = num_tasks
    
    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass - calculate total loss with adaptive weighting.
        
        Args:
            losses: List of individual loss terms
            
        Returns:
            total_loss: Weighted total loss
            metrics: Dictionary of loss components
        """
        return self.adaptive_loss.calculate_total_loss(losses)
    
    def get_metrics(self, losses: List[torch.Tensor]) -> Dict:
        """
        Get detailed metrics for monitoring.
        
        Args:
            losses: List of individual loss terms
            
        Returns:
            metrics: Dictionary of all metrics
        """
        total_loss, metrics = self.forward(losses)
        
        # Add loss contributions
        contributions = self.adaptive_loss.get_loss_contributions(losses)
        metrics.update(contributions)
        
        return metrics


# Example usage in training loop
class AdaptiveTrainingManager:
    """
    Training manager that uses adaptive loss weighting.
    """
    
    def __init__(self, model, optimizer, num_tasks: int, initial_log_sigma: float = -1.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_wrapper = MultiTaskLossWrapper(num_tasks, initial_log_sigma)
        
    def train_step(self, batch, loss_functions: List):
        """
        Perform one training step with adaptive loss weighting.
        
        Args:
            batch: Input batch
            loss_functions: List of loss functions
            
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Calculate individual losses
        losses = []
        for loss_fn in loss_functions:
            loss = loss_fn(outputs, batch)
            losses.append(loss)
        
        # Calculate total loss with adaptive weighting
        total_loss, metrics = self.loss_wrapper.forward(losses)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return metrics
    
    def get_loss_weights(self):
        """
        Get current loss weights.
        
        Returns:
            weights: Current loss weights
        """
        return self.loss_wrapper.adaptive_loss.get_weights().detach().cpu().numpy()
    
    def get_uncertainty_metrics(self):
        """
        Get uncertainty-related metrics.
        
        Returns:
            metrics: Uncertainty metrics
        """
        weights = self.get_loss_weights()
        log_sigma = self.loss_wrapper.adaptive_loss.log_sigma.detach().cpu().numpy()
        
        return {
            'loss_weights': weights.tolist(),
            'log_uncertainty': log_sigma.tolist(),
            'uncertainty_values': np.exp(log_sigma).tolist()
        }


# Integration with existing training
class AdaptiveReflowTrainer:
    """
    Reflow trainer with adaptive loss weighting.
    """
    
    def __init__(self, model, optimizer, num_tasks: int = 3, initial_log_sigma: float = -1.0):
        self.model = model
        self.optimizer = optimizer
        self.training_manager = AdaptiveTrainingManager(
            model, optimizer, num_tasks, initial_log_sigma
        )
        
        # Define loss functions
        self.loss_functions = [
            self._flow_loss,
            self._clash_loss,
            self._MaxRL_loss
        ]
    
    def _flow_loss(self, outputs, batch):
        """Flow loss component"""
        # Implement your flow loss here
        return torch.tensor(0.0, device=batch['x_L'].device)
    
    def _clash_loss(self, outputs, batch):
        """Clash loss component"""
        # Implement your clash loss here
        return torch.tensor(0.0, device=batch['x_L'].device)
    
    def _MaxRL_loss(self, outputs, batch):
        """MaxRL loss component"""
        # Implement your MaxRL loss here
        return torch.tensor(0.0, device=batch['x_L'].device)
    
    def train_epoch(self, data_loader):
        """
        Train for one epoch with adaptive loss weighting.
        
        Args:
            data_loader: Data loader
            
        Returns:
            epoch_metrics: Epoch metrics
        """
        all_metrics = []
        
        for batch in data_loader:
            metrics = self.training_manager.train_step(batch, self.loss_functions)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        epoch_metrics = self._aggregate_metrics(all_metrics)
        
        return epoch_metrics
    
    def _aggregate_metrics(self, metrics_list):
        """
        Aggregate metrics from multiple batches.
        
        Args:
            metrics_list: List of metrics dictionaries
            
        Returns:
            aggregated: Aggregated metrics
        """
        aggregated = {}
        
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def get_current_weights(self):
        """
        Get current adaptive loss weights.
        
        Returns:
            weights: Current loss weights
        """
        return self.training_manager.get_loss_weights()


# Utility functions for monitoring
class LossWeightMonitor:
    """
    Monitor and visualize loss weight evolution.
    """
    
    def __init__(self):
        self.history = []
        
    def record_weights(self, weights):
        """
        Record current weights.
        
        Args:
            weights: Current loss weights
        """
        self.history.append(weights.copy())
    
    def get_weight_trends(self):
        """
        Get trends in loss weights over time.
        
        Returns:
            trends: Dictionary of weight trends
        """
        if not self.history:
            return {}
        
        history = np.array(self.history)
        trends = {}
        
        for i in range(history.shape[1]):
            trends[f'weight_{i}'] = {
                'initial': history[0, i],
                'final': history[-1, i],
                'trend': 'increasing' if history[-1, i] > history[0, i] else 'decreasing'
            }
        
        return trends
    
    def plot_weight_evolution(self):
        """
        Plot weight evolution over time.
        
        Note: This requires matplotlib. If not available, returns data for plotting.
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.history:
                print("🔍 No data to plot")
                return
            
            history = np.array(self.history)
            
            plt.figure(figsize=(10, 6))
            for i in range(history.shape[1]):
                plt.plot(history[:, i], label=f'Weight {i}')
            
            plt.xlabel('Training Step')
            plt.ylabel('Weight Value')
            plt.title('Adaptive Loss Weight Evolution')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except ImportError:
            print("📊 Plotting requires matplotlib. Here's the data:")
            print(self.history)
