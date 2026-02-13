# maxflow/optimization/adaptive_loss.py
"""
[SOTA Phase 6] Homoscedastic Uncertainty Loss Wrapper.
Automatically balances multiple loss components using learnable precision parameters.
Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018).
"""

import torch
import torch.nn as nn

class AdaptiveLossWrapper(nn.Module):
    """
    Automatically balances Flow, Topo, Chiral, and Physical losses.
    """
    def __init__(self, num_tasks=5, task_names=None):
        super().__init__()
        # Initialize log_vars to 0 (sigma = 1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]

    def forward(self, losses_dict):
        """
        Calculates the weighted total loss.
        losses_dict: Dictionary mapping task_names to individual loss tensors.
        """
        total_loss = 0.0
        weighted_losses = {}
        
        for i, name in enumerate(self.task_names):
            if name in losses_dict:
                # precision = exp(-log_var) = 1/sigma^2
                precision = torch.exp(-self.log_vars[i])
                diff_loss = losses_dict[name]
                
                # Weighted loss + regularization term (log sigma)
                task_loss = precision * diff_loss + self.log_vars[i]
                total_loss += task_loss
                weighted_losses[f"weighted_{name}"] = task_loss.item()
                
        return total_loss, weighted_losses

    def get_sigmas(self):
        """Returns the current learned standard deviations for each task."""
        with torch.no_grad():
            return {name: torch.exp(self.log_vars[i] / 2).item() 
                    for i, name in enumerate(self.task_names)}
