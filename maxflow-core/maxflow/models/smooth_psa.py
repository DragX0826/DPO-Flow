"""
Smooth PSA Constraint Implementation
Implements Soft Gaussian Potential for BBB permeability guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SmoothPSARegularizer(nn.Module):
    """
    Smooth PSA (Polar Surface Area) constraint using Gaussian potential.
    
    Implements:
    - Soft Gaussian potential for PSA regularization
    - Smooth gradients for better optimization
    - Clinical intuition-based BBB permeability guidance
    """
    
    def __init__(self, target_psa: float = 75.0, sigma: float = 15.0, weight: float = 1.0):
        """
        Initialize PSA regularizer.
        
        Args:
            target_psa: Target PSA value (default: 75.0)
            sigma: Standard deviation for Gaussian (default: 15.0)
            weight: Regularization weight (default: 1.0)
        """
        super().__init__()
        self.target_psa = target_psa
        self.sigma = sigma
        self.weight = weight
        self.register_buffer('target_psa_buffer', torch.tensor(target_psa))
        self.register_buffer('sigma_buffer', torch.tensor(sigma))
        self.register_buffer('weight_buffer', torch.tensor(weight))
    
    def forward(self, psa_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate PSA regularization loss.
        
        Args:
            psa_values: PSA values to regularize
            
        Returns:
            loss: PSA regularization loss
            metrics: Dictionary of PSA metrics
        """
        # Calculate Gaussian potential
        diff = psa_values - self.target_psa_buffer
        exponent = - (diff ** 2) / (2 * self.sigma_buffer ** 2)
        gaussian_potential = torch.exp(exponent)
        
        # Calculate loss (negative Gaussian for minimization)
        loss = -gaussian_potential * self.weight_buffer
        
        # Calculate metrics
        metrics = self._calculate_metrics(psa_values, gaussian_potential)
        
        # Return mean loss for scalar output
        return loss.mean(), metrics
    
    def _calculate_metrics(self, psa_values: torch.Tensor, gaussian_potential: torch.Tensor) -> Dict:
        """
        Calculate PSA-related metrics.
        
        Args:
            psa_values: PSA values
            gaussian_potential: Gaussian potential values
            
        Returns:
            metrics: Dictionary of PSA metrics
        """
        metrics = {
            'mean_psa': torch.mean(psa_values).item(),
            'std_psa': torch.std(psa_values).item(),
            'min_psa': torch.min(psa_values).item(),
            'max_psa': torch.max(psa_values).item(),
            'target_psa': self.target_psa_buffer.item(),
            'sigma': self.sigma_buffer.item(),
            'mean_gaussian': torch.mean(gaussian_potential).item(),
            'std_gaussian': torch.std(gaussian_potential).item(),
            'min_gaussian': torch.min(gaussian_potential).item(),
            'max_gaussian': torch.max(gaussian_potential).item()
        }
        
        # Calculate distance from target
        diff_from_target = psa_values - self.target_psa
        metrics['mean_diff'] = torch.mean(diff_from_target).item()
        metrics['std_diff'] = torch.std(diff_from_target).item()
        metrics['min_diff'] = torch.min(diff_from_target).item()
        metrics['max_diff'] = torch.max(diff_from_target).item()
        
        # Calculate percentage within acceptable range
        acceptable_range = (self.target_psa - 2 * self.sigma, self.target_psa + 2 * self.sigma)
        within_range = (psa_values >= acceptable_range[0]) & (psa_values <= acceptable_range[1])
        metrics['percentage_within_range'] = torch.mean(within_range.float()).item() * 100
        
        return metrics
    
    def get_gradient(self, psa_values: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient of PSA regularization.
        
        Args:
            psa_values: PSA values
            
        Returns:
            gradient: Gradient of PSA regularization
        """
        # Gradient of Gaussian potential
        diff = psa_values - self.target_psa
        exponent = - (diff ** 2) / (2 * self.sigma ** 2)
        
        # Gradient: d/dx[exp(-(x-μ)^2/(2σ^2))] = -(x-μ)/σ^2 * exp(...)
        gradient = -(diff / (self.sigma ** 2)) * torch.exp(exponent)
        
        return gradient * self.weight
    
    def get_hessian(self, psa_values: torch.Tensor) -> torch.Tensor:
        """
        Calculate Hessian of PSA regularization.
        
        Args:
            psa_values: PSA values
            
        Returns:
            hessian: Hessian matrix of PSA regularization
        """
        # Second derivative of Gaussian potential
        diff = psa_values - self.target_psa
        exponent = - (diff ** 2) / (2 * self.sigma ** 2)
        
        # Hessian: d^2/dx^2[exp(-(x-μ)^2/(2σ^2))] = (1/σ^2 - (x-μ)^2/σ^4) * exp(...)
        hessian = ((1 / (self.sigma ** 2)) - (diff ** 2 / (self.sigma ** 4))) * torch.exp(exponent)
        
        return hessian * self.weight
    
    def get_curvature(self, psa_values: torch.Tensor) -> torch.Tensor:
        """
        Calculate curvature of PSA regularization.
        
        Args:
            psa_values: PSA values
            
        Returns:
            curvature: Curvature of PSA regularization
        """
        # Curvature is the second derivative (Hessian for 1D)
        return self.get_hessian(psa_values)
    
    def get_information_matrix(self, psa_values: torch.Tensor) -> torch.Tensor:
        """
        Calculate Fisher information matrix for PSA regularization.
        
        Args:
            psa_values: PSA values
            
        Returns:
            info_matrix: Fisher information matrix
        """
        # For Gaussian potential, Fisher information is related to Hessian
        hessian = self.get_hessian(psa_values)
        return -hessian  # Negative Hessian is positive definite for convex functions
    
    def get_uncertainty(self, psa_values: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty in PSA regularization.
        
        Args:
            psa_values: PSA values
            
        Returns:
            uncertainty: Estimated uncertainty
        """
        # Uncertainty is inversely related to curvature
        curvature = self.get_curvature(psa_values)
        return 1.0 / (torch.abs(curvature) + 1e-8)
    
    def get_confidence(self, psa_values: torch.Tensor) -> torch.Tensor:
        """
        Calculate confidence in PSA regularization.
        
        Args:
            psa_values: PSA values
            
        Returns:
            confidence: Confidence score (0 to 1)
        """
        # Confidence is related to Gaussian potential
        diff = psa_values - self.target_psa
        exponent = - (diff ** 2) / (2 * self.sigma ** 2)
        gaussian_potential = torch.exp(exponent)
        
        # Normalize to [0, 1]
        return gaussian_potential / torch.max(gaussian_potential)
    
    def get_penalty_strength(self, psa_values: torch.Tensor) -> torch.Tensor:
        """
        Get adaptive penalty strength based on PSA values.
        
        Args:
            psa_values: PSA values
            
        Returns:
            penalty_strength: Adaptive penalty strength
        """
        # Stronger penalty when far from target
        diff = psa_values - self.target_psa
        distance = torch.abs(diff)
        
        # Linear increase in penalty strength
        max_distance = 3 * self.sigma
        normalized_distance = torch.clamp(distance / max_distance, 0.0, 1.0)
        
        return normalized_distance * self.weight
    
    def get_smoothness_metrics(self, psa_values: torch.Tensor) -> Dict:
        """
        Get smoothness-related metrics.
        
        Args:
            psa_values: PSA values
            
        Returns:
            metrics: Smoothness metrics
        """
        # Calculate smoothness-related metrics
        metrics = {}
        
        # Gradient smoothness
        gradient = self.get_gradient(psa_values)
        metrics['mean_gradient'] = torch.mean(torch.abs(gradient)).item()
        metrics['std_gradient'] = torch.std(gradient).item()
        metrics['max_gradient'] = torch.max(torch.abs(gradient)).item()
        
        # Hessian smoothness
        hessian = self.get_hessian(psa_values)
        metrics['mean_hessian'] = torch.mean(torch.abs(hessian)).item()
        metrics['std_hessian'] = torch.std(hessian).item()
        metrics['max_hessian'] = torch.max(torch.abs(hessian)).item()
        
        # Curvature
        curvature = self.get_curvature(psa_values)
        metrics['mean_curvature'] = torch.mean(torch.abs(curvature)).item()
        metrics['std_curvature'] = torch.std(curvature).item()
        metrics['max_curvature'] = torch.max(torch.abs(curvature)).item()
        
        return metrics


class PSAConstraintWrapper(nn.Module):
    """
    Wrapper for PSA constraint in multi-task learning.
    
    Integrates PSA regularization with other loss terms.
    """
    
    def __init__(self, target_psa: float = 75.0, sigma: float = 15.0, weight: float = 1.0):
        super().__init__()
        self.psa_regularizer = SmoothPSARegularizer(target_psa, sigma, weight)
    
    def forward(self, outputs, batch) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate PSA regularization loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            loss: PSA regularization loss
            metrics: PSA metrics
        """
        # Extract PSA values from outputs
        psa_values = self._extract_psa_values(outputs)
        
        # Calculate PSA regularization
        loss, metrics = self.psa_regularizer(psa_values)
        
        return loss, metrics
    
    def _extract_psa_values(self, outputs):
        """
        Extract PSA values from model outputs.
        
        Args:
            outputs: Model outputs
            
        Returns:
            psa_values: Extracted PSA values
        """
        # This should be implemented based on your model architecture
        # For now, return dummy values
        return torch.randn(outputs.shape[0], device=outputs.device) * 50 + 75


# Integration with existing training
class PSAConstrainedTrainer:
    """
    Trainer that incorporates PSA constraints.
    """
    
    def __init__(self, model, optimizer, target_psa: float = 75.0, 
                 sigma: float = 15.0, psa_weight: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.psa_wrapper = PSAConstraintWrapper(target_psa, sigma, psa_weight)
        
        # Other loss components would be added here
        self.loss_components = [self.psa_wrapper]
    
    def train_step(self, batch):
        """
        Perform one training step with PSA constraint.
        
        Args:
            batch: Input batch
            
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Calculate individual losses
        total_loss = 0.0
        all_metrics = {}
        
        for loss_component in self.loss_components:
            loss, metrics = loss_component(outputs, batch)
            total_loss += loss
            all_metrics.update(metrics)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return all_metrics
    
    def get_psa_metrics(self, batch):
        """
        Get PSA-related metrics for a batch.
        
        Args:
            batch: Input batch
            
        Returns:
            metrics: PSA metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(batch)
            loss, metrics = self.psa_wrapper(outputs, batch)
        
        return metrics


# Utility functions for PSA analysis
class PSAAnalyzer:
    """
    Analyze PSA distribution and constraint effectiveness.
    """
    
    def __init__(self, target_psa: float = 75.0, sigma: float = 15.0):
        self.target_psa = target_psa
        self.sigma = sigma
        self.history = []
    
    def analyze_distribution(self, psa_values: torch.Tensor) -> Dict:
        """
        Analyze PSA value distribution.
        
        Args:
            psa_values: PSA values to analyze
            
        Returns:
            analysis: Distribution analysis
        """
        analysis = {
            'mean': torch.mean(psa_values).item(),
            'std': torch.std(psa_values).item(),
            'min': torch.min(psa_values).item(),
            'max': torch.max(psa_values).item(),
            'median': torch.median(psa_values).item(),
            'percentile_25': torch.quantile(psa_values, 0.25).item(),
            'percentile_75': torch.quantile(psa_values, 0.75).item(),
            'iqr': torch.quantile(psa_values, 0.75).item() - torch.quantile(psa_values, 0.25).item()
        }
        
        # Calculate distance from target
        diff = psa_values - self.target_psa
        analysis['mean_diff'] = torch.mean(diff).item()
        analysis['std_diff'] = torch.std(diff).item()
        analysis['max_abs_diff'] = torch.max(torch.abs(diff)).item()
        
        # Calculate within acceptable range
        acceptable_range = (self.target_psa - 2 * self.sigma, self.target_psa + 2 * self.sigma)
        within_range = (psa_values >= acceptable_range[0]) & (psa_values <= acceptable_range[1])
        analysis['percentage_within_range'] = torch.mean(within_range.float()).item() * 100
        analysis['count_within_range'] = torch.sum(within_range).item()
        analysis['count_total'] = psa_values.shape[0]
        
        return analysis
    
    def track_progress(self, psa_values: torch.Tensor):
        """
        Track PSA values over time.
        
        Args:
            psa_values: PSA values to track
        """
        analysis = self.analyze_distribution(psa_values)
        self.history.append(analysis)
    
    def get_progress_metrics(self):
        """
        Get progress metrics over time.
        
        Returns:
            metrics: Progress metrics
        """
        if not self.history:
            return {}
        
        # Calculate trends
        initial = self.history[0]
        final = self.history[-1]
        
        metrics = {
            'initial_mean_psa': initial['mean'],
            'final_mean_psa': final['mean'],
            'initial_std_psa': initial['std'],
            'final_std_psa': final['std'],
            'initial_within_range': initial['percentage_within_range'],
            'final_within_range': final['percentage_within_range'],
            'mean_diff_from_target': final['mean_diff'],
            'max_abs_diff_from_target': final['max_abs_diff']
        }
        
        # Calculate improvements
        metrics['improvement_within_range'] = final['percentage_within_range'] - initial['percentage_within_range']
        metrics['improvement_mean_psa'] = abs(final['mean'] - self.target_psa) - abs(initial['mean'] - self.target_psa)
        
        return metrics
    
    def plot_psa_distribution(self, psa_values: torch.Tensor, title: str = "PSA Distribution"):
        """
        Plot PSA value distribution.
        
        Args:
            psa_values: PSA values to plot
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.histplot(psa_values.cpu().numpy(), kde=True, bins=50)
            
            # Add target line
            plt.axvline(self.target_psa, color='red', linestyle='--', label=f'Target PSA: {self.target_psa}')
            
            # Add acceptable range
            acceptable_range = (self.target_psa - 2 * self.sigma, self.target_psa + 2 * self.sigma)
            plt.axvspan(acceptable_range[0], acceptable_range[1], color='green', alpha=0.1, label='Acceptable Range')
            
            plt.xlabel('PSA Value')
            plt.ylabel('Frequency')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("📊 Plotting requires matplotlib and seaborn. Here's the data:")
            print(self.analyze_distribution(psa_values))
