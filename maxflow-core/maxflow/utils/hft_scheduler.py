# maxflow/utils/hft_scheduler.py
import torch
import numpy as np

class AlmgrenChrissScheduler:
    """
    SOTA Phase 50: Optimal Trajectory Scheduling (HFT Inspired).
    Balances 'Market Impact' (Geometry Distortion) vs 'Volatility Risk' (Convergence).
    """
    def __init__(self, total_steps, initial_gamma=1.5, risk_aversion=0.1):
        self.total_steps = total_steps
        self.initial_gamma = initial_gamma
        self.k = risk_aversion # Risk aversion parameter
        
    def get_adaptive_gamma(self, t_curr, grad_volatility):
        """
        Adaptive guidance scale based on the 'Liquidity' of the energy landscape.
        If volatility is high, reduce speed (gamma) to avoid crashing.
        """
        # Base schedule: 4t(1-t)
        base_gamma = self.initial_gamma * (4 * t_curr * (1 - t_curr))
        
        # Adaptive adjustment: Inverse to volatility
        # gamma_adj = gamma / (1 + k * sigma)
        adj_factor = 1.0 / (1.0 + self.k * grad_volatility + 1e-6)
        
        return base_gamma * adj_factor

class KalmanGradientFilter:
    """
    SOTA Phase 50: Signal Denoising (HFT State-Space Modeling).
    Tracks the 'True' Physical Gradient by filtering micro-noise.
    """
    def __init__(self, dim=3, Q=1e-4, R=1e-2):
        self.Q = Q # Process noise covariance
        self.R = R # Measurement noise covariance
        self.state = None # Estimated true gradient
        self.P = None    # Error covariance
        
    def update(self, observed_grad):
        """
        Update the estimated gradient based on new noisy observation.
        observed_grad: (N, 3) tensor
        """
        if self.state is None:
            self.state = observed_grad
            # Use scalar error covariance to handle any N
            self.P = torch.tensor(1.0, device=observed_grad.device)
            return self.state
            
        # Prediction
        P_prior = self.P + self.Q
        
        # Gain (Scalar)
        K = P_prior / (P_prior + self.R)
        
        # Update with broadcasting: state is (N, 3), K is scalar
        self.state = self.state + K * (observed_grad - self.state)
        self.P = (1 - K) * P_prior
        
        return self.state

def calculate_grad_volatility(grad_history, window=5):
    """
    Computes the historical volatility of the gradient vector.
    grad_history: List of (N, 3) tensors
    """
    if len(grad_history) < 2: return 0.0
    
    # Take last N steps
    recent = grad_history[-window:]
    norms = torch.stack([g.norm() for g in recent])
    
    # Volatility = Standard Deviation of norms
    return torch.std(norms).item()
