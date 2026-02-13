
import torch
import torch.nn.functional as F

def maxrl_objective(log_probs, rewards, baseline_rewards):
    """
    Implements MaxRL (Maximum Likelihood RL) Objective.
    Paper: arXiv:2602.02710
    
    Formula: L = - E [ (R / E[R]) * log P(x) ]
    
    Args:
        log_probs: (B,) Log probability of the generated trajectory.
        rewards: (B,) Calculated reward (Vina + QED + Constraints).
        baseline_rewards: (B,) Moving average of rewards (baseline).
    """
    # 1. Importance Weight calculation
    # Epsilon prevents division by zero.
    # We clip the weight to prevent gradient explosion from very low baselines.
    epsilon = 1e-6
    weights = rewards / (baseline_rewards + epsilon)
    weights = torch.clamp(weights, min=0.0, max=10.0) # Stability clip
    
    # 2. Policy Gradient Term
    # We detach weights because we don't differentiate through the reward function itself here
    loss = -torch.mean(weights.detach() * log_probs)
    
    return loss
