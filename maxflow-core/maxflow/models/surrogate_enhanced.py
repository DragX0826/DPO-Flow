"""
Enhanced GNN Proxy with Uncertainty Estimation and Ensemble Support
Implements multi-model ensemble for uncertainty-aware reward prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from maxflow.models.backbone import CrossGVP
from maxflow.utils.quality_assessment import calculate_uncertainty_ensemble


class GNNProxyEnsemble(nn.Module):
    """
    Ensemble of GNN Proxy models for uncertainty-aware reward prediction.
    
    Supports:
    - Multiple model ensemble
    - Uncertainty estimation
    - Weighted predictions
    - Out-of-distribution detection
    """
    def __init__(self, node_in_dim=58, hidden_dim=64, num_models=3):
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            GNNProxy(node_in_dim, hidden_dim) for _ in range(num_models)
        ])
        
        # Learnable weights for ensemble (if desired)
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
        # Uncertainty estimation parameters
        self.uncertainty_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, data):
        """
        Forward pass through all ensemble models.
        
        Args:
            data: Input data batch
            
        Returns:
            predictions: List of predictions from each model
            uncertainty: Estimated uncertainty
            ensemble_prediction: Weighted ensemble prediction
        """
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            with torch.no_grad():
                pred = model(data)
                predictions.append(pred)
        
        # Calculate uncertainty
        uncertainty = calculate_uncertainty_ensemble([list(p.values()) for p in predictions])
        
        # Calculate weighted ensemble prediction
        ensemble_prediction = self._calculate_ensemble_prediction(predictions)
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'ensemble_prediction': ensemble_prediction
        }
    
    def _calculate_ensemble_prediction(self, predictions):
        """
        Calculate weighted ensemble prediction.
        
        Args:
            predictions: List of predictions from each model
            
        Returns:
            ensemble_prediction: Weighted average prediction
        """
        # Normalize weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Initialize ensemble prediction
        ensemble_pred = {}
        
        # Get keys from first prediction
        keys = predictions[0].keys()
        
        # Initialize with zeros
        for key in keys:
            ensemble_pred[key] = torch.zeros_like(predictions[0][key])
        
        # Weighted sum
        for i, pred in enumerate(predictions):
            for key in keys:
                ensemble_pred[key] += weights[i] * pred[key]
        
        return ensemble_pred
    
    def is_uncertain(self, uncertainty):
        """
        Check if uncertainty exceeds threshold.
        
        Args:
            uncertainty: Estimated uncertainty value
            
        Returns:
            is_uncertain: Boolean indicating high uncertainty
        """
        return uncertainty > self.uncertainty_threshold.item()


class UncertaintyAwareRewardModel:
    """
    Reward model with uncertainty estimation for robust optimization.
    
    Implements:
    - Multi-model ensemble
    - Uncertainty-aware rewards
    - OOD detection
    - Confidence-based filtering
    """
    def __init__(
        self,
        checkpoint_paths=None,
        node_in_dim=58,
        hidden_dim=64,
        num_models=3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        uncertainty_penalty=0.5
    ):
        self.device = device
        self.uncertainty_penalty = uncertainty_penalty
        
        # Initialize ensemble model
        self.ensemble_model = GNNProxyEnsemble(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_models=num_models
        ).to(device)
        
        # Load checkpoints if provided
        if checkpoint_paths:
            self._load_checkpoints(checkpoint_paths)
        
        self.ensemble_model.eval()
    
    def _load_checkpoints(self, checkpoint_paths):
        """
        Load multiple model checkpoints into ensemble.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
        """
        for i, path in enumerate(checkpoint_paths):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.ensemble_model.models[i].load_state_dict(checkpoint)
                print(f"✅ Loaded checkpoint {path} into model {i}")
            except Exception as e:
                print(f"⚠️ Could not load checkpoint {path}: {e}")
    
    @torch.no_grad()
    def predict_reward(self, data_batch, weights=None):
        """
        Predict reward with uncertainty estimation.
        
        Args:
            data_batch: Input data batch
            weights: Reward weights
            
        Returns:
            reward: Uncertainty-aware reward
            uncertainty: Estimated uncertainty
            is_uncertain: Boolean indicating high uncertainty
            confidence: Confidence score
        """
        if weights is None:
            weights = {
                'qed': 3.0, 
                'sa': 1.0, 
                'affinity': 0.1, 
                'tpsa_penalty': -0.1
            }
        
        # Get ensemble predictions
        results = self.ensemble_model(data_batch.to(self.device))
        predictions = results['predictions']
        uncertainty = results['uncertainty']
        ensemble_pred = results['ensemble_prediction']
        
        # Calculate base reward
        norm_sa = (10.0 - ensemble_pred['sa']) / 9.0
        
        # TPSA Penalty logic
        tp = ensemble_pred['tpsa']
        t_penalty = torch.zeros_like(tp)
        mask_low = tp < 60
        mask_high = tp > 90
        t_penalty[mask_low] = 60 - tp[mask_low]
        t_penalty[mask_high] = tp[mask_high] - 90
        
        base_reward = (
            weights.get('qed', 3.0) * ensemble_pred['qed'] +
            weights.get('sa', 1.0) * norm_sa +
            weights.get('affinity', 0.1) * ensemble_pred['affinity'] +
            weights.get('tpsa_penalty', -0.1) * t_penalty
        )
        
        # Apply uncertainty penalty
        uncertainty_penalty = self.uncertainty_penalty * uncertainty
        reward = base_reward - uncertainty_penalty
        
        # Calculate confidence
        confidence = torch.sigmoid(10 * (1 - uncertainty))  # Higher for lower uncertainty
        
        return {
            'reward': reward,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'is_uncertain': self.ensemble_model.is_uncertain(uncertainty),
            'base_reward': base_reward,
            'uncertainty_penalty': uncertainty_penalty
        }
    
    def filter_high_confidence(self, results, confidence_threshold=0.7):
        """
        Filter results based on confidence threshold.
        
        Args:
            results: Prediction results
            confidence_threshold: Minimum confidence to accept
            
        Returns:
            filtered_results: Filtered results
            mask: Boolean mask of accepted samples
        """
        mask = results['confidence'] >= confidence_threshold
        filtered_results = {k: v[mask] for k, v in results.items() if torch.is_tensor(v)}
        return filtered_results, mask
    
    def get_uncertainty_analysis(self, data_batch):
        """
        Get detailed uncertainty analysis for a batch.
        
        Args:
            data_batch: Input data batch
            
        Returns:
            analysis: Dictionary with uncertainty metrics
        """
        results = self.ensemble_model(data_batch.to(self.device))
        
        # Calculate variance across models
        variances = []
        for key in results['predictions'][0].keys():
            values = torch.stack([p[key] for p in results['predictions']])
            variance = torch.var(values, dim=0).mean()
            variances.append(variance.item())
        
        return {
            'avg_uncertainty': results['uncertainty'].item(),
            'per_objective_variance': dict(zip(results['predictions'][0].keys(), variances)),
            'max_variance': max(variances),
            'min_variance': min(variances),
            'is_uncertain': results['uncertainty'] > self.ensemble_model.uncertainty_threshold.item()
        }


class MultiObjectiveUncertaintyReward:
    """
    Multi-objective reward with uncertainty awareness.
    
    Implements:
    - Weighted multi-objective scoring
    - Uncertainty-based reward adjustment
    - Confidence-aware filtering
    """
    def __init__(self, uncertainty_penalty=0.5, confidence_threshold=0.7):
        self.uncertainty_penalty = uncertainty_penalty
        self.confidence_threshold = confidence_threshold
        
    def calculate_reward(self, predictions, uncertainty, weights=None):
        """
        Calculate uncertainty-aware reward.
        
        Args:
            predictions: Dictionary of predictions
            uncertainty: Estimated uncertainty
            weights: Reward weights
            
        Returns:
            reward: Final reward
            confidence: Confidence score
        """
        if weights is None:
            weights = {
                'qed': 3.0, 
                'sa': 1.0, 
                'affinity': 0.1, 
                'tpsa_penalty': -0.1
            }
        
        # Calculate base reward
        norm_sa = (10.0 - predictions['sa']) / 9.0
        
        # TPSA Penalty logic
        tp = predictions['tpsa']
        t_penalty = torch.zeros_like(tp)
        mask_low = tp < 60
        mask_high = tp > 90
        t_penalty[mask_low] = 60 - tp[mask_low]
        t_penalty[mask_high] = tp[mask_high] - 90
        
        base_reward = (
            weights.get('qed', 3.0) * predictions['qed'] +
            weights.get('sa', 1.0) * norm_sa +
            weights.get('affinity', 0.1) * predictions['affinity'] +
            weights.get('tpsa_penalty', -0.1) * t_penalty
        )
        
        # Apply uncertainty penalty
        uncertainty_penalty = self.uncertainty_penalty * uncertainty
        reward = base_reward - uncertainty_penalty
        
        # Calculate confidence
        confidence = torch.sigmoid(10 * (1 - uncertainty))
        
        return {
            'reward': reward,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'is_confident': confidence >= self.confidence_threshold,
            'base_reward': base_reward,
            'uncertainty_penalty': uncertainty_penalty
        }


# Update the main SurrogateScorer to use the new ensemble model
class SurrogateScorer:
    """
    Enhanced Surrogate Scorer with Uncertainty Awareness.
    """
    def __init__(
        self,
        checkpoint_paths=None,
        node_in_dim=58,
        hidden_dim=64,
        num_models=3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        uncertainty_penalty=0.5
    ):
        self.device = device
        self.uncertainty_penalty = uncertainty_penalty
        
        # Initialize ensemble model
        self.model = UncertaintyAwareRewardModel(
            checkpoint_paths=checkpoint_paths,
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_models=num_models,
            device=device,
            uncertainty_penalty=uncertainty_penalty
        )
        
        self.model.ensemble_model.eval()
    
    @torch.no_grad()
    def predict_batch_reward(self, data_batch, weights=None):
        """
        Fast proxy for MultiObjectiveScorer.calculate_batch_reward with uncertainty.
        Latency: < 10ms (vs 2s for RDKit/Vina).
        """
        results = self.model.predict_reward(data_batch.to(self.device), weights)
        
        return results['reward'], results['is_uncertain']
    
    def get_uncertainty_analysis(self, data_batch):
        """
        Get uncertainty analysis for a batch.
        """
        return self.model.get_uncertainty_analysis(data_batch)
