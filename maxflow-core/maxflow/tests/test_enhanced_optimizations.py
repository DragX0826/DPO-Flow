"""
Test Suite for Enhanced Reflow and Uncertainty-Aware Optimization
Validates all implemented optimizations and ensures correctness.
"""

import unittest
import torch
import numpy as np
from maxflow.models.surrogate_enhanced import GNNProxyEnsemble, UncertaintyAwareRewardModel
from maxflow.utils.quality_assessment import calculate_molecule_quality, calculate_consistency
from maxflow.scripts.generate_reflow_data_enhanced import calculate_consistency as calc_traj_consistency


class TestQualityAssessment(unittest.TestCase):
    """
    Test quality assessment functions
    """
    
    def test_calculate_molecule_quality(self):
        """Test molecule quality calculation"""
        # Create dummy molecule tensor
        tensor = torch.randn(10, 6)  # 10 atoms, 6 features
        
        quality = calculate_molecule_quality(tensor)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
    
    def test_calculate_consistency(self):
        """Test consistency calculation"""
        # Create simple trajectory
        traj = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ])
        
        consistency = calculate_consistency(traj)
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
        self.assertGreater(consistency, 0.9)  # Should be very consistent
    
    def test_calculate_traj_consistency(self):
        """Test trajectory consistency calculation"""
        # Create trajectory with some noise
        traj = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.1, 0.9, 1.0],
            [2.0, 2.0, 2.0]
        ])
        
        consistency = calc_traj_consistency(traj)
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)


class TestGNNProxyEnsemble(unittest.TestCase):
    """
    Test GNN Proxy Ensemble functionality
    """
    
    def setUp(self):
        """Setup test environment"""
        self.ensemble = GNNProxyEnsemble(node_in_dim=6, hidden_dim=16, num_models=3)
        
        # Create dummy data
        self.data = {
            'x_L': torch.randn(5, 6),  # 5 atoms, 6 features
            'pos_L': torch.randn(5, 3),
            'x_P': torch.randn(50, 6),
            'pos_P': torch.randn(50, 3),
            'pocket_center': torch.randn(3)
        }
        self.batch = torch.utils.data.Batch(**self.data)
    
    def test_forward_pass(self):
        """Test forward pass through ensemble"""
        results = self.ensemble(self.batch)
        
        self.assertIn('predictions', results)
        self.assertIn('uncertainty', results)
        self.assertIn('ensemble_prediction', results)
        
        # Check predictions
        self.assertEqual(len(results['predictions']), 3)
        
        # Check uncertainty
        self.assertIsInstance(results['uncertainty'], float)
        self.assertGreaterEqual(results['uncertainty'], 0.0)
        self.assertLessEqual(results['uncertainty'], 1.0)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction calculation"""
        # Get predictions
        results = self.ensemble(self.batch)
        
        # Check ensemble prediction structure
        ensemble_pred = results['ensemble_prediction']
        self.assertIn('qed', ensemble_pred)
        self.assertIn('sa', ensemble_pred)
        self.assertIn('affinity', ensemble_pred)
        self.assertIn('tpsa', ensemble_pred)
        
        # Check tensor shapes
        for key in ensemble_pred:
            self.assertEqual(ensemble_pred[key].shape, (1,))
    
    def test_uncertainty_threshold(self):
        """Test uncertainty threshold functionality"""
        # Get predictions
        results = self.ensemble(self.batch)
        
        # Check uncertainty threshold
        is_uncertain = self.ensemble.is_uncertain(results['uncertainty'])
        self.assertIsInstance(is_uncertain, bool)


class TestUncertaintyAwareRewardModel(unittest.TestCase):
    """
    Test uncertainty-aware reward model
    """
    
    def setUp(self):
        """Setup test environment"""
        self.model = UncertaintyAwareRewardModel(
            node_in_dim=6,
            hidden_dim=16,
            num_models=3,
            uncertainty_penalty=0.5
        )
        
        # Create dummy data
        self.data = {
            'x_L': torch.randn(5, 6),  # 5 atoms, 6 features
            'pos_L': torch.randn(5, 3),
            'x_P': torch.randn(50, 6),
            'pos_P': torch.randn(50, 3),
            'pocket_center': torch.randn(3)
        }
        self.batch = torch.utils.data.Batch(**self.data)
    
    def test_predict_reward(self):
        """Test reward prediction with uncertainty"""
        results = self.model.predict_reward(self.batch)
        
        self.assertIn('reward', results)
        self.assertIn('uncertainty', results)
        self.assertIn('confidence', results)
        self.assertIn('is_uncertain', results)
        self.assertIn('base_reward', results)
        self.assertIn('uncertainty_penalty', results)
        
        # Check types
        self.assertIsInstance(results['reward'], torch.Tensor)
        self.assertIsInstance(results['uncertainty'], float)
        self.assertIsInstance(results['confidence'], torch.Tensor)
        self.assertIsInstance(results['is_uncertain'], bool)
    
    def test_filter_high_confidence(self):
        """Test high confidence filtering"""
        results = self.model.predict_reward(self.batch)
        filtered_results, mask = self.model.filter_high_confidence(results)
        
        self.assertIsInstance(filtered_results, dict)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.dtype, torch.bool)
    
    def test_get_uncertainty_analysis(self):
        """Test uncertainty analysis"""
        analysis = self.model.get_uncertainty_analysis(self.batch)
        
        self.assertIn('avg_uncertainty', analysis)
        self.assertIn('per_objective_variance', analysis)
        self.assertIn('max_variance', analysis)
        self.assertIn('min_variance', analysis)
        self.assertIn('is_uncertain', analysis)
        
        # Check types
        self.assertIsInstance(analysis['avg_uncertainty'], float)
        self.assertIsInstance(analysis['per_objective_variance'], dict)
        self.assertIsInstance(analysis['max_variance'], float)
        self.assertIsInstance(analysis['min_variance'], float)
        self.assertIsInstance(analysis['is_uncertain'], bool)


class TestMultiObjectiveUncertaintyReward(unittest.TestCase):
    """
    Test multi-objective uncertainty reward calculation
    """
    
    def setUp(self):
        """Setup test environment"""
        self.reward_calculator = MultiObjectiveUncertaintyReward(
            uncertainty_penalty=0.5,
            confidence_threshold=0.7
        )
        
        # Create dummy predictions
        self.predictions = {
            'qed': torch.tensor(0.8),
            'sa': torch.tensor(2.5),
            'affinity': torch.tensor(-8.0),
            'tpsa': torch.tensor(75.0)
        }
        self.uncertainty = 0.3
    
    def test_calculate_reward(self):
        """Test reward calculation"""
        results = self.reward_calculator.calculate_reward(
            self.predictions, 
            self.uncertainty
        )
        
        self.assertIn('reward', results)
        self.assertIn('uncertainty', results)
        self.assertIn('confidence', results)
        self.assertIn('is_confident', results)
        self.assertIn('base_reward', results)
        self.assertIn('uncertainty_penalty', results)
        
        # Check types
        self.assertIsInstance(results['reward'], torch.Tensor)
        self.assertIsInstance(results['uncertainty'], float)
        self.assertIsInstance(results['confidence'], torch.Tensor)
        self.assertIsInstance(results['is_confident'], bool)
    
    def test_reward_bounds(self):
        """Test reward stays within reasonable bounds"""
        results = self.reward_calculator.calculate_reward(
            self.predictions, 
            self.uncertainty
        )
        
        # Check reward is finite
        self.assertTrue(torch.isfinite(results['reward']))
        
        # Check confidence is between 0 and 1
        self.assertGreaterEqual(results['confidence'].item(), 0.0)
        self.assertLessEqual(results['confidence'].item(), 1.0)


class TestEnhancedSurrogateScorer(unittest.TestCase):
    """
    Test enhanced surrogate scorer
    """
    
    def setUp(self):
        """Setup test environment"""
        self.scorer = SurrogateScorer(
            node_in_dim=6,
            hidden_dim=16,
            num_models=3,
            uncertainty_penalty=0.5
        )
        
        # Create dummy data
        self.data = {
            'x_L': torch.randn(5, 6),  # 5 atoms, 6 features
            'pos_L': torch.randn(5, 3),
            'x_P': torch.randn(50, 6),
            'pos_P': torch.randn(50, 3),
            'pocket_center': torch.randn(3)
        }
        self.batch = torch.utils.data.Batch(**self.data)
    
    def test_predict_batch_reward(self):
        """Test batch reward prediction"""
        reward, is_uncertain = self.scorer.predict_batch_reward(self.batch)
        
        self.assertIsInstance(reward, torch.Tensor)
        self.assertIsInstance(is_uncertain, bool)
        self.assertEqual(reward.shape, (1,))
    
    def test_get_uncertainty_analysis(self):
        """Test uncertainty analysis"""
        analysis = self.scorer.get_uncertainty_analysis(self.batch)
        
        self.assertIn('avg_uncertainty', analysis)
        self.assertIn('per_objective_variance', analysis)
        self.assertIn('max_variance', analysis)
        self.assertIn('min_variance', analysis)
        self.assertIn('is_uncertain', analysis)


if __name__ == '__main__':
    unittest.main(verbosity=2)
