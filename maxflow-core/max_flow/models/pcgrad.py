"""
PCGrad Implementation for Multi-Objective Gradient Surgery
Implements gradient projection to resolve conflicts between competing objectives.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np


class PCGrad:
    """
    PCGrad: Gradient Surgery for Multi-Task Learning.
    
    Implements:
    - Gradient conflict detection
    - Gradient projection to orthogonal planes
    - Pareto optimal optimization
    - Automatic conflict resolution
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize PCGrad.
        
        Args:
            reduction: How to reduce gradients ('mean', 'sum', 'none')
        """
        self.reduction = reduction
        self.conflict_history = []
        self.gradient_stats = []
        
    def pc_gradient(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply PCGrad to gradients.
        
        Args:
            gradients: List of gradients for each task
            
        Returns:
            projected_gradients: List of projected gradients
        """
        if len(gradients) <= 1:
            return gradients
        
        # Flatten all gradients
        flat_gradients = []
        shapes = []
        
        for grad in gradients:
            shapes.append(grad.shape)
            flat_gradients.append(grad.view(-1))
        
        # Apply PCGrad to flattened gradients
        projected_flat_gradients = self._pc_gradient_flat(flat_gradients)
        
        # Reshape back to original shapes
        projected_gradients = []
        for grad, shape in zip(projected_flat_gradients, shapes):
            projected_gradients.append(grad.view(shape))
        
        return projected_gradients
    
    def _pc_gradient_flat(self, flat_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply PCGrad to flattened gradients.
        
        Args:
            flat_gradients: List of flattened gradients
            
        Returns:
            projected_flat_gradients: List of projected flattened gradients
        """
        num_tasks = len(flat_gradients)
        
        # Calculate cosine similarities
        cosine_matrix = torch.zeros(num_tasks, num_tasks)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    cosine_matrix[i, j] = self._cosine_similarity(flat_gradients[i], flat_gradients[j])
        
        # Project gradients
        projected_gradients = []
        
        for i in range(num_tasks):
            current_grad = flat_gradients[i].clone()
            
            # Check conflicts with other tasks
            conflicts = []
            for j in range(num_tasks):
                if i != j:
                    if cosine_matrix[i, j] < 0:  # Conflict detected
                        conflicts.append(j)
            
            # Project conflicting gradients
            if conflicts:
                # Record conflict
                self.conflict_history.append({
                    'task_i': i,
                    'conflicting_tasks': conflicts,
                    'cosine_similarities': [cosine_matrix[i, j].item() for j in conflicts]
                })
                
                # Project current gradient
                for j in conflicts:
                    current_grad = self._project_gradient(current_grad, flat_gradients[j])
            
            projected_gradients.append(current_grad)
        
        # Record gradient statistics
        self._record_gradient_stats(flat_gradients, projected_gradients, cosine_matrix)
        
        return projected_gradients
    
    def _cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> torch.Tensor:
        """
        Calculate cosine similarity between two gradients.
        
        Args:
            grad1: First gradient
            grad2: Second gradient
            
        Returns:
            similarity: Cosine similarity
        """
        # Normalize gradients
        grad1_norm = torch.nn.functional.normalize(grad1, p=2, dim=0)
        grad2_norm = torch.nn.functional.normalize(grad2, p=2, dim=0)
        
        # Calculate cosine similarity
        similarity = torch.sum(grad1_norm * grad2_norm)
        
        return similarity
    
    def _project_gradient(self, grad_to_project: torch.Tensor, grad_to_avoid: torch.Tensor) -> torch.Tensor:
        """
        Project gradient to orthogonal plane of another gradient.
        
        Args:
            grad_to_project: Gradient to project
            grad_to_avoid: Gradient to avoid
            
        Returns:
            projected_grad: Projected gradient
        """
        # Normalize gradient to avoid
        grad_to_avoid_norm = torch.nn.functional.normalize(grad_to_avoid, p=2, dim=0)
        
        # Calculate projection
        projection = torch.sum(grad_to_project * grad_to_avoid_norm) * grad_to_avoid_norm
        
        # Subtract projection to get orthogonal component
        projected_grad = grad_to_project - projection
        
        return projected_grad
    
    def _record_gradient_stats(self, original_gradients: List[torch.Tensor], 
                               projected_gradients: List[torch.Tensor], 
                               cosine_matrix: torch.Tensor):
        """
        Record gradient statistics for analysis.
        
        Args:
            original_gradients: Original gradients
            projected_gradients: Projected gradients
            cosine_matrix: Cosine similarity matrix
        """
        stats = {
            'num_tasks': len(original_gradients),
            'conflicts': [],
            'gradient_norms': [],
            'projection_ratios': []
        }
        
        # Calculate conflicts
        for i in range(len(original_gradients)):
            for j in range(len(original_gradients)):
                if i != j and cosine_matrix[i, j] < 0:
                    stats['conflicts'].append({
                        'task_pair': (i, j),
                        'cosine_similarity': cosine_matrix[i, j].item()
                    })
        
        # Calculate gradient norms
        for i, (orig, proj) in enumerate(zip(original_gradients, projected_gradients)):
            orig_norm = torch.norm(orig)
            proj_norm = torch.norm(proj)
            stats['gradient_norms'].append({
                'task': i,
                'original_norm': orig_norm.item(),
                'projected_norm': proj_norm.item(),
                'norm_ratio': (proj_norm / (orig_norm + 1e-8)).item()
            })
        
        # Calculate projection ratios
        for i, (orig, proj) in enumerate(zip(original_gradients, projected_gradients)):
            if torch.norm(orig) > 0:
                ratio = torch.norm(proj) / torch.norm(orig)
                stats['projection_ratios'].append(ratio.item())
        
        self.gradient_stats.append(stats)
    
    def get_conflict_analysis(self) -> Dict:
        """
        Get analysis of gradient conflicts.
        
        Returns:
            analysis: Conflict analysis
        """
        if not self.conflict_history:
            return {'message': 'No conflicts recorded'}
        
        total_conflicts = len(self.conflict_history)
        
        # Analyze conflict patterns
        task_conflict_counts = {}
        avg_cosine_similarities = {}
        
        for conflict in self.conflict_history:
            task_i = conflict['task_i']
            if task_i not in task_conflict_counts:
                task_conflict_counts[task_i] = 0
            task_conflict_counts[task_i] += len(conflict['conflicting_tasks'])
            
            avg_cosine = np.mean(conflict['cosine_similarities'])
            avg_cosine_similarities[task_i] = avg_cosine
        
        analysis = {
            'total_conflicts': total_conflicts,
            'task_conflict_counts': task_conflict_counts,
            'avg_cosine_similarities': avg_cosine_similarities,
            'conflict_rate': total_conflicts / len(self.conflict_history) if self.conflict_history else 0
        }
        
        return analysis
    
    def get_gradient_analysis(self) -> Dict:
        """
        Get analysis of gradient statistics.
        
        Returns:
            analysis: Gradient analysis
        """
        if not self.gradient_stats:
            return {'message': 'No gradient statistics recorded'}
        
        # Aggregate statistics
        all_norm_ratios = []
        all_projection_ratios = []
        
        for stats in self.gradient_stats:
            all_norm_ratios.extend([g['norm_ratio'] for g in stats['gradient_norms']])
            all_projection_ratios.extend(stats['projection_ratios'])
        
        # Handle empty arrays gracefully
        if len(all_norm_ratios) == 0:
            analysis = {
                'avg_norm_ratio': 1.0,
                'std_norm_ratio': 0.0,
                'min_norm_ratio': 1.0,
                'max_norm_ratio': 1.0,
                'avg_projection_ratio': 1.0,
                'std_projection_ratio': 0.0,
                'min_projection_ratio': 1.0,
                'max_projection_ratio': 1.0
            }
        else:
            analysis = {
                'avg_norm_ratio': np.mean(all_norm_ratios),
                'std_norm_ratio': np.std(all_norm_ratios),
                'min_norm_ratio': np.min(all_norm_ratios),
                'max_norm_ratio': np.max(all_norm_ratios),
                'avg_projection_ratio': np.mean(all_projection_ratios) if all_projection_ratios else 1.0,
                'std_projection_ratio': np.std(all_projection_ratios) if all_projection_ratios else 0.0,
                'min_projection_ratio': np.min(all_projection_ratios) if all_projection_ratios else 1.0,
                'max_projection_ratio': np.max(all_projection_ratios) if all_projection_ratios else 1.0
            }
        
        return analysis
    
    def reset_statistics(self):
        """
        Reset conflict and gradient statistics.
        """
        self.conflict_history = []
        self.gradient_stats = []


class MultiObjectivePCGrad:
    """
    Multi-objective optimization with PCGrad.
    
    Handles multiple competing objectives using gradient surgery.
    """
    
    def __init__(self, objectives: List[str], pc_grad: PCGrad = None):
        """
        Initialize multi-objective PCGrad.
        
        Args:
            objectives: List of objective names
            pc_grad: PCGrad instance (creates new one if None)
        """
        self.objectives = objectives
        self.pc_grad = pc_grad or PCGrad()
        self.objective_gradients = {}
        self.pareto_front = []
        
    def calculate_objectives(self, model, batch) -> Dict[str, torch.Tensor]:
        """
        Calculate multiple objectives.
        
        Args:
            model: Model to optimize
            batch: Input batch
            
        Returns:
            objectives: Dictionary of objective values
        """
        # This should be implemented based on your specific objectives
        # For now, return dummy objectives
        return {
            'affinity': torch.tensor(0.0),
            'tpsa': torch.tensor(0.0),
            'qed': torch.tensor(0.0),
            'sa': torch.tensor(0.0)
        }
    
    def calculate_gradients(self, model, batch, objectives: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate gradients for each objective.
        
        Args:
            model: Model to optimize
            batch: Input batch
            objectives: Objective values
            
        Returns:
            gradients: Dictionary of gradients
        """
        gradients = {}
        
        for obj_name, obj_value in objectives.items():
            # Zero gradients
            model.zero_grad()
            
            # Calculate gradient for this objective
            obj_value.backward(retain_graph=True)
            
            # Collect gradients
            grad_dict = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_dict[name] = param.grad.clone()
            
            gradients[obj_name] = grad_dict
        
        return gradients
    
    def apply_pc_grad(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply PCGrad to gradients.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            projected_gradients: Dictionary of projected gradients
        """
        # Flatten gradients for PCGrad
        all_grads = []
        grad_info = {}
        
        for obj_name, grad_dict in gradients.items():
            flat_grads = []
            for param_name, grad in grad_dict.items():
                flat_grads.append(grad.view(-1))
                grad_info[(obj_name, param_name)] = grad.shape
            
            all_grads.append(torch.cat(flat_grads))
        
        # Apply PCGrad
        projected_flat_grads = self.pc_grad.pc_gradient(all_grads)
        
        # Reshape back
        projected_gradients = {}
        
        for i, (obj_name, _) in enumerate(gradients.items()):
            projected_gradients[obj_name] = {}
            
            start_idx = 0
            for (obj_key, param_name), shape in grad_info.items():
                if obj_key == obj_name:
                    param_size = np.prod(shape)
                    flat_grad = projected_flat_grads[i][start_idx:start_idx + param_size]
                    projected_gradients[obj_name][param_name] = flat_grad.view(shape)
                    start_idx += param_size
        
        return projected_gradients
    
    def update_model(self, model, projected_gradients: Dict[str, torch.Tensor], 
                    optimizer: torch.optim.Optimizer):
        """
        Update model with projected gradients.
        
        Args:
            model: Model to update
            projected_gradients: Projected gradients
            optimizer: Optimizer
        """
        # Zero gradients
        model.zero_grad()
        
        # Apply projected gradients
        for obj_name, grad_dict in projected_gradients.items():
            for name, param in model.named_parameters():
                if name in grad_dict:
                    param.grad = grad_dict[name]
        
        # Update model
        optimizer.step()
    
    def find_pareto_optimal(self, objectives: Dict[str, torch.Tensor]) -> bool:
        """
        Check if current objectives are Pareto optimal.
        
        Args:
            objectives: Current objective values
            
        Returns:
            is_pareto: Whether objectives are Pareto optimal
        """
        # Simple Pareto optimality check
        # This should be implemented based on your specific Pareto criteria
        
        # For now, assume all points are potentially Pareto optimal
        return True
    
    def update_pareto_front(self, objectives: Dict[str, torch.Tensor]):
        """
        Update Pareto front with new objectives.
        
        Args:
            objectives: New objective values
        """
        if self.find_pareto_optimal(objectives):
            self.pareto_front.append(objectives.copy())
            
            # Keep only non-dominated solutions
            self._prune_pareto_front()
    
    def _prune_pareto_front(self):
        """
        Prune Pareto front to keep only non-dominated solutions.
        """
        # This should be implemented based on your specific dominance criteria
        # For now, keep all solutions
        pass
    
    def get_pareto_metrics(self) -> Dict:
        """
        Get metrics about Pareto front.
        
        Returns:
            metrics: Pareto front metrics
        """
        if not self.pareto_front:
            return {'message': 'No Pareto solutions found'}
        
        metrics = {
            'pareto_front_size': len(self.pareto_front),
            'objective_names': self.objectives,
            'objective_ranges': {}
        }
        
        # Calculate objective ranges
        for obj_name in self.objectives:
            values = [sol[obj_name].item() for sol in self.pareto_front]
            metrics['objective_ranges'][obj_name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return metrics
