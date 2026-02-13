# maxflow/optimization/pcgrad.py
"""
[SOTA Phase 12] PCGrad: Gradient Surgery for Multi-Objective Optimization.
Resolves conflicts between competing objectives (e.g., Affinity vs ADMET) 
by projecting conflicting gradients.
Based on: "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020).
"""

import torch
import torch.nn as nn
import random

class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def pcgrad_backward(self, objectives):
        """
        Calculates gradients for each objective separately and resolves conflicts.
        objectives: list of loss tensors
        """
        grads, shapes, has_grad = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grad)
        self._set_grad(pc_grad)

    def _project_conflicting(self, grads, has_grad):
        """Project conflicting gradients."""
        num_tasks = len(grads)
        pc_grad = grads.copy()
        
        # Randomize task order for stability
        task_indices = list(range(num_tasks))
        random.shuffle(task_indices)
        
        for i in task_indices:
            for j in task_indices:
                if i == j: continue
                # Dot product
                dot_prod = torch.dot(pc_grad[i], grads[j])
                if dot_prod < 0:
                    # Conflict detected: project g_i onto the plane normal to g_j
                    pc_grad[i] -= (dot_prod / (torch.norm(grads[j])**2 + 1e-8)) * grads[j]
        
        return torch.stack(pc_grad).sum(dim=0)

    def _pack_grad(self, objectives):
        """Collect and flatten gradients for each task."""
        grads = []
        shapes = []
        has_grad = []
        
        for loss in objectives:
            self._optim.zero_grad()
            loss.backward(retain_graph=True)
            
            grad_vec = []
            for group in self._optim.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grad_vec.append(p.grad.view(-1))
                        if len(shapes) < len(group['params']): # Only store shapes once
                            shapes.append(p.shape)
                    else:
                        if len(shapes) < len(group['params']):
                             shapes.append(p.shape)
            
            if grad_vec:
                grads.append(torch.cat(grad_vec))
                has_grad.append(True)
            else:
                has_grad.append(False)
        
        return grads, shapes, has_grad

    def _set_grad(self, pc_grad):
        """Apply modified gradients back to parameters."""
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.grad = pc_grad[idx:idx + numel].view(p.shape).clone()
                idx += numel
