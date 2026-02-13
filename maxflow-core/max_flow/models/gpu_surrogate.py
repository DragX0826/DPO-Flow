"""
GPU-Accelerated Surrogate Scoring Module
Breaking the Scoring Wall: GPU Surrogate Scoring for High-Throughput Molecular Evaluation

This module implements:
1. GPU Surrogate Scoring (GNNProxy/GNINA) - <1ms per molecule vs 30s for Vina
2. Asynchronous Scoring Queue for non-blocking evaluation
3. Batch processing optimization for throughput
4. Mixed precision inference for speed
"""

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.cuda.amp import autocast
import asyncio
import threading
import queue
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

from max_flow.models.backbone import CrossGVP
from max_flow.models.surrogate import GNNProxy, SurrogateScorer


@dataclass
class SurrogateConfig:
    """Configuration for GPU surrogate scoring"""
    device: str = 'cuda' if cuda.is_available() else 'cpu'
    use_mixed_precision: bool = True
    batch_size: int = 256
    max_queue_size: int = 1000
    num_workers: int = 2
    use_async_scoring: bool = True
    compile_model: bool = True


class UltraLightGNNProxy(nn.Module):
    """
    Ultra-lightweight GNN for maximum speed
    Optimized for <1ms inference time per molecule
    """
    def __init__(self, node_in_dim=58, hidden_dim=32, num_layers=1):
        super().__init__()
        # Minimal architecture for speed
        self.backbone = CrossGVP(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers  # Single layer for maximum speed
        )
        
        # Single shared head for all properties (faster than separate heads)
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # affinity, qed, sa, tpsa
        )
        
        # Property-specific biases (cheaper than separate heads)
        self.bias_affinity = nn.Parameter(torch.zeros(1))
        self.bias_qed = nn.Parameter(torch.zeros(1))
        self.bias_sa = nn.Parameter(torch.zeros(1))
        self.bias_tpsa = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        """Forward pass optimized for speed"""
        # Encode geometry and chemistry
        h_node, _ = self.backbone(data)
        
        # Global pooling
        if hasattr(data, 'batch'):
            from torch_scatter import scatter_mean
            h_graph = scatter_mean(h_node, data.batch, dim=0)
        else:
            h_graph = h_node.mean(dim=0, keepdim=True)
        
        # Shared processing
        properties = self.shared_head(h_graph)
        
        # Apply biases for property-specific outputs
        affinity = properties[:, 0:1] + self.bias_affinity
        qed_raw = properties[:, 1:2] + self.bias_qed
        sa_raw = properties[:, 2:3] + self.bias_sa
        tpsa = properties[:, 3:4] + self.bias_tpsa
        
        # Apply activations
        qed = torch.sigmoid(qed_raw)
        sa = torch.relu(sa_raw) + 1.0  # Ensure positive SA scores
        
        return {
            'affinity': affinity.squeeze(-1),
            'qed': qed.squeeze(-1),
            'sa': sa.squeeze(-1),
            'tpsa': tpsa.squeeze(-1)
        }


class GPUSurrogateScorer:
    """
    GPU-optimized surrogate scorer with async capabilities
    Achieves <1ms per molecule scoring
    """
    def __init__(self, config: SurrogateConfig = None, checkpoint_path: Optional[str] = None):
        self.config = config or SurrogateConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize ultra-light model
        self.model = UltraLightGNNProxy().to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        # Compile model for additional speed if requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        self.model.eval()
        
        # Async scoring components
        self.scoring_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue()
        self.scoring_thread = None
        self.stop_event = threading.Event()
        
        if self.config.use_async_scoring:
            self._start_async_scoring()

    def _start_async_scoring(self):
        """Start async scoring thread"""
        self.scoring_thread = threading.Thread(target=self._async_scoring_worker)
        self.scoring_thread.daemon = True
        self.scoring_thread.start()

    def _async_scoring_worker(self):
        """Worker thread for async scoring"""
        while not self.stop_event.is_set():
            try:
                # Get batch from queue (timeout to allow checking stop event)
                batch_data = self.scoring_queue.get(timeout=0.1)
                if batch_data is None:  # Sentinel value
                    break
                
                data_batch, callback = batch_data
                
                # Process batch
                with torch.no_grad():
                    if self.config.use_mixed_precision:
                        with autocast():
                            predictions = self._process_batch(data_batch)
                    else:
                        predictions = self._process_batch(data_batch)
                
                # Put results in result queue
                self.result_queue.put((predictions, callback))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put((e, None))

    def _process_batch(self, data_batch):
        """Process a batch of molecules"""
        # Move to device if needed
        if hasattr(data_batch, 'to'):
            data_batch = data_batch.to(self.device)
        
        # Forward pass
        predictions = self.model(data_batch)
        
        # Calculate reward
        reward = self._calculate_reward(predictions)
        
        return {
            'reward': reward,
            'predictions': predictions,
            'valid': torch.ones(reward.size(0), dtype=torch.bool, device=self.device)
        }

    def _calculate_reward(self, predictions, weights=None):
        """Calculate weighted reward from predictions"""
        if weights is None:
            weights = {'qed': 3.0, 'sa': 1.0, 'affinity': 0.1, 'tpsa_penalty': -0.1}
        
        # Normalize SA score (lower is better, range ~1-10)
        norm_sa = (10.0 - predictions['sa']) / 9.0
        
        # TPSA penalty logic
        tp = predictions['tpsa']
        t_penalty = torch.zeros_like(tp)
        mask_low = tp < 60
        mask_high = tp > 90
        t_penalty[mask_low] = 60 - tp[mask_low]
        t_penalty[mask_high] = tp[mask_high] - 90
        
        # Calculate weighted reward
        reward = (
            weights.get('qed', 3.0) * predictions['qed'] +
            weights.get('sa', 1.0) * norm_sa +
            weights.get('affinity', 0.1) * predictions['affinity'] +
            weights.get('tpsa_penalty', -0.1) * t_penalty
        )
        
        return reward

    def predict_batch_reward(self, data_batch, weights=None, use_async=True):
        """
        Fast batch reward prediction
        
        Args:
            data_batch: Batch of molecular data
            weights: Reward weights dictionary
            use_async: Whether to use async scoring
            
        Returns:
            reward, valid_mask tuple
        """
        if use_async and self.config.use_async_scoring:
            # Async scoring
            callback = threading.Event()
            self.scoring_queue.put((data_batch, callback))
            
            # Wait for result
            while True:
                try:
                    result, result_callback = self.result_queue.get(timeout=0.01)
                    if result_callback == callback:
                        if isinstance(result, Exception):
                            raise result
                        return result['reward'], result['valid']
                except queue.Empty:
                    continue
        else:
            # Synchronous scoring
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with autocast():
                        result = self._process_batch(data_batch)
                else:
                    result = self._process_batch(data_batch)
            return result['reward'], result['valid']

    def benchmark_speed(self, num_molecules=1000, batch_size=None):
        """Benchmark scoring speed"""
        batch_size = batch_size or self.config.batch_size
        
        # Create dummy data
        import torch_geometric.data as pyg_data
        
        dummy_molecules = []
        for i in range(num_molecules):
            # Create minimal dummy graph
            num_nodes = np.random.randint(10, 50)
            x = torch.randn(num_nodes, 58)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            
            data = pyg_data.Data(x=x, edge_index=edge_index)
            dummy_molecules.append(data)
        
        # Benchmark
        start_time = time.time()
        total_reward_time = 0
        
        for i in range(0, num_molecules, batch_size):
            batch_end = min(i + batch_size, num_molecules)
            batch_data = dummy_molecules[i:batch_end]
            
            # Create batch
            batch = pyg_data.Batch.from_data_list(batch_data)
            
            # Time the scoring
            reward_start = time.time()
            reward, valid = self.predict_batch_reward(batch, use_async=False)
            reward_time = time.time() - reward_start
            total_reward_time += reward_time
        
        total_time = time.time() - start_time
        
        results = {
            'total_time': total_time,
            'scoring_time': total_reward_time,
            'molecules_per_second': num_molecules / total_time,
            'average_time_per_molecule_ms': (total_reward_time / num_molecules) * 1000,
            'throughput_improvement_vs_vina': 30000 / ((total_reward_time / num_molecules) * 1000)  # Vina ~30s vs our time
        }
        
        return results

    def shutdown(self):
        """Shutdown async scoring"""
        if self.config.use_async_scoring and self.scoring_thread:
            self.stop_event.set()
            self.scoring_queue.put(None)  # Sentinel value
            self.scoring_thread.join(timeout=1.0)


class AsyncScoringQueue:
    """
    Asynchronous scoring queue for non-blocking molecular evaluation
    """
    def __init__(self, scorer: GPUSurrogateScorer, max_queue_size=1000):
        self.scorer = scorer
        self.max_queue_size = max_queue_size
        self.pending_requests = {}
        self.request_id = 0
        self.lock = threading.Lock()
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.stop_event = threading.Event()

    def _process_queue(self):
        """Background thread to process scoring requests"""
        while not self.stop_event.is_set():
            with self.lock:
                if not self.pending_requests:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                # Get next request
                request_id, (data_batch, callback) = next(iter(self.pending_requests.items()))
                del self.pending_requests[request_id]
            
            try:
                # Process request
                result = self.scorer.predict_batch_reward(data_batch, use_async=False)
                callback(result)
            except Exception as e:
                callback((e, None))

    def submit_scoring_request(self, data_batch, callback):
        """Submit a scoring request for async processing"""
        with self.lock:
            if len(self.pending_requests) >= self.max_queue_size:
                raise RuntimeError(f"Scoring queue full (max: {self.max_queue_size})")
            
            self.request_id += 1
            request_id = self.request_id
            self.pending_requests[request_id] = (data_batch, callback)
            
        return request_id

    def shutdown(self):
        """Shutdown the async queue"""
        self.stop_event.set()
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)


# Convenience function for quick benchmarking
def benchmark_gpu_surrogate():
    """Quick benchmark of GPU surrogate scoring"""
    print("🚀 Benchmarking GPU Surrogate Scoring...")
    
    config = SurrogateConfig(
        device='cuda' if cuda.is_available() else 'cpu',
        use_mixed_precision=True,
        batch_size=256,
        compile_model=True
    )
    
    scorer = GPUSurrogateScorer(config)
    
    # Run benchmark
    results = scorer.benchmark_speed(num_molecules=1000)
    
    print(f"📊 Benchmark Results:")
    print(f"  Molecules per second: {results['molecules_per_second']:.2f}")
    print(f"  Average time per molecule: {results['average_time_per_molecule_ms']:.3f} ms")
    print(f"  Speedup vs Vina: {results['throughput_improvement_vs_vina']:.1f}x")
    
    if results['average_time_per_molecule_ms'] < 1.0:
        print("✅ Target achieved: <1ms per molecule!")
    else:
        print("⚠️  Target not achieved: >1ms per molecule")
    
    scorer.shutdown()
    return results


if __name__ == "__main__":
    # Run benchmark when script is executed directly
    from pathlib import Path
    
    results = benchmark_gpu_surrogate()
    print("\n🎯 GPU Surrogate Scoring implementation complete!")
