"""
FlashAttention-2 Integration Module
High-performance attention implementation with 2-4x speedup for Transformer layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import time
import numpy as np

# Try to import FlashAttention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️ FlashAttention not available, using standard attention fallback")


class FlashAttention2(nn.Module):
    """
    FlashAttention-2 implementation with fallback to standard attention
    Provides 2-4x speedup for Transformer layers
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, device: str = 'cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.device = device
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device)
        
        # Dropout for standard attention fallback
        self.attn_dropout = nn.Dropout(dropout)
        # Compile for additional speedup (disabled due to system compatibility)
        # if hasattr(torch, 'compile'):
        #     self.forward = torch.compile(self.forward, mode="reduce-overhead")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with FlashAttention-2 or standard attention fallback
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if FLASH_ATTN_AVAILABLE and self.device == 'cuda':
            # Use FlashAttention-2
            with torch.amp.autocast('cuda'):
                attn_output = flash_attn_func(Q, K, V, dropout_p=self.dropout if self.training else 0.0)
            attn_output = attn_output.view(batch_size, seq_len, embed_dim)
        else:
            # Standard attention fallback
            attn_output = self._standard_attention(Q, K, V, mask)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output
    
    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention implementation as fallback"""
        batch_size, seq_len, num_heads, head_dim = Q.shape
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        return attn_output


class FlashTransformerLayer(nn.Module):
    """
    Transformer layer with FlashAttention-2 integration
    Drop-in replacement for standard Transformer layers
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, device: str = 'cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        
        # Self-attention with FlashAttention-2
        self.self_attn = FlashAttention2(embed_dim, num_heads, dropout, device)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, device=device),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim, device=device)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim, device=device)
        self.ln2 = nn.LayerNorm(embed_dim, device=device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections and layer norm"""
        # Self-attention with residual
        attn_output = self.self_attn(x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        
        return x


def benchmark_attention_speed(seq_len: int = 512, 
                              batch_size: int = 8,
                              embed_dim: int = 256,
                              num_heads: int = 8,
                              num_trials: int = 10) -> dict:
    """
    Benchmark FlashAttention vs standard attention performance
    
    Returns:
        Dictionary with timing results and speedup metrics
    """
    print(f"⚡ Benchmarking attention for seq_len={seq_len}, batch_size={batch_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Initialize attention modules
    flash_attention = FlashAttention2(
        embed_dim, num_heads, dropout=0.0, device=str(device)
    )
    standard_attention = nn.MultiheadAttention(
        embed_dim, num_heads, dropout=0.0, batch_first=True, device=device
    )
    
    # Benchmark FlashAttention
    flash_times = []
    for _ in range(num_trials):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            _ = flash_attention(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        flash_times.append(time.time() - start_time)
    
    # Benchmark standard attention
    standard_times = []
    for _ in range(num_trials):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            _ = standard_attention(x, x, x)[0]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        standard_times.append(time.time() - start_time)
    
    # Calculate statistics
    avg_flash_time = np.mean(flash_times)
    avg_standard_time = np.mean(standard_times)
    speedup = avg_standard_time / avg_flash_time if avg_flash_time > 0 else float('inf')
    
    results = {
        'flash_attention_time': avg_flash_time,
        'standard_attention_time': avg_standard_time,
        'speedup': speedup,
        'flash_attention_available': FLASH_ATTN_AVAILABLE,
        'device': str(device)
    }
    
    print(f"📊 Attention Benchmark Results:")
    print(f"  FlashAttention time: {avg_flash_time*1000:.2f}ms")
    print(f"  Standard attention time: {avg_standard_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    
    return results


if __name__ == "__main__":
    # Run benchmark when script is executed directly
    results = benchmark_attention_speed()
    print(f"\n🎯 Benchmark completed with {results['speedup']:.1f}x speedup")
