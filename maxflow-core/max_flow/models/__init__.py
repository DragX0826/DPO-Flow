# max_flow/models/__init__.py

from .surrogate import GNNProxy, SurrogateScorer
from .flash_attention import FlashAttention2, FlashTransformerLayer, benchmark_attention_speed
from .vina_gpu import VinaGPUWrapper, VinaGPUConfig, benchmark_vina_gpu_speed
from .max_rl import MaxRL
from .flow_matching import RectifiedFlow
from .adaptive_loss import AdaptiveLossWeighting
from .gpu_surrogate import UltraLightGNNProxy, SurrogateConfig
from .pcgrad import PCGrad
from .smooth_psa import SmoothPSARegularizer
from .backbone import GlobalContextBlock, GVPEncoder
from .layers import HyperConnection, ManifoldConstrainedHC

__all__ = [
    # Core surrogate models
    'GNNProxy',
    'SurrogateScorer', 
    'UltraLightGNNProxy',
    'SurrogateConfig',
    
    # FlashAttention components
    'FlashAttention2',
    'FlashTransformerLayer',
    'benchmark_attention_speed',
    
    # Vina-GPU integration
    'VinaGPUWrapper',
    'VinaGPUConfig',
    'benchmark_vina_gpu_speed',
    
    # MaxRL training
    'MaxRL',
    
    # Flow matching
    'RectifiedFlow',
    
    # Loss functions
    'AdaptiveLossWeighting',
    
    # PSA calculation
    'SmoothPSARegularizer',
    
    # Optimization
    'PCGrad',
    
    # Neural network components
    'GlobalContextBlock',
    'GVPEncoder',
    'HyperConnection',
    'ManifoldConstrainedHC',
]
