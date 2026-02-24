import torch
import logging

_ESM_MODEL_CACHE = {}

def get_esm_model(model_name="esm2_t33_650M_UR50D", force_fp32=False, device=None):
    """
    Ensures the 2.5GB ESM model is only loaded once per process/device.
    In multi-GPU spawn mode, each worker process gets its own cache.
    
    Args:
        device: torch.device or str, e.g. 'cuda:0' or 'cpu'.
                If None, auto-selects CUDA if available, else CPU.
    """
    _logger = logging.getLogger("SAEB-Flow.utils.esm")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Cache key includes device so each GPU has its own copy
    cache_key = (model_name, force_fp32, str(device))
    if cache_key in _ESM_MODEL_CACHE:
        return _ESM_MODEL_CACHE[cache_key]
    
    try:
        import esm
        _logger.info(f"Loading ESM-2 Model ({model_name}, fp32={force_fp32}) on {device}...")
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device)
        if not force_fp32:
            model = model.half()
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        _ESM_MODEL_CACHE[cache_key] = (model, alphabet)
    except ImportError:
        _logger.warning("ESM-2 not installed. Falling back to learned embeddings. "
                        "Install with: pip install fair-esm")
        return None, None
    except Exception as e:
        _logger.error(f"ESM-2 Load ERROR: {e}")
        return None, None
    return _ESM_MODEL_CACHE[cache_key]
