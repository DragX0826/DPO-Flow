import torch
import logging

_ESM_MODEL_CACHE = {}

def get_esm_model(model_name="esm2_t33_650M_UR50D", force_fp32=False):
    """
    Ensures the 2.5GB ESM model is only loaded once and shared.
    Optional FP32 mode for precision parity checking.
    """
    _logger = logging.getLogger("SAEB-Flow.utils.esm")
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_key = (model_name, force_fp32)
    if cache_key in _ESM_MODEL_CACHE:
        return _ESM_MODEL_CACHE[cache_key]
    
    try:
        import esm
        _logger.info(f"Loading ESM-2 Model ({model_name}, fp32={force_fp32})...")
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model = model.to(_device)
        if not force_fp32:
            model = model.half()
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        _ESM_MODEL_CACHE[cache_key] = (model, alphabet)
    except Exception as e:
        _logger.error(f"ESM-2 Load ERROR: {e}")
        return None, None
    return _ESM_MODEL_CACHE[cache_key]
