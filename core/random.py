import random
import os
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int, python: bool = True, numpy: bool = True, torch: bool = True, cuda: bool = True):
    """
    Locks the random seed globally across all libraries to ensure reproducible compiler runs.
    """
    if python:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    if numpy:
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
            
    if torch:
        try:
            import torch as th
            th.manual_seed(seed)
            if cuda and th.cuda.is_available():
                th.cuda.manual_seed_all(seed)
                # Ensure deterministic algorithms
                th.backends.cudnn.deterministic = True
                th.backends.cudnn.benchmark = False
        except ImportError:
            pass
            
    logger.info(f"Global random seed locked to {seed}")
