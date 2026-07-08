from typing import List, Dict, Any
from core.domain.prompt.provider_request import ProviderRequest
import logging

logger = logging.getLogger(__name__)

class RenderScheduler:
    """
    Sorts and batches rendering jobs to minimize expensive context switches
    (e.g., swapping LoRAs, ControlNets, or IP-Adapters).
    """
    def __init__(self):
        pass
        
    def schedule(self, requests: List[ProviderRequest]) -> List[ProviderRequest]:
        if not requests:
            return []
            
        logger.info(f"Scheduling {len(requests)} render jobs...")
        
        # A simple heuristic: sort by primary LoRA, then IP-Adapter character
        def sort_key(req: ProviderRequest):
            primary_lora = req.bindings.loras[0] if req.bindings.loras else ""
            primary_character = list(req.conditioning.ip_adapter.keys())[0] if req.conditioning.ip_adapter else ""
            return (primary_lora, primary_character)
            
        sorted_requests = sorted(requests, key=sort_key)
        
        logger.info("Render jobs scheduled successfully.")
        return sorted_requests
