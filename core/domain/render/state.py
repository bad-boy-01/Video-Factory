from pydantic import BaseModel
from typing import List, Optional

class CharacterRenderState(BaseModel):
    """
    Ephemeral render state. Lives only during rendering. 
    Never serialized into the canonical StoryBible.
    """
    character_id: str
    reference_image_path: Optional[str] = None
    embedding_path: Optional[str] = None
    ip_adapter_image_path: Optional[str] = None
    lora_stack: List[str] = []
    pose_history: List[str] = []
    last_camera_angle: str = ""
