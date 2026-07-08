from pydantic import BaseModel

class PromptPack(BaseModel):
    """
    A profile containing stylistic modifiers, negative constraints,
    and camera vocabulary for a specific visual style.
    """
    name: str
    positive_prefix: str = ""
    positive_suffix: str = "masterpiece, best quality, highly detailed"
    negative_prompt: str = "lowres, bad anatomy, bad hands, text, error"
    cfg_scale: float = 7.0
