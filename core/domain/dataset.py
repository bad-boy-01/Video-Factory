from pydantic import BaseModel
from typing import List, Optional

class DatasetManifest(BaseModel):
    title: str = "Unknown"
    author: Optional[str] = None
    language: str = "en"
    source: Optional[str] = None
    chapters: Optional[int] = None
    encoding: str = "utf-8"

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
