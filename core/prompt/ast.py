from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptAST:
    character: str
    outfit: str
    scene: str
    camera: str
    lighting: str
    style: str
    negative: Optional[str] = None

    def to_canonical_hash(self) -> str:
        """Serializes the AST into a canonical deterministic hash."""
        import json
        import hashlib
        # Convert dataclass to dict and strip None/empty values if needed, sort keys to ensure deterministic output
        ast_dict = {k: v for k, v in self.__dict__.items() if v is not None}
        canonical_str = json.dumps(ast_dict, sort_keys=True)
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
