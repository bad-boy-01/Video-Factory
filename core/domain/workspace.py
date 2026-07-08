import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

class WorkspaceManager:
    """
    Central owner of all filesystem operations.
    Manages cache, final outputs, registries, and temporary files.
    """
    def __init__(self, base_dir: str = "workspace"):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "cache"
        self.assets_dir = self.cache_dir / "assets"
        self.outputs_dir = self.base_dir / "outputs"
        self.temp_dir = self.base_dir / "temp"
        self.manifests_dir = self.base_dir / "manifests"
        
        self._initialize_directories()

    def _initialize_directories(self):
        """Creates required directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def get_asset_dir(self, asset_hash: str) -> Path:
        """Deprecated. Use get_cas_dir."""
        target = self.assets_dir / asset_hash
        target.mkdir(parents=True, exist_ok=True)
        return target
        
    def get_versioned_asset_dir(self, shot_id: str, version: int) -> Path:
        """Deprecated. Use get_cas_dir."""
        target = self.assets_dir / shot_id / f"v{version}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_cas_dir(self, content_hash: str) -> Path:
        """Returns the Content Addressable Storage directory for a specific hash."""
        target = self.assets_dir / content_hash
        target.mkdir(parents=True, exist_ok=True)
        return target
        
    def get_db_path(self, db_name: str) -> str:
        """Returns the path to a SQLite database in the cache directory."""
        return str(self.cache_dir / db_name)
        
    def get_temp_path(self, filename: str) -> str:
        """Returns a path inside the temp directory."""
        return str(self.temp_dir / filename)
        
    def save_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Saves a JSON file directly into the base workspace."""
        filepath = self.base_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Loads a JSON file from the base workspace."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def get_output_path(self, filename: str) -> str:
        """Returns a path inside the outputs directory."""
        return str(self.outputs_dir / filename)
