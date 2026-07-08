import json
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from core.domain.assets.execution import GeneratedImage, ProvenanceRecord


class CacheProvider:
    """
    Manages the cache of execution nodes.
    A hit restores the entire GeneratedImage artifact and its provenance.
    """
    def __init__(self, cache_dir: str = ".cache/generation"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_node_dir(self, cache_key: str) -> Path:
        return self.cache_dir / cache_key

    def check_cache(self, cache_key: str) -> Optional[GeneratedImage]:
        node_dir = self._get_node_dir(cache_key)
        
        if not node_dir.exists():
            return None
            
        metadata_file = node_dir / "metadata.json"
        provenance_file = node_dir / "provenance.json"
        
        if not metadata_file.exists() or not provenance_file.exists():
            return None
            
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        with open(provenance_file, "r") as f:
            prov_data = json.load(f)
            
        provenance = ProvenanceRecord(**prov_data)
        
        return GeneratedImage(
            image_path=Path(metadata["image_path"]),
            width=metadata["width"],
            height=metadata["height"],
            seed=metadata["seed"],
            prompt_hash=metadata["prompt_hash"],
            model_id=metadata["model_id"],
            cache_hit=True,
            provenance=provenance
        )

    def save_to_cache(self, cache_key: str, image: GeneratedImage):
        """
        Saves the execution node to the cache.
        The image path is copied into the cache directory.
        """
        node_dir = self._get_node_dir(cache_key)
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image to cache dir
        cached_image_path = node_dir / "image.png"
        shutil.copy(image.image_path, cached_image_path)
        
        # Save metadata
        metadata = {
            "image_path": str(cached_image_path),
            "width": image.width,
            "height": image.height,
            "seed": image.seed,
            "prompt_hash": image.prompt_hash,
            "model_id": image.model_id
        }
        with open(node_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Save provenance
        if image.provenance:
            with open(node_dir / "provenance.json", "w") as f:
                json.dump(asdict(image.provenance), f, indent=2)
