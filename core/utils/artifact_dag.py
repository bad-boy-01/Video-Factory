import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

class ArtifactDAG:
    """
    Lightweight DAG for incremental rebuilds.
    Tracks content hashes of artifacts and their dependencies to determine if a stage needs to run.
    """
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.manifests_dir = self.workspace_dir / "manifests"
        self._dag_path = self.manifests_dir / "artifact_dag.json"
        self._dag = self._load()
    
    def _load(self) -> Dict[str, dict]:
        if self._dag_path.exists():
            try:
                with open(self._dag_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save(self):
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        with open(self._dag_path, "w", encoding="utf-8") as f:
            json.dump(self._dag, f, indent=2)
            
    def _hash_deps(self, deps: List[str]) -> str:
        """Hash a list of dependency fingerprints/strings."""
        return hashlib.sha256("_".join(sorted(deps)).encode()).hexdigest()

    def is_fresh(self, artifact_name: str, deps: List[str], generator_signature: str) -> bool:
        """Returns True if artifact exists and all deps are unchanged."""
        node = self._dag.get(artifact_name)
        if not node:
            return False
            
        path = self.manifests_dir / f"{artifact_name}.json"
        if not path.exists():
            return False
            
        dep_hash = self._hash_deps(deps)
        return (
            node.get("dep_hash") == dep_hash and 
            node.get("generator_signature") == generator_signature
        )
    
    def record(self, artifact_name: str, deps: List[str], generator_signature: str):
        self._dag[artifact_name] = {
            "dep_hash": self._hash_deps(deps),
            "generator_signature": generator_signature,
            "path": str(self.manifests_dir / f"{artifact_name}.json"),
            "timestamp": datetime.utcnow().isoformat()
        }
        self._save()
