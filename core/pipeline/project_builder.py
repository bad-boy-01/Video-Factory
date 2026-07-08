from pathlib import Path
from core.domain.story.project import ProjectManifest, ProjectMetadata

class ProjectBuilder:
    def __init__(self, projects_dir: Path, datasets_dir: Path):
        self.projects_dir = projects_dir
        self.datasets_dir = datasets_dir

    def build_project(self, project_name: str, dataset_name: str) -> ProjectManifest:
        dataset_path = self.datasets_dir / dataset_name
        project_path = self.projects_dir / project_name

        self._validate_dataset(dataset_path)
        self._create_folder_structure(project_path)
        
        manifest = self._generate_manifest(project_name, dataset_name, project_path)
        self._initialize_cache(project_path)
        
        return manifest

    def _validate_dataset(self, dataset_path: Path):
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found at {dataset_path}")
        if not (dataset_path / "novel.txt").exists():
            raise ValueError(f"Dataset missing novel.txt at {dataset_path}")

    def _create_folder_structure(self, project_path: Path):
        for folder in ["cache/v1/scenes", "assets", "exports", "checkpoints"]:
            (project_path / folder).mkdir(parents=True, exist_ok=True)

    def _generate_manifest(self, project_name: str, dataset_name: str, project_path: Path) -> ProjectManifest:
        manifest = ProjectManifest(metadata=ProjectMetadata(project_name=project_name, dataset_id=dataset_name))
        with open(project_path / "project.json", "w") as f:
            f.write(manifest.model_dump_json(indent=4))
        return manifest

    def _initialize_cache(self, project_path: Path):
        # Intentionally empty for MVP; ready for cache initialization logic
        pass
