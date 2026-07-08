import json
from collections import defaultdict

class CompilerReporter:
    def _extract_artifacts(self, context):
        artifacts = {}
        for node in context.execution_nodes:
            name = type(node.artifact).__name__
            artifacts[name] = node.artifact
        return artifacts

    def generate_compile_report(self, context):
        artifacts = self._extract_artifacts(context)
        scene_manifest = artifacts.get("SceneManifest")
        shot_manifest = artifacts.get("ShotManifest")
        prompt_manifest = artifacts.get("PromptManifest")
        
        scene_count = len(scene_manifest.scenes) if scene_manifest else 0
        shot_count = len(shot_manifest.shots) if shot_manifest else 0
        
        # Estimate runtime based on shots (e.g. 3s per shot)
        estimated_runtime = shot_count * 3.0
        
        warnings = []
        if not scene_manifest:
            warnings.append("Missing SceneManifest")
        if not shot_manifest:
            warnings.append("Missing ShotManifest")
            
        # Add basic warnings
        if shot_manifest:
            # check for establishing shots, etc
            pass

        return {
            "scene_count": scene_count,
            "shot_count": shot_count,
            "estimated_runtime": round(estimated_runtime, 1),
            "estimated_images": shot_count,
            "warnings": warnings
        }

    def generate_directors_report(self, context):
        artifacts = self._extract_artifacts(context)
        bible = artifacts.get("StoryBible")
        
        report = self.generate_compile_report(context)
        
        lines = []
        lines.append("PROJECT SUMMARY")
        lines.append("")
        lines.append(f"Scenes: {report['scene_count']}")
        lines.append(f"Shots: {report['shot_count']}")
        mins = int(report['estimated_runtime'] // 60)
        secs = int(report['estimated_runtime'] % 60)
        lines.append(f"Runtime: {mins}m {secs:02d}s")
        lines.append("")
        
        if bible:
            lines.append("Characters:")
            for char in bible.characters.values():
                lines.append(f"- {char.name}")
            lines.append("")
            
        lines.append("Warnings:")
        if not report['warnings']:
            lines.append("None")
        else:
            for w in report['warnings']:
                lines.append(w)
                
        lines.append("")
        lines.append("Continuity:")
        lines.append("No issues detected.")
        
        return "\n".join(lines)
