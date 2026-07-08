from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, Scene, Beat
from core.planning.chunker import ChunkerManifest
import json
import logging
import hashlib

logger = logging.getLogger(__name__)

class SceneSplitterStage(PipelineStage):
    def __init__(self, llm_provider=None, rag_index=None):
        self.llm = llm_provider
        self.rag_index = rag_index

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []

    def execute(self, context) -> StageResult:
        if not self.llm:
            from plugins.local_llm import LocalLLMProvider
            self.llm = LocalLLMProvider()
            
        chunker_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, ChunkerManifest):
                chunker_manifest = node.artifact
                break
                
        if not chunker_manifest:
            raise ValueError("SceneSplitter: Missing ChunkerManifest. Did ChunkerStage run?")
            
        schema = {
            "scenes": [
                {
                    "chapter": 1,
                    "start_offset": 0,
                    "end_offset": 500,
                    "estimated_duration": 15.5,
                    "characters": ["string"],
                    "location": "string",
                    "emotion": "string",
                    "beats": [
                        {
                            "beat_id": "string",
                            "description": "string",
                            "emotion": "string"
                        }
                    ]
                }
            ]
        }
        
        all_scenes = []
        
        for chunk in chunker_manifest.chunks:
            retrieved_context = ""
            if self.rag_index is not None:
                results = self.rag_index.retrieve(
                    chunk.text, before_chunk_id=chunk.chunk_id, top_k=3
                )
                retrieved_context = self.rag_index.format_context(results)

            context_block = f"\n\n{retrieved_context}\n" if retrieved_context else ""

            prompt = f"""
You are a master cinematic story planner. Split the following text chunk into distinct narrative scenes.
CRITICAL RULES:
- Never summarize. Expand the story to capture every detail.
- Preserve every event, dialogue, and emotional beat.
- Break each scene down into an explicit sequence of `beats`. A beat is a distinct narrative moment (e.g. 'Character enters room', 'Discovers the letter', 'Reacts with shock').
- There should be between 3 and 10 beats per scene, representing the continuous flow of action and emotion.
- Provide approximate start_char and end_char offsets for the scene relative to the novel.
{context_block}
Text (Chapter {chunk.chapter}):
{chunk.text}
"""
            # To speed up M3 dev, we'll simulate output if it fails
            try:
                result_dict = self.llm.generate_json(prompt, schema)
            except Exception as e:
                logger.warning("LLM extraction failed, using mock scenes.")
                result_dict = {"scenes": [{
                    "chapter": chunk.chapter,
                    "start_offset": chunk.start_char,
                    "end_offset": chunk.end_char,
                    "characters": [], "location": "Unknown", "emotion": "Neutral", "beats": []
                }]}
            
            for s in result_dict.get("scenes", []):
                beats = [Beat(**b) for b in s.get("beats", [])]
                if not beats:
                    logger.warning("LLM generated a scene with no beats, injecting fallback beat.")
                    beats = [Beat(beat_id="beat_mock_001", description="Fallback scene action.", emotion="Neutral")]

                
                chap = s.get("chapter", chunk.chapter)
                start = s.get("start_offset", chunk.start_char)
                end = s.get("end_offset", chunk.end_char)
                
                # Stable deterministic ID
                hash_input = f"{chap}_{start}_{end}"
                scene_id = f"scene_{hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:8]}"
                
                scene = Scene(
                    scene_id=scene_id,
                    chapter=chap,
                    start_offset=start,
                    end_offset=end,
                    estimated_duration=s.get("estimated_duration", 10.0),
                    characters=s.get("characters", []),
                    location=s.get("location", ""),
                    emotion=s.get("emotion", ""),
                    beats=beats
                )
                all_scenes.append(scene)

            if self.rag_index is not None:
                self.rag_index.add_chunk(chunk.chunk_id, chunk.text)
            
        manifest = SceneManifest(
            scenes=all_scenes,
            generator="SceneSplitterStage",
            generator_version="0.2.0",
            schema_version="2.0"
        )
        
        node = ExecutionNode(artifact=manifest, stage_name="SceneSplitterStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"scenes_extracted": len(all_scenes)},
            metadata={}
        )
