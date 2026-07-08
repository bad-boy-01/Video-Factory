from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.base import DomainModel
from pydantic import BaseModel
from typing import List
import hashlib

class TextChunk(BaseModel):
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    chapter: int = 1

class ChunkerManifest(DomainModel):
    chunks: List[TextChunk] = []

class ChunkerStage(PipelineStage):
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def get_providers(self) -> list:
        return []

    def execute(self, context) -> StageResult:
        source_text = context.project_manifest.source_text
        
        chunks = []
        start = 0
        text_len = len(source_text)
        chapter = 1
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Snap to paragraph boundary (only when not already at the end)
            if end < text_len:
                next_para = source_text.find('\n\n', end - 500, end + 500)
                if next_para != -1:
                    end = next_para + 2
                    
            chunk_text = source_text[start:end]
            chunk_id = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:8]
            
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_char=start,
                end_char=end,
                chapter=chapter
            ))

            # Simple chapter increment mock
            if "Chapter" in chunk_text:
                chapter += 1

            # If we've consumed to the end, stop — otherwise the overlap
            # would set start < text_len again causing an infinite loop.
            if end >= text_len:
                break

            start = end - self.overlap
            
        manifest = ChunkerManifest(chunks=chunks, generator="ChunkerStage", generator_version="1.0")
        node = ExecutionNode(artifact=manifest, stage_name="ChunkerStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"num_chunks": len(chunks)},
            metadata={}
        )
