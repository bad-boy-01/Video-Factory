from typing import Protocol, Iterator, List
from pathlib import Path
import json
import codecs
import re

from core.domain.dataset import DatasetManifest, ValidationResult
from core.domain.story import Chapter, Scene, Beat

class ChapterSplitter(Protocol):
    def split(self, text: str) -> List[Chapter]:
        ...

class RegexChapterSplitter:
    """A basic regex-based chapter splitter for MVP."""
    def split(self, text: str) -> List[Chapter]:
        # Simple heuristic: split by lines that start with Chapter or large gaps
        # For MVP, we'll split by double newline as scenes, and treat the whole text as one chapter 
        # if no explicit "Chapter X" is found, or split by "Chapter X".
        chapters = []
        
        # Very simple fallback logic for testing purposes
        lines = text.split("\n\n")
        beats = [Beat(text=line.strip()) for line in lines if line.strip()]
        if beats:
            scene = Scene(beats=beats)
            chapters.append(Chapter(title="Chapter 1", scenes=[scene]))
            
        return chapters

class DatasetValidator:
    def validate(self, dataset_path: Path) -> ValidationResult:
        errors = []
        warnings = []
        
        if not dataset_path.exists() or not dataset_path.is_dir():
            errors.append(f"Dataset path {dataset_path} does not exist or is not a directory.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        novel_file = dataset_path / "novel.txt"
        if not novel_file.exists():
            errors.append("Missing novel.txt file.")
        elif novel_file.stat().st_size == 0:
            errors.append("novel.txt is empty.")

        metadata_file = dataset_path / "metadata.json"
        if not metadata_file.exists():
            warnings.append("Missing metadata.json file. Defaults will be used.")
            
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

class DatasetProvider(Protocol):
    def load_manifest(self) -> DatasetManifest: ...
    def load_novel(self) -> str: ...
    def load_chapter(self, chapter_index: int) -> Chapter: ...
    def iter_chapters(self) -> Iterator[Chapter]: ...
    def validate(self) -> ValidationResult: ...

class LocalDatasetProvider:
    def __init__(self, dataset_path: Path, splitter: ChapterSplitter = None):
        self.dataset_path = dataset_path
        self.validator = DatasetValidator()
        self.splitter = splitter or RegexChapterSplitter()
        self._manifest: DatasetManifest | None = None
        self._encoding = "utf-8"

    def validate(self) -> ValidationResult:
        return self.validator.validate(self.dataset_path)

    def _detect_encoding(self, file_path: Path) -> str:
        # Check BOM or fallback
        raw = file_path.read_bytes()
        if raw.startswith(codecs.BOM_UTF8):
            return "utf-8-sig"
        if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
            return "utf-16"
        
        try:
            raw.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "windows-1252"

    def load_manifest(self) -> DatasetManifest:
        if self._manifest:
            return self._manifest
            
        metadata_file = self.dataset_path / "metadata.json"
        if metadata_file.exists():
            try:
                data = json.loads(metadata_file.read_text(encoding="utf-8"))
                self._manifest = DatasetManifest(**data)
            except Exception:
                self._manifest = DatasetManifest()
        else:
            self._manifest = DatasetManifest()
            
        novel_file = self.dataset_path / "novel.txt"
        if novel_file.exists():
            self._encoding = self._detect_encoding(novel_file)
            self._manifest.encoding = self._encoding
            
        return self._manifest

    def load_novel(self) -> str:
        novel_file = self.dataset_path / "novel.txt"
        if not novel_file.exists():
            raise FileNotFoundError("novel.txt not found in dataset")
        
        if not self._manifest:
            self.load_manifest()
            
        return novel_file.read_text(encoding=self._encoding)

    def iter_chapters(self) -> Iterator[Chapter]:
        text = self.load_novel()
        chapters = self.splitter.split(text)
        for chap in chapters:
            yield chap

    def load_chapter(self, chapter_index: int) -> Chapter:
        # In a real lazy-loader, this wouldn't parse the whole novel if it can index it.
        # But for this MVP architecture, it splits on the fly.
        chapters = list(self.iter_chapters())
        if chapter_index < 0 or chapter_index >= len(chapters):
            raise IndexError("Chapter index out of range")
        return chapters[chapter_index]
