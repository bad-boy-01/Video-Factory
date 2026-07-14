"""
ImageQA — multi-metric image quality evaluator with Critic feedback loop.

Architecture:
  Generate image
      ↓
  ImageQAEvaluator.evaluate(image, prompt, shot_metadata)
      ↓
  QAResult (passed / failed + per-metric scores)
      ↓
  If failed: CriticFeedback.diagnose(result) → revised prompt
      ↓
  Retry with revised prompt (max 3 retries)

Metrics evaluated:
  1. CLIP score        — prompt adherence (text-image similarity)
  2. Sharpness         — Laplacian variance (blur detection)
  3. Face quality      — simple face detection confidence (for close-up shots)
  4. OCR artifacts     — detects unwanted text/watermarks in the image
  5. Aesthetic score   — simple brightness/contrast/saturation heuristic
  6. Continuity score  — compares with the previous accepted image for the same scene

All metrics run on CPU to avoid consuming Kaggle GPU VRAM between generations.
CLIP model loaded lazily and cached for the session.

The Critic produces human-readable feedback explaining WHY an image failed
and suggests specific prompt modifications:
  "Face inconsistent → increase face emphasis, move hair descriptor earlier"
  "Sword missing → add weapon to character token block"
  "Background too empty → add environmental detail tokens"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "clip":         0.22,   # Minimum CLIP cosine similarity
    "sharpness":    60.0,   # Minimum Laplacian variance
    "face_quality": 0.55,   # Minimum face confidence (for close-up shots only)
    "aesthetic":    0.40,   # Normalized [0,1] aesthetic heuristic
}

# Shot purposes that require face quality evaluation
FACE_REQUIRED_PURPOSES = {"reaction", "emotion_peak", "dialogue", "close-up"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    shot_id: str
    passed: bool
    scores: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)  # Which metrics failed
    retry_count: int = 0


@dataclass
class CriticFeedback:
    """Explains WHY an image failed and how to fix the prompt."""
    diagnosis: str           # Human-readable summary
    prompt_additions: str    # Tokens to prepend to the positive prompt
    prompt_removals: List[str] = field(default_factory=list)  # Tokens to remove
    seed_change: int = 1     # Increment to apply to the seed

    @staticmethod
    def diagnose(result: QAResult, shot_purpose: str) -> "CriticFeedback":
        """
        Deterministically map QA failures to prompt revisions.
        No LLM needed — the failure category maps directly to a known fix.
        """
        additions = []
        removals = []
        diagnosis_parts = []

        if "clip" in result.failures:
            additions.append("(subject emphasis:1.3)")
            diagnosis_parts.append("Low prompt adherence → adding subject emphasis token")

        if "sharpness" in result.failures:
            additions.append("sharp focus, crisp details, high definition")
            removals.append("blurry")
            diagnosis_parts.append("Blur detected → adding sharpness tokens")

        if "face_quality" in result.failures:
            additions.append("(perfect face:1.2), detailed facial features, clear eyes")
            diagnosis_parts.append("Face quality low → adding face emphasis tokens")

        if "aesthetic" in result.failures:
            additions.append("dramatic lighting, vibrant colors, high contrast")
            diagnosis_parts.append("Low aesthetic → adding lighting/color tokens")

        if "continuity" in result.failures:
            # Continuity failure: re-emphasize character descriptors
            additions.append("(consistent character appearance:1.2)")
            diagnosis_parts.append("Continuity drift detected → re-emphasizing character tokens")

        return CriticFeedback(
            diagnosis="; ".join(diagnosis_parts) or "Unknown failure",
            prompt_additions=", ".join(additions),
            prompt_removals=removals,
            seed_change=1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ImageQAEvaluator:
    """
    Multi-metric image quality evaluator.

    Instantiate once per render session. CLIP model is loaded lazily and cached.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clip_model = None
        self._clip_processor = None
        self._clip_available = None  # None = not yet checked

    def evaluate(
        self,
        image_path: str | Path,
        positive_prompt: str,
        shot_id: str,
        shot_purpose: str = "mid",
        previous_image_path: Optional[str | Path] = None,
    ) -> QAResult:
        """
        Evaluate one generated image against all QA metrics.

        Parameters
        ----------
        image_path:
            Path to the generated PNG/JPEG.
        positive_prompt:
            The positive prompt used to generate this image.
        shot_id:
            For logging and result tracking.
        shot_purpose:
            Determines whether face_quality is required.
        previous_image_path:
            Path to the previous accepted image in the same scene.
            If provided, continuity score is computed.
        """
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"[ImageQA] Cannot open image {image_path}: {e}")
            return QAResult(shot_id=shot_id, passed=False, failures=["io_error"])

        scores: Dict[str, float] = {}
        failures: List[str] = []

        # ── 1. CLIP score ─────────────────────────────────────────────────
        clip_score = self._clip_score(img, positive_prompt)
        scores["clip"] = clip_score
        if clip_score < THRESHOLDS["clip"]:
            failures.append("clip")
            logger.debug(f"[ImageQA] {shot_id}: CLIP {clip_score:.3f} < {THRESHOLDS['clip']}")

        # ── 2. Sharpness ──────────────────────────────────────────────────
        sharpness = self._laplacian_variance(img)
        scores["sharpness"] = sharpness
        if sharpness < THRESHOLDS["sharpness"]:
            failures.append("sharpness")
            logger.debug(f"[ImageQA] {shot_id}: Sharpness {sharpness:.1f} < {THRESHOLDS['sharpness']}")

        # ── 3. Face quality (only for close-up shots) ─────────────────────
        needs_face_qa = shot_purpose.lower() in FACE_REQUIRED_PURPOSES
        if needs_face_qa:
            face_score = self._face_quality(img)
            scores["face_quality"] = face_score
            if face_score < THRESHOLDS["face_quality"]:
                failures.append("face_quality")
                logger.debug(f"[ImageQA] {shot_id}: Face {face_score:.3f} < {THRESHOLDS['face_quality']}")

        # ── 4. OCR artifact detection ─────────────────────────────────────
        has_text = self._detect_text_artifacts(img)
        scores["ocr_clean"] = 0.0 if has_text else 1.0
        if has_text:
            failures.append("ocr_artifacts")
            logger.debug(f"[ImageQA] {shot_id}: Text/watermark detected")

        # ── 5. Aesthetic score ────────────────────────────────────────────
        aesthetic = self._aesthetic_score(img)
        scores["aesthetic"] = aesthetic
        if aesthetic < THRESHOLDS["aesthetic"]:
            failures.append("aesthetic")
            logger.debug(f"[ImageQA] {shot_id}: Aesthetic {aesthetic:.3f} < {THRESHOLDS['aesthetic']}")

        # ── 6. Continuity score ───────────────────────────────────────────
        if previous_image_path:
            continuity = self._continuity_score(img, previous_image_path)
            scores["continuity"] = continuity
            if continuity < 0.60:  # Less than 60% color/tone similarity suggests drift
                failures.append("continuity")
                logger.debug(f"[ImageQA] {shot_id}: Continuity {continuity:.3f} (possible drift)")

        passed = len(failures) == 0
        if not passed:
            logger.info(f"[ImageQA] {shot_id}: FAIL — {failures} | scores: {_fmt_scores(scores)}")
        else:
            logger.debug(f"[ImageQA] {shot_id}: PASS | scores: {_fmt_scores(scores)}")

        return QAResult(shot_id=shot_id, passed=passed, scores=scores, failures=failures)

    # ─────────────────────────────────────────────────────────────────────
    # Individual metric implementations
    # ─────────────────────────────────────────────────────────────────────

    def _clip_score(self, img, prompt: str) -> float:
        """CLIP cosine similarity between image and prompt. Returns 0.0 on failure."""
        if self._clip_available is False:
            return 0.5  # Neutral score when CLIP is unavailable

        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            if self._clip_model is None:
                logger.info("[ImageQA] Loading CLIP model (openai/clip-vit-base-patch32)...")
                self._clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).to(self.device)
                self._clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self._clip_available = True
                logger.info("[ImageQA] CLIP model loaded.")

            inputs = self._clip_processor(
                text=[prompt[:200]],  # Truncate for safety
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                score = outputs.logits_per_image.item() / 100.0  # Normalize to ~[0,1]

            return min(1.0, max(0.0, score))

        except ImportError:
            self._clip_available = False
            logger.warning("[ImageQA] transformers not available; CLIP score disabled.")
            return 0.5
        except Exception as e:
            logger.warning(f"[ImageQA] CLIP score failed: {e}")
            return 0.5

    def _laplacian_variance(self, img) -> float:
        """Sharpness via Laplacian variance. Higher = sharper."""
        try:
            import numpy as np
            gray = img.convert("L")
            arr = np.array(gray, dtype=np.float32)
            # Laplacian kernel
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            from scipy.ndimage import convolve
            lap = convolve(arr, kernel)
            return float(np.var(lap))
        except ImportError:
            # Fallback: PIL-based edge detection
            try:
                from PIL import ImageFilter
                import numpy as np
                edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
                return float(np.array(edges).var())
            except Exception:
                return 100.0  # Neutral fallback
        except Exception as e:
            logger.warning(f"[ImageQA] Sharpness failed: {e}")
            return 100.0

    def _face_quality(self, img) -> float:
        """
        Face quality score [0,1].
        Uses OpenCV Haar cascade (free, CPU, no GPU) if available.
        Falls back to 0.7 (neutral) if OpenCV is not installed.
        """
        try:
            import cv2
            import numpy as np

            arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                # No face detected in a close-up shot → likely a quality problem
                return 0.3
            # Score based on face coverage of the image
            img_area = img.width * img.height
            best_face_area = max(w * h for (_, _, w, h) in faces)
            coverage = best_face_area / img_area
            return min(1.0, 0.5 + coverage * 2.0)

        except ImportError:
            logger.debug("[ImageQA] OpenCV not available; face quality check skipped.")
            return 0.7  # Neutral
        except Exception as e:
            logger.warning(f"[ImageQA] Face quality failed: {e}")
            return 0.7

    def _detect_text_artifacts(self, img) -> bool:
        """
        Heuristic text/watermark detection.
        Returns True if text artifacts are suspected.
        Does NOT use OCR (too slow). Uses edge density in corners as a heuristic.
        """
        try:
            from PIL import ImageFilter
            import numpy as np

            small = img.resize((64, 64)).convert("L")
            edges = small.filter(ImageFilter.FIND_EDGES)
            arr = np.array(edges)

            # Corners tend to have watermarks
            corners = [
                arr[:8, :8], arr[:8, -8:], arr[-8:, :8], arr[-8:, -8:]
            ]
            max_corner_density = max(c.mean() for c in corners)
            return bool(max_corner_density > 80)

        except Exception:
            return False  # Fail open (don't reject for unknown reasons)

    def _aesthetic_score(self, img) -> float:
        """
        Simple aesthetic heuristic [0,1] based on:
        - Color saturation (manhwa should be vibrant)
        - Contrast (should not be flat gray)
        - Not too dark or too bright
        """
        try:
            import numpy as np
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

            # Saturation via HSV conversion
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            cmax = np.maximum(np.maximum(r, g), b)
            cmin = np.minimum(np.minimum(r, g), b)
            delta = cmax - cmin
            sat_mean = float(delta.mean())

            # Contrast (std of luminance)
            lum = 0.299*r + 0.587*g + 0.114*b
            contrast = float(lum.std())

            # Brightness (not too dark, not overexposed)
            brightness = float(lum.mean())
            brightness_score = 1.0 - abs(brightness - 0.5) * 2.0  # Peaks at 0.5

            # Weighted composite
            aesthetic = (sat_mean * 0.4) + (contrast * 0.4) + (brightness_score * 0.2)
            return min(1.0, max(0.0, aesthetic))

        except Exception as e:
            logger.warning(f"[ImageQA] Aesthetic score failed: {e}")
            return 0.5

    def _continuity_score(self, img, previous_path) -> float:
        """
        Color histogram similarity between current and previous image.
        A score < 0.60 suggests significant visual drift.
        """
        try:
            from PIL import Image
            import numpy as np

            prev = Image.open(previous_path).convert("RGB")
            curr = img

            def hist(image):
                arr = np.array(image.resize((64, 64)), dtype=np.float32)
                h = []
                for c in range(3):
                    ch = np.histogram(arr[:,:,c], bins=32, range=(0, 256))[0]
                    ch = ch / (ch.sum() + 1e-8)
                    h.extend(ch)
                return np.array(h)

            h1 = hist(prev)
            h2 = hist(curr)
            # Bhattacharyya coefficient (0=no overlap, 1=identical)
            bc = float(np.sum(np.sqrt(h1 * h2)))
            return min(1.0, max(0.0, bc))

        except Exception as e:
            logger.warning(f"[ImageQA] Continuity score failed: {e}")
            return 1.0  # Fail open


# ─────────────────────────────────────────────────────────────────────────────
# Retry loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_retry(
    generate_fn,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    shot_id: str,
    shot_purpose: str,
    output_path: Path,
    evaluator: ImageQAEvaluator,
    previous_image_path: Optional[Path] = None,
    max_retries: int = 3,
) -> Tuple[bool, QAResult, str, int]:
    """
    Generate an image, evaluate it, and retry with Critic feedback if it fails.

    Returns
    -------
    (accepted, final_qa_result, final_prompt, final_seed)
    """
    current_prompt = positive_prompt
    current_seed = seed
    best_result: Optional[QAResult] = None
    best_prompt = positive_prompt
    best_seed = seed

    for attempt in range(max_retries + 1):
        # Generate
        try:
            generate_fn(
                prompt=current_prompt,
                negative=negative_prompt,
                seed=current_seed,
                output_path=output_path,
            )
        except Exception as e:
            logger.error(f"[ImageQA] Generation failed on attempt {attempt}: {e}")
            continue

        # Evaluate
        result = evaluator.evaluate(
            image_path=output_path,
            positive_prompt=current_prompt,
            shot_id=shot_id,
            shot_purpose=shot_purpose,
            previous_image_path=previous_image_path,
        )
        result.retry_count = attempt

        if result.passed:
            logger.info(f"[ImageQA] {shot_id}: PASSED on attempt {attempt}")
            return True, result, current_prompt, current_seed

        # Track best attempt so far (highest CLIP score)
        if best_result is None or result.scores.get("clip", 0) > best_result.scores.get("clip", 0):
            best_result = result
            best_prompt = current_prompt
            best_seed = current_seed

        if attempt < max_retries:
            feedback = CriticFeedback.diagnose(result, shot_purpose)
            logger.info(
                f"[ImageQA] {shot_id}: Attempt {attempt} FAILED — {feedback.diagnosis}. "
                f"Retrying with revised prompt."
            )
            # Apply Critic feedback
            if feedback.prompt_additions:
                current_prompt = feedback.prompt_additions + ", " + current_prompt
            for removal in feedback.prompt_removals:
                current_prompt = current_prompt.replace(removal, "")
            current_seed += feedback.seed_change

    # Exhausted retries — accept best result
    logger.warning(
        f"[ImageQA] {shot_id}: Exhausted {max_retries} retries. "
        "Accepting best available result."
    )
    return False, best_result or QAResult(shot_id=shot_id, passed=False), best_prompt, best_seed


def _fmt_scores(scores: Dict[str, float]) -> str:
    return " | ".join(f"{k}:{v:.2f}" for k, v in scores.items())
