import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import sys

from plugins.interfaces import LLMProvider
from plugins.json_utils import extract_json


class LocalLLMProvider(LLMProvider):

    def __init__(self, model_id="Qwen/Qwen1.5-4B-Chat", cache_dir=None):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        # Falls back to a cwd-relative path if the caller doesn't scope this
        # to a specific project's workspace.
        self.cache_dir = cache_dir or "workspace/cache/llm"

    # -------------------------
    # KAGGLE LOCAL PATH RESOLVER
    # -------------------------
    def _find_kaggle_model(self) -> tuple:
        """
        When running on Kaggle with a model dataset attached, the weights are
        in /kaggle/input/<slug>/ which is READ-ONLY.  We must load directly
        from that path using local_files_only=True — never set HF_HOME there.

        Strategy:
          1. Check KAGGLE_LLM_INPUT env var (set by Cell 1 of kaggle_notebook.py)
          2. Derive slug from model_id  (e.g. Qwen/Qwen1.5-4B-Chat → qwen1-5-4b-chat)
          3. Walk up to 4 levels deep looking for config.json
          4. Return (resolved_path, True) if found, else (self.model_id, False)
        """
        import os
        from pathlib import Path

        # Candidate root directories to search
        candidates = []
        env_path = os.environ.get("KAGGLE_LLM_INPUT", "")
        if env_path:
            candidates.append(env_path)

        # Also try the slug derived from model_id
        slug = self.model_id.lower().replace("/", "-").replace(".", "-").replace("_", "-")
        candidates.append(f"/kaggle/input/{slug}")

        for base in candidates:
            base_path = Path(base)
            if not base_path.is_dir():
                continue
            # Search up to 4 levels for config.json
            for depth in range(5):
                pattern = "/".join(["*"] * depth) + "/config.json" if depth > 0 else "config.json"
                matches = list(base_path.glob(pattern))
                if matches:
                    model_dir = str(matches[0].parent)
                    print(f"[LLM] Found Kaggle model at: {model_dir}")
                    return model_dir, True

        return self.model_id, False

    # -------------------------
    # LOAD MODEL (VRAM CONTROLLED)
    # -------------------------
    def initialize(self):
        print("[LLM] Initializing tokenizer and config (No VRAM penalty)...")
        model_path, local_only = self._find_kaggle_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )

    def load(self):
        if self.model is not None:
            print("[Resource] Reusing resident LLM.")
            return

        print("[Resource] Loading LLM weights into VRAM...")
        if not self.tokenizer:
            self.initialize()

        model_path, local_only = self._find_kaggle_model()

        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            print("[LLM] Using bitsandbytes 4-bit quantization.")
        except Exception as e:
            bnb_config = None
            print(f"[LLM] WARNING: bitsandbytes unavailable ({e}). Falling back to fp16.")

        if bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
            )

    def unload(self):
        from core.utils.vram import flush_vram
        print("[Resource] Unloading LLM...")
        if self.model:
            del self.model
            self.model = None
        flush_vram("LLM unloaded")

    def shutdown(self):
        print("[LLM] Shutting down tokenizer...")
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None


    # -------------------------
    # JSON GENERATION CORE
    # -------------------------
    def generate_json(self, prompt: str, schema: dict) -> dict:
        full_prompt = f"""You are a strict JSON generator. Output ONLY a single valid JSON object.
Do NOT output a JSON array at the top level.
Do NOT include any explanation, markdown, or code fences.
The response must start with {{ and end with }}.

SCHEMA (the exact keys your object must contain):
{json.dumps(schema, indent=2)}

TASK:
{prompt}

JSON OUTPUT:
"""

        temperature = 0.3
        top_p = 0.9
        
        # LLM Response Cache with PromptFingerprint
        import hashlib
        from pathlib import Path
        from plugins.interfaces import PromptFingerprint
        
        prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()
        model_hash = hashlib.sha256(self.model_id.encode()).hexdigest()
        sampling_hash = hashlib.sha256(f"temp={temperature}_top_p={top_p}_max_tokens=512_seed=42".encode()).hexdigest()
        schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        
        fingerprint = PromptFingerprint(
            provider_name="LocalLLMProvider",
            provider_version="transformers-pipeline",
            prompt_hash=prompt_hash,
            model_hash=model_hash,
            sampling_hash=sampling_hash,
            schema_hash=schema_hash
        )
        
        cache_key = fingerprint.key
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            print(f"[LLM] Prompt Cache HIT ({cache_key[:8]})", flush=True)
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        n_prompt_tokens = inputs.input_ids.shape[1]
        print(f"[LLM] Generating... ({n_prompt_tokens} prompt tokens) (Cache MISS: {cache_key[:8]})", flush=True)

        output = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

        # Decode ONLY the newly generated tokens (strip the echoed input prompt)
        input_length = inputs.input_ids.shape[1]
        new_tokens = output[0][input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"[LLM] Raw output ({len(decoded)} chars): {decoded[:120]!r}", flush=True)

        json_text = extract_json(decoded)
        result = json.loads(json_text)
        
        # Validate that the extracted JSON has the expected top-level keys
        # This prevents the extractor from silently latching onto an inner JSON object (like a single Beat) if the outer JSON is truncated.
        missing_keys = [k for k in schema.keys() if k not in result]
        if missing_keys:
            raise ValueError(f"Extracted JSON is missing required schema keys: {missing_keys}")
        
        # Save to LLM Cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

