from core.prompt.ast import PromptAST


class PromptCompiler:

    def __init__(self, priority_weights=None):
        # identity ALWAYS wins
        self.priority = priority_weights or {
            "character": 1.0,
            "outfit": 0.9,
            "scene": 0.7,
            "camera": 0.6,
            "lighting": 0.5,
            "style": 0.3
        }

    def compile(self, ast: PromptAST) -> str:

        # HARD ORDERING RULE (prevents CLIP dilution issues)
        parts = []

        # 1. Identity block (never diluted)
        parts.append(f"{ast.character}, {ast.outfit}")

        # 2. Scene semantics
        parts.append(ast.scene)

        # 3. Camera system
        parts.append(f"camera: {ast.camera}")

        # 4. Lighting
        parts.append(f"lighting: {ast.lighting}")

        # 5. Style (LOW PRIORITY — intentionally last)
        parts.append(f"style: {ast.style}")

        # 6. Negative prompt (separate channel, NOT mixed)
        if ast.negative:
            parts.append(f"negative: {ast.negative}")

        return ", ".join(parts)
