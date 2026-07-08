from core.contracts.base import GenerativeContract, ContractResult


class RequireFrameCompleteness(GenerativeContract):

    def validate(self, context, artifact):

        ast_beats = getattr(context.ast, "beats", [])
        artifact_frames = getattr(artifact, "frames", [])
        
        expected = len(ast_beats)
        actual = len(artifact_frames)

        if actual != expected:
            return ContractResult(
                passed=False,
                message=f"Frame mismatch: expected {expected}, got {actual}",
                severity="hard",
                contract_name="RequireFrameCompleteness"
            )

        return ContractResult(
            passed=True,
            message="Frame completeness OK",
            contract_name="RequireFrameCompleteness"
        )
