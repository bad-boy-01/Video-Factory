import hashlib
import json
from core.contracts.base import GenerativeContract, ContractResult


class RequireCharacterHashMatch(GenerativeContract):
    """
    Ensures character identity is invariant across pipeline.
    """

    def validate(self, context, artifact):

        # Assuming context.story_bible is an object/dict with a .characters attribute
        # In a real implementation, we'd use a Pydantic model here.
        story_bible_chars = getattr(context.story_bible, "characters", {})
        final_identity = getattr(artifact, "identity_blob", {})

        expected_hash = hashlib.sha256(
            json.dumps(story_bible_chars, sort_keys=True).encode()
        ).hexdigest()

        actual_hash = hashlib.sha256(
            json.dumps(final_identity, sort_keys=True).encode()
        ).hexdigest()

        if expected_hash != actual_hash:
            return ContractResult(
                passed=False,
                message="Character identity mismatch detected",
                severity="hard",
                contract_name="RequireCharacterHashMatch"
            )

        return ContractResult(
            passed=True,
            message="Identity contract passed",
            contract_name="RequireCharacterHashMatch"
        )
