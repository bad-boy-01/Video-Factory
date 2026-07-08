from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ContractResult:
    passed: bool
    message: str
    severity: str = "hard"  # hard | soft
    contract_name: str = ""


class GenerativeContract(ABC):
    """
    Absolute truth gate.
    No ML. No probabilities. No baselines.
    """

    @abstractmethod
    def validate(self, context: Any, artifact: Any) -> ContractResult:
        pass
