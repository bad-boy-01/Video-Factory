class ContractFailure(Exception):
    def __init__(self, failures):
        self.failures = failures
        super().__init__("GENERATIVE CONTRACT FAILURE:\n" + "\n".join(failures))


class ContractEngine:
    """
    Runs all hard gates before evaluation or CI.
    """

    def __init__(self, contracts):
        self.contracts = contracts

    def run(self, context, artifact):
        failures = []

        for contract in self.contracts:
            result = contract.validate(context, artifact)

            if not result.passed and result.severity == "hard":
                prefix = f"[{result.contract_name}] " if getattr(result, 'contract_name', '') else ""
                failures.append(f"{prefix}{result.message}")

        if failures:
            raise ContractFailure(failures)

        return True
