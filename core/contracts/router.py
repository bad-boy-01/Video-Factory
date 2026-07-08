class ContractRouter:
    """
    Decides which contracts apply to which stage.
    """

    def __init__(self, contract_map):
        self.contract_map = contract_map

    def get_contracts(self, stage_name):
        return self.contract_map.get(stage_name, [])
