from typing import Dict, Any
from core.rendering.render_graph import RenderNode, RenderArtifact, NodeRegistry
from core.domain.prompt.provider_request import ProviderRequest

class PromptCompilerNode(RenderNode):
    def get_name(self) -> str:
        return "PromptCompilerNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        plan = inputs.get("RENDER_PLAN")
        if not plan:
            return {}
        
        request: ProviderRequest = inputs.get("PROVIDER_REQUEST", RenderArtifact(kind="PROVIDER_REQUEST", data=ProviderRequest())).data
        
        # In real code, parse plan.logical into generation/conditioning strings
        request.conditioning.prompt = f"({plan.data.logical.subject}:1.2) {plan.data.logical.framing}"
        request.conditioning.negative_prompt = "bad anatomy, low quality"
        request.generation.resolution = (plan.data.physical.width, plan.data.physical.height)
        request.generation.seed = plan.data.physical.seed
        request.generation.steps = plan.data.physical.steps
        request.generation.cfg = plan.data.physical.cfg
        
        return {"PROVIDER_REQUEST": RenderArtifact(kind="PROVIDER_REQUEST", data=request)}

class BindingResolverNode(RenderNode):
    def get_name(self) -> str:
        return "BindingResolverNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        request: ProviderRequest = inputs["PROVIDER_REQUEST"].data
        # Resolve any LoRAs/embeddings from the plan
        plan = inputs.get("RENDER_PLAN")
        if plan:
            request.bindings.loras = plan.data.physical.loras
        return {"PROVIDER_REQUEST": RenderArtifact(kind="PROVIDER_REQUEST", data=request)}

class ConditionBuilderNode(RenderNode):
    def get_name(self) -> str:
        return "ConditionBuilderNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        # Would parse ShotLayout into ControlNet maps, IP-Adapter tensors, etc.
        request: ProviderRequest = inputs["PROVIDER_REQUEST"].data
        # Mocking finding conditions
        request.conditioning.controlnets["depth"] = "mock_depth_map"
        return {"PROVIDER_REQUEST": RenderArtifact(kind="PROVIDER_REQUEST", data=request)}

class GenerateNode(RenderNode):
    def __init__(self, provider=None):
        self.provider = provider

    def get_name(self) -> str:
        return "GenerateNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        request: ProviderRequest = inputs["PROVIDER_REQUEST"].data
        # Execute provider
        if self.provider:
            image = self.provider.generate(request)
        else:
            # Mock
            from PIL import Image
            image = Image.new('RGB', request.generation.resolution, color='green')
        return {"IMAGE": RenderArtifact(kind="IMAGE", data=image)}

class FaceRefinerNode(RenderNode):
    def get_name(self) -> str:
        return "FaceRefinerNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        image = inputs["IMAGE"].data
        # Mock face refinement
        return {"IMAGE": RenderArtifact(kind="IMAGE", data=image)}

class UpscaleNode(RenderNode):
    def get_name(self) -> str:
        return "UpscaleNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        request: ProviderRequest = inputs["PROVIDER_REQUEST"].data
        image = inputs["IMAGE"].data
        if request.postprocess.upscale:
            image = image.resize((image.width * 2, image.height * 2))
        return {"IMAGE": RenderArtifact(kind="IMAGE", data=image)}

class ExportNode(RenderNode):
    def get_name(self) -> str:
        return "ExportNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        image = inputs["IMAGE"].data
        # In a real export, this writes to CAS
        # For now, it simply marks the pipeline as done
        return {"FINAL_IMAGE": RenderArtifact(kind="IMAGE", data=image)}

# Register nodes
NodeRegistry.register("PromptCompilerNode", PromptCompilerNode)
NodeRegistry.register("BindingResolverNode", BindingResolverNode)
NodeRegistry.register("ConditionBuilderNode", ConditionBuilderNode)
NodeRegistry.register("GenerateNode", GenerateNode)
NodeRegistry.register("FaceRefinerNode", FaceRefinerNode)
NodeRegistry.register("UpscaleNode", UpscaleNode)
NodeRegistry.register("ExportNode", ExportNode)
