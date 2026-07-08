import logging
from typing import List, Protocol

logger = logging.getLogger(__name__)

class ResourceProvider(Protocol):
    def load(self) -> None:
        ...
    def unload(self) -> None:
        ...

class ResourceSession:
    """
    Context manager to hold models resident in memory across multiple stages.
    """
    def __init__(self, resources: List[ResourceProvider]):
        self.resources = resources

    def __enter__(self):
        logger.info("[ResourceSession] Acquiring resources...")
        for resource in self.resources:
            if hasattr(resource, "load"):
                resource.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("[ResourceSession] Releasing resources...")
        for resource in self.resources:
            if hasattr(resource, "unload"):
                resource.unload()
