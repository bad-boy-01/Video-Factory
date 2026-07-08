import pkgutil
import importlib

def test_pipeline_imports():
    """
    CI smoke test to verify all modules in the 'core' package can be imported
    without SyntaxErrors, NameErrors, or missing dependencies.
    """
    for module in pkgutil.walk_packages(["core"], prefix="core."):
        try:
            importlib.import_module(module.name)
        except Exception as e:
            raise AssertionError(f"Failed to import {module.name}: {e}")
