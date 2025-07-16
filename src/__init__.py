# CTU-FlowRAG package root
__all__ = []
__version__ = "0.1.0"

# The tests sometimes import via both `prep.xxx` and `src.prep.xxx`
# When the project root is on the PYTHONPATH, `prep` resolves directly.
# The alias below lets the shorter import path work even when `src` is already imported.
import importlib
import sys

for _subpkg in (
    "prep",
    "ctu",
    "role",
    "rag",
    "prompt",
    "io",
    "cdge",
    "retriever",
    "flow",
    "poster",
    "utils",
):
    try:
        sys.modules.setdefault(_subpkg, importlib.import_module(f"src.{_subpkg}"))
    except ModuleNotFoundError:
        # will be created later when the subpackage is imported the first time
        pass
