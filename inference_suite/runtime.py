"""Read-only runtime identity, including editable installs without wheel metadata."""
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

from .config import digest


def info(binary):
    result = {"binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest()}
    mapping = importlib.metadata.packages_distributions()
    for module in ("ais_bench", "transformers"):
        spec = importlib.util.find_spec(module)
        if spec is None:
            raise ValueError(f"required module not installed in configured Python: {module}")
        distributions = mapping.get(module, [])
        versions = {name: importlib.metadata.version(name) for name in distributions}
        files = {}
        for location in spec.submodule_search_locations or []:
            root = Path(location)
            # Editable AISBench installs can have no distribution metadata at all.
            # Hash Python source to detect behavior changes within the same version.
            if module == "ais_bench":
                files.update({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(root.rglob("*.py"))})
        result[module] = {"versions": versions, "source_sha256": digest(files) if files else None}
    return result
