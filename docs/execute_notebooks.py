from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


paths = sorted((Path(__file__).parent / "source/notebooks").glob("*.ipynb"))
for path in paths:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=120, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent.resolve())}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()
    nbformat.write(notebook, path)
    print(f"Executed {path.name}", flush=True)
