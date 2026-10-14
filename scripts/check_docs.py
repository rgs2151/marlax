from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import sys


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        self.references.extend(value for name, value in attrs if name in ("href", "src") and value)


root = Path(sys.argv[1]).resolve()
notebooks = sorted(Path("docs/source/notebooks").glob("*.ipynb"))
assert len(notebooks) == 6
for path in notebooks:
    notebook = json.loads(path.read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert cells and all(cell["execution_count"] is not None for cell in cells), path
    assert all(output["output_type"] != "error" for cell in cells for output in cell["outputs"]), path
    assert any(cell["outputs"] for cell in cells), path
    assert (root / "notebooks" / (path.stem + ".html")).is_file(), path
    assert list((root / "_downloads").rglob(path.name)), path

for name in ("index", "guide", "tutorials", "inventory", "api"):
    assert (root / f"{name}.html").is_file(), name

for path in root.rglob("*.html"):
    if "_static" in path.relative_to(root).parts:
        continue
    parser = Links()
    parser.feed(path.read_text())
    for reference in parser.references:
        url = urlsplit(reference)
        if url.scheme or url.netloc or not url.path:
            continue
        target = path.parent / unquote(url.path)
        assert target.exists(), (path, reference)

assert list((root / "_images").glob("*.png"))
print(f"Checked {len(notebooks)} saved tutorials, downloads, images, and local document links.")
