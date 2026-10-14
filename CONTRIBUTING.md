# Contributing

Install the development and documentation dependencies from the repository root:

```bash
python -m pip install -e '.[dev,docs]'
```

Run checks locally before pushing:

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python -m sphinx -W --keep-going -b html docs/source docs/build/html
python scripts/check_docs.py docs/build/html
```

Preview the documentation with `python -m http.server 8001 --directory docs/build/html`.
When removing a page, use an empty build directory so obsolete HTML and downloads
are not retained.

## Documentation

User guides explain installation, API behavior, and reproducible workflows.
Tutorials demonstrate one task at a time, with explanations in code comments.
Keep development status, planning notes, validation reports, and benchmark claims
out of the user documentation and README. Design discussions belong in `design/`;
experiment records belong with their experiments.

To refresh notebook outputs, install `.[tutorials]` and run
`python docs/execute_notebooks.py`. The script executes each notebook in a fresh
kernel. Sphinx renders the saved outputs without executing notebooks.

Use [STYLE.md](STYLE.md) when contributing diagnostic plots.

## Publishing

GitHub Actions builds the documentation and gallery, then publishes successful
builds from `main`. Pull requests produce a downloadable site artifact. Tests and
documentation checks run locally; the publishing workflow does not run them.
