"""Execute notebook code cells headlessly and optionally save every figure.

This lightweight verifier avoids requiring a running Jupyter server. It is
intended for CI/smoke validation; interactive notebook execution remains the
normal analysis workflow.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    notebook_path = args.notebook.resolve()
    if not notebook_path.is_file():
        raise FileNotFoundError(notebook_path)

    cache_dir = (args.figure_dir or notebook_path.parent / ".matplotlib_cache") / "mpl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.figure_dir:
        figure_dir = args.figure_dir.resolve()
        figure_dir.mkdir(parents=True, exist_ok=True)
        counter = itertools.count(1)

        def save_and_close(*_args: object, **_kwargs: object) -> None:
            group = next(counter)
            for item, number in enumerate(plt.get_fignums(), start=1):
                destination = figure_dir / f"figure_{group:02d}_{item:02d}.png"
                plt.figure(number).savefig(destination, dpi=120, bbox_inches="tight")
            plt.close("all")

        plt.show = save_and_close

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    executed = 0
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}:cell-{cell_number}", "exec"), namespace)
        executed += 1
    print(f"Validated {executed} code cells in {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
