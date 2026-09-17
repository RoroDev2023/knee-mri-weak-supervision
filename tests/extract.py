"""Load notebook cells and execute the extracted logic hermetically.

The notebooks are the source under test: every helper here reads the
.ipynb JSON directly, so the suite fails if a future edit regresses one
of the B1-B7 fixes. GPU loops cannot run in CI, so cells are executed
either in full (guard cells that raise early) or truncated before their
model-driving tail (see ``exec_cell_head``).
"""

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NB01 = "01_report_labeling.ipynb"
NB02 = "02_mri_feature_extraction.ipynb"


def cell_source(filename, cell_index):
    """Return the raw source of one code cell."""
    notebook = json.loads((REPO_ROOT / filename).read_text())
    cell = notebook["cells"][cell_index]
    if cell["cell_type"] != "code":
        raise ValueError(f"{filename} cell {cell_index} is not a code cell")
    return "".join(cell["source"])


def load_code_cells(filename):
    """Return ``[(cell_index, source), ...]`` for every code cell."""
    notebook = json.loads((REPO_ROOT / filename).read_text())
    return [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]


def strip_ipython_magics(source):
    """Blank out IPython shell/magic lines, keeping line numbers stable.

    A line whose first token starts with ``!`` or ``%`` is a shell escape
    or magic during notebook execution. ``!=`` and ``%=`` are Python
    operators that legitimately open continuation lines (nb01 cell 10 has
    one), so they are kept.
    """
    kept = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("!=", "%=")):
            kept.append(line)
        elif stripped.startswith(("!", "%")):
            kept.append("\n")
        else:
            kept.append(line)
    return "".join(kept)


def exec_source(source, namespace):
    """Compile (magics stripped) and exec one full cell into ``namespace``."""
    compiled = compile(strip_ipython_magics(source), "<notebook cell>", "exec")
    exec(compiled, namespace)  # noqa: S102 -- test harness for extracted notebook code
    return namespace


def exec_cell_head(source, namespace, stop_before):
    """Exec a cell's top-level statements up to the named assignment.

    The parser cells mix reusable definitions with a GPU smoke-test tail
    that cannot run hermetically (it drives the real model). The cut
    point is the first top-level assignment whose target is in
    ``stop_before`` -- e.g. ``test_row = ...`` opens the smoke-test tail
    of nb01 cell 9. Everything before it (imports, guards, prompts,
    function definitions) is executed; the model tail is not.
    """
    tree = ast.parse(strip_ipython_magics(source))
    cutoff = len(tree.body)
    for position, node in enumerate(tree.body):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if names & set(stop_before):
                cutoff = position
                break
    head = ast.Module(body=tree.body[:cutoff], type_ignores=[])
    exec(compile(head, "<notebook cell head>", "exec"), namespace)
    return namespace


def function_defs(source):
    """Map top-level function name -> AST node for one cell."""
    tree = ast.parse(strip_ipython_magics(source))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def extract_function(source, name, namespace):
    """Exec one top-level function definition from a cell into ``namespace``.

    The function's globals are ``namespace``, so callers seed it with the
    module-level names the function reads (paths, ``os``, ``pd``).
    """
    definitions = function_defs(source)
    if name not in definitions:
        raise KeyError(f"function {name!r} not found in cell")
    module = ast.Module(body=[definitions[name]], type_ignores=[])
    exec(compile(module, "<notebook function>", "exec"), namespace)
    return namespace[name]


def labels_from_nb01():
    """The twelve prediction targets, read from nb01 cell 2."""
    source = cell_source(NB01, 2)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LABELS"
            for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("LABELS definition not found in nb01 cell 2")
