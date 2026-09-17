"""B4: checkpoints are written atomically (tmp file + os.replace).

Behavioral half: the extracted helper writes a complete, parseable CSV
and leaves no temporary file behind. Structural half: os.replace is the
only path that touches the destination, so a crash mid-write can never
corrupt the previous checkpoint.
"""

import ast
import os

import pandas as pd
import pytest

from extract import NB01, cell_source, extract_function, function_defs

CHECKPOINT = "CHECKPOINT_PATH"  # nb01 cell 10 destination
STATE = "STATE_PATH"  # nb01 cell 13 destination

SAVE_HELPERS = [
    (10, "save_evaluation_checkpoint", CHECKPOINT),
    (13, "save_checkpoint", STATE),
]

# Methods that write through a path object. Read-only uses of the
# destination (with_suffix to derive the tmp path) are allowed.
WRITE_METHODS = {
    "to_csv",
    "to_pickle",
    "to_parquet",
    "to_excel",
    "to_json",
    "write_text",
    "write_bytes",
}


def _destination_uses(function_node, destination):
    """Calls that write through `destination`: as a write-method receiver,
    or as an argument (only os.replace may receive the destination)."""
    uses = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        receiver_writes = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in WRITE_METHODS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == destination
        )
        argument = any(isinstance(a, ast.Name) and a.id == destination for a in node.args) or any(
            isinstance(kw.value, ast.Name) and kw.value.id == destination
            for kw in node.keywords
        )
        if receiver_writes or argument:
            uses.append(node)
    return uses


def _is_os_replace_of_destination(node, destination):
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == destination
    )


@pytest.mark.parametrize(("cell_index", "function_name", "destination"), SAVE_HELPERS)
def test_atomic_save_writes_complete_csv(tmp_path, cell_index, function_name, destination):
    namespace = {"os": os, "pd": pd, destination: tmp_path / "checkpoint.csv"}
    save = extract_function(cell_source(NB01, cell_index), function_name, namespace)

    if function_name == "save_evaluation_checkpoint":
        save(
            pd.DataFrame(
                {
                    "StudyInstanceUID": ["study-1", "study-2", "study-3"],
                    "ParsingError": ["", "", ""],
                    "pred_ACL": [1, 0, None],
                }
            )
        )
    else:
        frame = pd.DataFrame({"StudyInstanceUID": ["study-1", "study-2"]})
        saved = save(frame, [{"StudyInstanceUID": "study-3"}])
        assert len(saved) == 3

    destination_path = namespace[destination]
    assert destination_path.exists(), f"{function_name} did not write the destination"

    restored = pd.read_csv(destination_path)
    assert len(restored) == 3, "destination CSV is incomplete"
    assert restored["StudyInstanceUID"].tolist() == ["study-1", "study-2", "study-3"]

    leftover = [entry.name for entry in tmp_path.iterdir() if ".tmp" in entry.name]
    assert leftover == [], f"temporary file left behind: {leftover}"


@pytest.mark.parametrize(("cell_index", "function_name", "destination"), SAVE_HELPERS)
def test_os_replace_is_the_only_destination_write(cell_index, function_name, destination):
    function_node = function_defs(cell_source(NB01, cell_index))[function_name]
    uses = _destination_uses(function_node, destination)
    assert uses, f"{function_name}: no writes to {destination} found"
    for node in uses:
        assert _is_os_replace_of_destination(node, destination), (
            f"{function_name}: {ast.dump(node)} writes the destination directly; "
            "it must go through the tmp file + os.replace"
        )
    assert len(uses) == 1
