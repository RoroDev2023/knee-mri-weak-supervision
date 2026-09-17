"""Structural AST checks (B3, B5, B7) and whole-repo compilation.

These guards assert the fixed code shape, not runtime behavior: the GPU
loops cannot run hermetically, but their structure is exactly what the
B3/B5/B7 fixes changed.
"""

import ast
import py_compile
from unittest import mock

import pandas as pd
import pytest

from extract import (
    NB01,
    NB02,
    cell_source,
    exec_cell_head,
    function_defs,
    labels_from_nb01,
    load_code_cells,
    strip_ipython_magics,
)

LABELS = labels_from_nb01()


def _assignments_to(tree, target_name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == target_name for target in node.targets)
    ]


def test_cell10_resume_set_excludes_errored_rows():
    """B3: the resume set keeps only rows with an empty ParsingError."""
    tree = ast.parse(strip_ipython_magics(cell_source(NB01, 10)))
    assignments = _assignments_to(tree, "completed_studies")
    assert assignments, "completed_studies assignment not found in nb01 cell 10"
    for assignment in assignments:
        value = assignment.value
        fresh_start = (
            isinstance(value, ast.Set) and not value.elts
        ) or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "set"
            and not value.args
            and not value.keywords
        )
        if fresh_start:
            continue  # no checkpoint yet: an empty resume set is correct
        strings = {
            node.value
            for node in ast.walk(assignment)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "ParsingError" in strings, (
            "nb01 cell 10: the resume set no longer filters on ParsingError — "
            "errored rows would be skipped forever instead of retried"
        )
        filters_empty = any(
            isinstance(node, ast.Compare)
            and any(isinstance(op, ast.Eq) for op in node.ops)
            and any(
                isinstance(comparator, ast.Constant) and comparator.value == ""
                for comparator in node.comparators
            )
            for node in ast.walk(assignment)
        )
        assert filters_empty, (
            "nb01 cell 10: only rows with an empty ParsingError may count as completed"
        )


def test_cell13_loop_is_wrapped_in_try_finally_that_saves():
    """B5: interrupting the extraction loop must still save the buffer."""
    tree = ast.parse(strip_ipython_magics(cell_source(NB01, 13)))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Try) and node.finalbody):
            continue
        finally_saves = any(
            isinstance(leaf, ast.Call)
            and isinstance(leaf.func, ast.Name)
            and leaf.func.id == "save_checkpoint"
            for statement in node.finalbody
            for leaf in ast.walk(statement)
        )
        body_has_loop = any(isinstance(child, ast.For) for child in node.body)
        if finally_saves and body_has_loop:
            return
    raise AssertionError(
        "nb01 cell 13: the extraction loop is not wrapped in a try/finally "
        "whose finally calls save_checkpoint"
    )


@pytest.mark.parametrize(("filename", "cell_index"), [(NB01, 4), (NB02, 4)])
def test_figure_cells_end_with_plt_close(filename, cell_index):
    """B7: plt.close(fig) after plt.show() so re-runs don't leak figures."""
    tree = ast.parse(strip_ipython_magics(cell_source(filename, cell_index)))
    assert tree.body, f"{filename} cell {cell_index} has no statements"
    last = tree.body[-1]
    closes_figure = (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Attribute)
        and last.value.func.attr == "close"
        and isinstance(last.value.func.value, ast.Name)
        and last.value.func.value.id == "plt"
        and any(isinstance(arg, ast.Name) and arg.id == "fig" for arg in last.value.args)
    )
    assert closes_figure, f"{filename} cell {cell_index} does not end with plt.close(fig)"


def _outcome(function, raw):
    """Normalize a probe to a comparable result-or-exception tuple."""
    try:
        return ("parsed", function(raw))
    except Exception as error:  # the probes intentionally include failures
        return ("raised", type(error).__name__, str(error))


def _parser_namespace():
    return {
        "model": mock.MagicMock(name="model"),
        "tokenizer": mock.MagicMock(name="tokenizer"),
        "LABELS": LABELS,
        "report_analysis": pd.DataFrame(
            {
                "StudyInstanceUID": ["study-0"],
                "Report": ["normal study"],
                **{label: [0] for label in LABELS},
            }
        ),
    }


def test_parser_copies_stay_identical_between_cells_9_and_12():
    """Cell 12's redefined copies must behave identically to cell 9's.

    extract_json_object is compared AST-exact; parse_model_response is
    compared behaviorally (the two copies differ in docstring and local
    naming, so text equality is the wrong invariant for it).
    """
    nine_source = cell_source(NB01, 9)
    twelve_source = cell_source(NB01, 12)
    nine = function_defs(nine_source)
    twelve = function_defs(twelve_source)
    for name in ("extract_json_object", "parse_model_response"):
        assert name in nine, f"{name} missing from cell 9"
        assert name in twelve, f"{name} missing from cell 12"
    assert ast.dump(nine["extract_json_object"]) == ast.dump(twelve["extract_json_object"]), (
        "the extract_json_object copies in cells 9 and 12 diverged"
    )

    nine_namespace = exec_cell_head(
        nine_source, _parser_namespace(), stop_before={"test_row"}
    )
    twelve_namespace = exec_cell_head(
        twelve_source, _parser_namespace(), stop_before={"batch_test_rows"}
    )
    nine_parse = nine_namespace["parse_model_response"]
    twelve_parse = twelve_namespace["parse_model_response"]

    probes = [
        '{"ACL": 1, "MCL": 0}',
        '```json\n{"ACL": 1}\n```',
        'The findings are {"ACL": 1} and {review} noted {today}.',
        '{"ACL": 0} {"ACL": 1}',
        '{"MCL": 0, "ACL": "note } inside"} trailing { prose',
        "I could not produce JSON.",
        '{"ACL": 1',
        '{"ACL": 5, "Fracture": "tear"}',
    ]
    for probe in probes:
        assert _outcome(nine_parse, probe) == _outcome(twelve_parse, probe), (
            f"cells 9 and 12 disagree on {probe!r}"
        )


def test_nb01_checkpoint_writes_use_os_replace():
    for cell_index in (10, 13):
        tree = ast.parse(strip_ipython_magics(cell_source(NB01, cell_index)))
        replace_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
        ]
        assert replace_calls, f"nb01 cell {cell_index}: no os.replace checkpoint write"


def test_every_code_cell_byte_compiles(tmp_path):
    """No fix may break cell syntax; magics are stripped as the runner sees them."""
    failures = []
    for filename in (NB01, NB02):
        for cell_index, source in load_code_cells(filename):
            target = tmp_path / f"{filename.replace('.ipynb', '')}_cell_{cell_index}.py"
            target.write_text(strip_ipython_magics(source))
            try:
                py_compile.compile(str(target), cfile=str(target) + "c", doraise=True)
            except py_compile.PyCompileError as error:
                failures.append(f"{filename} cell {cell_index}: {error}")
    assert not failures, "cells failed to byte-compile:\n" + "\n".join(failures)
