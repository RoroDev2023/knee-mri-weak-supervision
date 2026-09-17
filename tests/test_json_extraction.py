"""B6: the response parser extracts the first balanced JSON object.

The regression cases mirror the greedy-regex failure mode the fix
replaced: ``re.search(r"\\{.*\\}", ...)`` matched from the first ``{`` to
the last ``}``, so brace-bearing prose and later objects broke parsing
with an uncaught JSONDecodeError.
"""

from unittest import mock

import pandas as pd
import pytest

from extract import NB01, cell_source, exec_cell_head, labels_from_nb01

LABELS = labels_from_nb01()


@pytest.fixture(scope="module")
def parser_namespace():
    """nb01 cell 9 exec'd up to (excluding) its GPU smoke-test tail."""
    namespace = {
        "model": mock.MagicMock(name="model"),
        "tokenizer": mock.MagicMock(name="tokenizer"),
        "LABELS": LABELS,
        "report_analysis": pd.DataFrame(
            {
                "StudyInstanceUID": ["study-0"],
                "Report": ["normal study"],
                "Language": ["en"],
                "LanguageName": ["English"],
                **{label: [0] for label in LABELS},
            }
        ),
    }
    exec_cell_head(cell_source(NB01, 9), namespace, stop_before={"test_row"})
    return namespace


def test_clean_json_object_parses(parser_namespace):
    parsed = parser_namespace["extract_json_object"]('{"ACL": 1, "MCL": 0}')
    assert parsed == {"ACL": 1, "MCL": 0}


def test_parse_model_response_maps_all_labels(parser_namespace):
    predictions = parser_namespace["parse_model_response"](
        '{"ACL": 1, "MCL": 0, "Effusion": null, "Contusion": 5, "Fracture": "tear"}'
    )
    assert predictions["ACL"] == 1
    assert predictions["MCL"] == 0
    assert predictions["Effusion"] is None
    assert predictions["Contusion"] is None
    assert predictions["Fracture"] is None
    assert set(predictions) == set(LABELS)


def test_json_inside_markdown_fences_parses(parser_namespace):
    raw = "```json\n{\"ACL\": 1, \"MCL\": 0}\n```"
    assert parser_namespace["extract_json_object"](raw) == {"ACL": 1, "MCL": 0}


def test_json_followed_by_prose_with_braces_parses(parser_namespace):
    # The B6 regression case: the old greedy regex matched from the first
    # "{" through the last "}" (swallowing "{review} {today}") and the
    # invalid match crashed parsing.
    raw = 'The findings are {"ACL": 1, "MCL": 0} and {review} noted {today}.'
    assert parser_namespace["extract_json_object"](raw) == {"ACL": 1, "MCL": 0}


def test_two_json_objects_returns_first(parser_namespace):
    raw = '{"ACL": 0} {"ACL": 1}'
    assert parser_namespace["extract_json_object"](raw)["ACL"] == 0


def test_braces_inside_string_values_do_not_truncate(parser_namespace):
    # The brace-depth scan is string-aware: a "}" inside a JSON string
    # value must not close the object early.
    raw = '{"MCL": 0, "ACL": "note } inside"} trailing { prose'
    parsed = parser_namespace["extract_json_object"](raw)
    assert parsed == {"MCL": 0, "ACL": "note } inside"}


def test_response_without_braces_raises(parser_namespace):
    with pytest.raises(ValueError, match="No JSON object found"):
        parser_namespace["extract_json_object"]("I could not produce JSON.")


def test_unbalanced_braces_raise(parser_namespace):
    # The old code raised json.JSONDecodeError here with an unrelated
    # message; the fix raises the documented fallback error instead.
    with pytest.raises(ValueError, match="No balanced JSON object found"):
        parser_namespace["extract_json_object"]('{"ACL": 1')
