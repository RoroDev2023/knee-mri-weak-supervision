"""B1/B2: restart and prerequisite guards raise actionable RuntimeErrors.

Each test execs the real cell in a namespace missing its prerequisites.
Reverting a guard turns the clear RuntimeError back into a raw
NameError/KeyError, which fails the test.
"""

import pandas as pd
import pytest

from extract import NB01, cell_source, exec_source


def test_cell6_restart_guard_names_cell_0():
    # Fresh kernel: nothing is defined, so the DATA_DIR guard fires first.
    with pytest.raises(RuntimeError, match="cell 0"):
        exec_source(cell_source(NB01, 6), {})


def test_cell6_restart_guard_names_cell_2_for_labels():
    # train exists (reload skipped) but LABELS is gone; the LABELS guard
    # must fire before the report_analysis rebuild uses it.
    namespace = {"train": pd.DataFrame({"StudyInstanceUID": ["study-0"]})}
    with pytest.raises(RuntimeError, match="cell 2"):
        exec_source(cell_source(NB01, 6), namespace)


def test_cell7_prerequisite_guard_names_cell_6():
    # Partial re-run: report_analysis exists but cell 6's Language
    # columns and language_summary do not.
    namespace = {
        "report_analysis": pd.DataFrame({"StudyInstanceUID": []}),
    }
    with pytest.raises(RuntimeError, match="cell 6"):
        exec_source(cell_source(NB01, 7), namespace)
