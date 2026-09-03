from pathlib import Path
import sys

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from common.household_enrollment import count_other_enrolled_members


def test_excludes_an_insured_focal_person():
    members = pd.DataFrame(
        {
            "household": ["h1", "h1", "h1"],
            "person": ["p1", "p2", "p3"],
            "insured": [1, 1, 0],
        }
    )
    focals = pd.DataFrame({"household": ["h1"], "person": ["p1"]})

    counts = count_other_enrolled_members(
        members, focals, "household", "person", "insured"
    )

    assert counts.tolist() == [1]


def test_does_not_subtract_when_focal_person_is_uninsured():
    members = pd.DataFrame(
        {
            "household": ["h1", "h1", "h1"],
            "person": ["p1", "p2", "p3"],
            "insured": [0, 1, 1],
        }
    )
    focals = pd.DataFrame({"household": ["h1"], "person": ["p1"]})

    counts = count_other_enrolled_members(
        members, focals, "household", "person", "insured"
    )

    assert counts.tolist() == [2]


def test_counts_each_focal_person_separately_within_one_household():
    members = pd.DataFrame(
        {
            "household": ["h1", "h1", "h1"],
            "person": ["p1", "p2", "p3"],
            "insured": [1, 0, 1],
        }
    )
    focals = pd.DataFrame(
        {"household": ["h1", "h1"], "person": ["p1", "p2"]},
        index=[10, 20],
    )

    counts = count_other_enrolled_members(
        members, focals, "household", "person", "insured"
    )

    assert counts.to_dict() == {10: 1, 20: 2}

