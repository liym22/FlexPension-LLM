"""Household-level pension enrollment counts."""

import pandas as pd


def count_other_enrolled_members(
    members,
    focal_people,
    household_col,
    person_col,
    enrolled_col,
):
    """Count enrolled household members after excluding each focal person."""
    required_member_cols = {household_col, person_col, enrolled_col}
    required_focal_cols = {household_col, person_col}
    missing_member_cols = required_member_cols.difference(members.columns)
    missing_focal_cols = required_focal_cols.difference(focal_people.columns)
    if missing_member_cols or missing_focal_cols:
        missing = sorted(missing_member_cols | missing_focal_cols)
        raise KeyError(f"Missing required columns: {missing}")

    household_groups = {
        household_id: group
        for household_id, group in members.groupby(
            household_col, sort=False, dropna=False
        )
    }
    counts = []
    for household_id, focal_person_id in focal_people[
        [household_col, person_col]
    ].itertuples(index=False, name=None):
        group = household_groups.get(household_id)
        if group is None:
            counts.append(0)
            continue

        other_members = group[person_col].ne(focal_person_id).fillna(True)
        enrolled = pd.to_numeric(
            group.loc[other_members, enrolled_col], errors="coerce"
        ).fillna(0)
        counts.append(int(enrolled.sum()))

    return pd.Series(counts, index=focal_people.index, dtype="int64")
