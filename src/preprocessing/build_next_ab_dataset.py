"""Build the event-level "did the pitcher reuse the same pitch type in the
batter's next plate appearance after allowing an extra-base hit" dataset.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.pitch_family import is_cutter, map_pitch_family

EXTRA_BASE_HIT_EVENTS = {"double", "triple", "home_run"}


def identify_events_by_outcome(pitches: pd.DataFrame, outcome_events: set[str]) -> pd.DataFrame:
    """Pitches whose `events` value is in `outcome_events` (e.g. XBH labels,
    or a placebo outcome like {"field_out"} for a control-group comparison).
    """
    return pitches[pitches["events"].isin(outcome_events)].copy()


def identify_extra_base_hit_events(pitches: pd.DataFrame) -> pd.DataFrame:
    return identify_events_by_outcome(pitches, EXTRA_BASE_HIT_EVENTS)


def find_next_at_bat_pitches(game_pitches: pd.DataFrame, next_at_bat_number: int) -> pd.DataFrame:
    """`game_pitches` must already be filtered to one (game_pk, pitcher)."""
    return game_pitches[game_pitches["at_bat_number"] == next_at_bat_number]


def compute_score_diff(hit_row: pd.Series) -> int:
    """Score differential from the pitching team's perspective at the moment of the hit."""
    if hit_row["inning_topbot"] == "Top":
        return int(hit_row["home_score"] - hit_row["away_score"])
    return int(hit_row["away_score"] - hit_row["home_score"])


def compute_baseline_usage(
    game_pitches: pd.DataFrame,
    before_at_bat_number: int,
    before_pitch_number: int,
    hit_pitch_type: str,
) -> float:
    """Share of `hit_pitch_type` among this pitcher's pitches in the same game,
    strictly before the extra-base-hit pitch (earlier at-bats, plus earlier
    pitches within the same at-bat). NaN if there were no prior pitches.
    """
    prior = game_pitches[
        (game_pitches["at_bat_number"] < before_at_bat_number)
        | (
            (game_pitches["at_bat_number"] == before_at_bat_number)
            & (game_pitches["pitch_number"] < before_pitch_number)
        )
    ]
    if len(prior) == 0:
        return float("nan")
    return float((prior["pitch_type"] == hit_pitch_type).mean())


NEXT_PA_MODES = ("next_batter", "same_batter")


def build_same_batter_next_pa_lookup(pitches: pd.DataFrame) -> dict[tuple[int, int, int], int]:
    """(game_pk, batter, at_bat_number) -> at_bat_number of that same batter's
    immediately following plate appearance in the same game (any pitcher).
    Plate appearances with no later PA by that batter are absent.
    """
    pa = (
        pitches[["game_pk", "batter", "at_bat_number"]]
        .drop_duplicates()
        .sort_values(["game_pk", "batter", "at_bat_number"])
    )
    pa["next_at_bat_number"] = pa.groupby(["game_pk", "batter"])["at_bat_number"].shift(-1)
    pa = pa.dropna(subset=["next_at_bat_number"])
    return {
        (int(game_pk), int(batter), int(at_bat)): int(next_at_bat)
        for game_pk, batter, at_bat, next_at_bat in zip(
            pa["game_pk"], pa["batter"], pa["at_bat_number"], pa["next_at_bat_number"]
        )
    }


def filter_to_same_batter_rematch(pitches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Rows of `events` whose batter's immediate next PA in the same game was
    (at least partly) thrown by the same pitcher -- i.e. events that have a
    same-batter rematch. Cheap vectorless pre-filter so a large candidate
    pool (e.g. every field_out) need not go through the per-event builder.
    """
    lookup = build_same_batter_next_pa_lookup(pitches)
    pitcher_pas = pitches[["game_pk", "at_bat_number", "pitcher"]].drop_duplicates()
    pitcher_pa_keys = set(
        zip(
            pitcher_pas["game_pk"].astype("int64"),
            pitcher_pas["at_bat_number"].astype("int64"),
            pitcher_pas["pitcher"].astype("int64"),
        )
    )
    keep = []
    for game_pk, batter, at_bat, pitcher in zip(
        events["game_pk"], events["batter"], events["at_bat_number"], events["pitcher"]
    ):
        next_at_bat = lookup.get((int(game_pk), int(batter), int(at_bat)))
        keep.append(next_at_bat is not None and (int(game_pk), next_at_bat, int(pitcher)) in pitcher_pa_keys)
    return events[pd.Series(keep, index=events.index)]


def compute_platoon_match(stand: str | float | None, p_throws: str | float | None) -> float:
    """1 if the batter and pitcher are same-handed (R vs R, L vs L), 0 if
    opposite-handed, NaN if either is missing.
    """
    if pd.isna(stand) or pd.isna(p_throws):
        return float("nan")
    return int(stand == p_throws)


def compute_catcher_changed(event_fielder_2: float, next_first_pitch_fielder_2: float) -> float:
    """1 if the catcher (fielder_2) differs between the event pitch and the
    first pitch of the next at-bat, 0 if the same, NaN if either is missing.
    """
    if pd.isna(event_fielder_2) or pd.isna(next_first_pitch_fielder_2):
        return float("nan")
    return int(event_fielder_2 != next_first_pitch_fielder_2)


def build_event_dataset_for_events(
    pitches: pd.DataFrame, target_events: pd.DataFrame, next_pa_mode: str = "next_batter"
) -> pd.DataFrame:
    """Compute the next-at-bat pitch-reuse feature set for an arbitrary set
    of outcome pitches (`target_events`, a row subset of `pitches`).

    This is the shared core used for both the real extra-base-hit dataset
    and a placebo/control group (e.g. `field_out` events) built from
    `identify_events_by_outcome`, so the two groups are guaranteed to be
    computed with identical logic.

    `next_pa_mode` picks which "next at-bat" is compared against:
    - "next_batter" (default): the very next plate appearance in the game
      (at_bat_number + 1), if the same pitcher threw it -- always a
      different batter.
    - "same_batter": the same batter's immediately following plate
      appearance in the game (a rematch), if the same pitcher threw it.
    """
    if next_pa_mode not in NEXT_PA_MODES:
        raise ValueError(f"next_pa_mode must be one of {NEXT_PA_MODES}, got {next_pa_mode!r}")

    grouped = {key: group for key, group in pitches.groupby(["game_pk", "pitcher"])}
    same_batter_lookup = build_same_batter_next_pa_lookup(pitches) if next_pa_mode == "same_batter" else None
    has_catcher_data = "fielder_2" in pitches.columns

    records = []
    for _, event_row in target_events.iterrows():
        game_pitches = grouped[(event_row["game_pk"], event_row["pitcher"])]

        if same_batter_lookup is None:
            next_at_bat_number = event_row["at_bat_number"] + 1
        else:
            next_at_bat_number = same_batter_lookup.get(
                (int(event_row["game_pk"]), int(event_row["batter"]), int(event_row["at_bat_number"]))
            )
        if next_at_bat_number is None:
            next_ab = game_pitches.iloc[0:0]
        else:
            next_ab = find_next_at_bat_pitches(game_pitches, next_at_bat_number)
        has_next_ab = len(next_ab) > 0
        next_ab_pitch_count = len(next_ab)
        # An event pitch with no pitch_type has no "same type" to compare, so
        # its reuse metrics stay NaN rather than reading as "not reused"/0%.
        has_pitch_type = pd.notna(event_row["pitch_type"])
        if has_next_ab and has_pitch_type:
            reused_same_type = int((next_ab["pitch_type"] == event_row["pitch_type"]).any())
            same_type_share = float((next_ab["pitch_type"] == event_row["pitch_type"]).mean())
        else:
            reused_same_type = float("nan")
            same_type_share = float("nan")

        if has_pitch_type:
            baseline_usage = compute_baseline_usage(
                game_pitches, event_row["at_bat_number"], event_row["pitch_number"], event_row["pitch_type"]
            )
        else:
            baseline_usage = float("nan")

        record = {
            "game_pk": event_row["game_pk"],
            "game_date": event_row["game_date"],
            "season": pd.to_datetime(event_row["game_date"]).year,
            "pitcher": event_row["pitcher"],
            "pitcher_name": event_row["player_name"],
            "batter": event_row["batter"],
            "stand": event_row["stand"],
            "p_throws": event_row["p_throws"],
            "platoon_match": compute_platoon_match(event_row["stand"], event_row["p_throws"]),
            "at_bat_number": event_row["at_bat_number"],
            "event_pitch_number": event_row["pitch_number"],
            "next_at_bat_number": next_at_bat_number if has_next_ab else float("nan"),
            "events": event_row["events"],
            "hit_pitch_type": event_row["pitch_type"],
            "pitch_family": map_pitch_family(event_row["pitch_type"]),
            "is_cutter": is_cutter(event_row["pitch_type"]),
            "balls": event_row["balls"],
            "strikes": event_row["strikes"],
            "outs_when_up": event_row["outs_when_up"],
            "inning": event_row["inning"],
            "inning_topbot": event_row["inning_topbot"],
            "score_diff": compute_score_diff(event_row),
            "baseline_usage": baseline_usage,
            "has_next_ab": has_next_ab,
            "next_ab_pitch_count": next_ab_pitch_count,
            "reused_same_type": reused_same_type,
            "same_type_share": same_type_share,
        }

        if has_catcher_data:
            if has_next_ab:
                next_first_pitch = next_ab.loc[next_ab["pitch_number"].idxmin()]
                record["catcher_changed"] = compute_catcher_changed(
                    event_row["fielder_2"], next_first_pitch["fielder_2"]
                )
            else:
                record["catcher_changed"] = float("nan")

        records.append(record)

    return pd.DataFrame.from_records(records)


def build_event_dataset(pitches: pd.DataFrame) -> pd.DataFrame:
    return build_event_dataset_for_events(pitches, identify_extra_base_hit_events(pitches))
