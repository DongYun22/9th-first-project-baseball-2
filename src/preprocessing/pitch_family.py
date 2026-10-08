"""Map Statcast pitch_type codes to a fastball/breaking/offspeed family.

Statcast's own pitch_type taxonomy has ~15 codes and grows over time
(e.g. "ST" and "SV" were added after 2022). Codes are grouped by the
standard fastball / breaking-ball / offspeed split (matching Baseball
Savant's own pitch-arsenal grouping); anything genuinely unrecognized
falls into "other" rather than being silently mis-bucketed or dropped.
"""

FASTBALL_CODES = {"FF", "SI", "FT", "FA"}
CUTTER_CODE = "FC"
BREAKING_CODES = {"SL", "ST", "CU", "KC", "CS", "SV", "SC"}
OFFSPEED_CODES = {"CH", "FS", "FO", "EP", "KN"}


def _is_missing(pitch_type: str | float | None) -> bool:
    return pitch_type is None or (isinstance(pitch_type, float) and pitch_type != pitch_type)


def is_cutter(pitch_type: str | float | None) -> bool:
    if _is_missing(pitch_type):
        return False
    return pitch_type.upper() == CUTTER_CODE


def map_pitch_family(pitch_type: str | float | None) -> str | None:
    if _is_missing(pitch_type):
        return None
    code = pitch_type.upper()
    if code in FASTBALL_CODES or code == CUTTER_CODE:
        return "fastball"
    if code in BREAKING_CODES:
        return "breaking"
    if code in OFFSPEED_CODES:
        return "offspeed"
    return "other"
