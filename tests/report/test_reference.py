from src.report.reference import REPORT_VALUES, compare


def _row(table, label):
    return table[table["항목"] == label].iloc[0]


def test_values_agree_after_rounding_to_the_reports_precision():
    table = compare({"ladder_M3_net": 14.3149, "omega_squared": 0.266}, full_run=True)
    assert _row(table, "순수 감소 M3 (%p)")["결과"] == "일치"
    assert _row(table, "회피 성향의 투수 고유 몫 (ω²)")["결과"] == "일치"


def test_a_different_value_is_reported_as_a_difference_with_both_numbers():
    row = _row(compare({"ladder_M3_net": 14.2}, full_run=True), "순수 감소 M3 (%p)")
    assert row["결과"] == "차이" and row["보고서"] == "14.31" and row["재현"] == "14.20"


def test_a_step_that_did_not_run_is_marked_skipped_not_different():
    table = compare({}, full_run=True)
    assert len(table) == len(REPORT_VALUES) and (table["결과"] == "건너뜀").all()


def test_bootstrap_numbers_are_flagged_when_repetitions_were_cut():
    results = {"selection_ci_low": 0.0002, "ladder_M3_net": 14.2}
    quick, full = compare(results, full_run=False), compare(results, full_run=True)
    assert _row(quick, "보수적 95% CI 하한")["결과"] == "차이 (반복 수 축소)"
    assert _row(full, "보수적 95% CI 하한")["결과"] == "차이"
    assert _row(quick, "순수 감소 M3 (%p)")["결과"] == "차이"  # not a bootstrap number


def test_text_values_are_compared_as_written():
    top = REPORT_VALUES["cases_top"][2]
    table = compare({"cases_top": top, "cases_bottom": "Someone 1%"}, full_run=True)
    assert _row(table, "회피 상위 5명")["결과"] == "일치" and _row(table, "회피 하위 5명")["결과"] == "차이"


def test_markdown_table_has_a_header_a_rule_and_one_line_per_row():
    from src.report.reference import to_markdown

    text = to_markdown(compare({"omega_squared": 0.266}, full_run=True).tail(1))
    lines = text.splitlines()
    assert lines[0] == "| 쪽 | 항목 | 보고서 | 재현 | 결과 |"
    assert lines[1] == "|---|---|---|---|---|"
    assert lines[2] == "| 11 | 회피 성향의 투수 고유 몫 (ω²) | 0.27 | 0.27 | 일치 |"
