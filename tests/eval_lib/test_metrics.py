from eval_lib.metrics import (
    aggregate_metrics,
    check_answer_match,
    check_recall_hit,
    classify_bucket,
)


def test_check_answer_match_all_facts_present():
    assert check_answer_match("The lot number is 18356721.", ["18356721"]) is True


def test_check_answer_match_case_insensitive():
    assert check_answer_match("Sterilized via AUTOCLAVE at 121C.", ["Autoclave", "121"]) is True


def test_check_answer_match_missing_fact():
    assert check_answer_match("The lot number is 18356721.", ["18356721", "20260315"]) is False


def test_check_answer_match_empty_facts_list_is_trivially_true():
    assert check_answer_match("Anything at all.", []) is True


def test_check_recall_hit_overlapping_range():
    assert check_recall_hit([{"pages": "6-6"}], (6, 6)) is True


def test_check_recall_hit_no_overlap():
    assert check_recall_hit([{"pages": "1-1"}], (6, 6)) is False


def test_check_recall_hit_multi_page_overlap():
    assert check_recall_hit([{"pages": "3-4"}], (4, 4)) is True


def test_check_recall_hit_no_sources():
    assert check_recall_hit([], (6, 6)) is False


def test_check_recall_hit_ignores_malformed_pages_field():
    assert check_recall_hit([{"pages": "unknown"}, {"pages": "6-6"}], (6, 6)) is True


class _FakePage:
    def __init__(self, page_num, was_ocr):
        self.page_num = page_num
        self.was_ocr = was_ocr


def test_classify_bucket_digital():
    pages_info = [_FakePage(0, False), _FakePage(1, False)]
    assert classify_bucket(pages_info, (0, 0)) == "digital"


def test_classify_bucket_scanned():
    pages_info = [_FakePage(0, True), _FakePage(1, False)]
    assert classify_bucket(pages_info, (0, 0)) == "scanned"


def test_classify_bucket_scanned_if_any_page_in_range_was_ocr():
    pages_info = [_FakePage(3, False), _FakePage(4, True)]
    assert classify_bucket(pages_info, (3, 4)) == "scanned"


def test_aggregate_metrics_basic():
    rows = [
        {"bucket": "digital", "correct": True, "recall_hit": True, "latency": 2.0, "error": False},
        {"bucket": "digital", "correct": False, "recall_hit": True, "latency": 4.0, "error": False},
        {"bucket": "scanned", "correct": True, "recall_hit": False, "latency": 3.0, "error": False},
    ]
    result = aggregate_metrics(rows)
    assert result["digital"]["answer_match_pct"] == 50.0
    assert result["digital"]["recall_pct"] == 100.0
    assert result["digital"]["avg_latency"] == 3.0
    assert result["scanned"]["answer_match_pct"] == 100.0
    assert round(result["overall"]["answer_match_pct"], 2) == 66.67


def test_aggregate_metrics_excludes_errors_from_match_and_recall():
    rows = [
        {"bucket": "digital", "correct": True, "recall_hit": True, "latency": 2.0, "error": False},
        {"bucket": "digital", "correct": False, "recall_hit": False, "latency": 0.0, "error": True},
    ]
    result = aggregate_metrics(rows)
    assert result["digital"]["answer_match_pct"] == 100.0
    assert result["digital"]["error_rate_pct"] == 50.0


def test_aggregate_metrics_empty_bucket_reports_zeros_not_error():
    rows = [
        {"bucket": "digital", "correct": True, "recall_hit": True, "latency": 1.0, "error": False},
    ]
    result = aggregate_metrics(rows)
    assert result["scanned"]["answer_match_pct"] == 0.0
    assert result["scanned"]["error_rate_pct"] == 0.0
