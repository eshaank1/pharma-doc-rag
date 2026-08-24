import json
import tempfile
from pathlib import Path

from eval_lib.report import format_console_table, write_results_json


def _sample_aggregate():
    return {
        "digital": {"answer_match_pct": 100.0, "recall_pct": 100.0, "avg_latency": 2.1, "error_rate_pct": 0.0},
        "scanned": {"answer_match_pct": 75.0, "recall_pct": 88.0, "avg_latency": 3.4, "error_rate_pct": 0.0},
        "overall": {"answer_match_pct": 92.0, "recall_pct": 96.0, "avg_latency": 2.6, "error_rate_pct": 0.0},
    }


def test_format_console_table_contains_all_metric_labels():
    table = format_console_table(_sample_aggregate(), k=4)
    assert "Answer Match %" in table
    assert "Retrieval Recall@4" in table
    assert "Avg Latency (s)" in table
    assert "Error Rate" in table


def test_format_console_table_contains_formatted_values():
    table = format_console_table(_sample_aggregate(), k=4)
    assert "100%" in table
    assert "75%" in table
    assert "2.1" in table


def test_write_results_json_creates_file_with_expected_structure():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        rows = [{"question": "What is the lot number?", "correct": True}]
        path = write_results_json(results_dir, _sample_aggregate(), rows)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["aggregate"]["digital"]["answer_match_pct"] == 100.0
        assert data["cases"] == rows
        assert "timestamp" in data


def test_write_results_json_creates_missing_results_dir():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "nested" / "results"
        assert not results_dir.exists()
        write_results_json(results_dir, _sample_aggregate(), [])
        assert results_dir.exists()
