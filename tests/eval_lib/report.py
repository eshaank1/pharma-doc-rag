import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _metric_labels(k: int):
    return [
        ("answer_match_pct", "Answer Match %", "{:.0f}%"),
        ("recall_pct", f"Retrieval Recall@{k}", "{:.0f}%"),
        ("avg_latency", "Avg Latency (s)", "{:.1f}"),
        ("error_rate_pct", "Error Rate", "{:.0f}%"),
    ]


def format_console_table(aggregate: Dict[str, Dict[str, float]], k: int) -> str:
    """Render the Digital/Scanned/Overall metrics table as plain text."""
    col_width = 22
    header = f"{'Metric':<{col_width}}| {'Digital':<8}| {'Scanned':<8}| Overall"
    separator = "-" * col_width + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 8
    lines = [header, separator]
    for key, label, fmt in _metric_labels(k):
        digital = fmt.format(aggregate["digital"][key])
        scanned = fmt.format(aggregate["scanned"][key])
        overall = fmt.format(aggregate["overall"][key])
        lines.append(f"{label:<{col_width}}| {digital:<8}| {scanned:<8}| {overall}")
    return "\n".join(lines)


def write_results_json(results_dir: Path, aggregate: Dict[str, Dict[str, float]],
                        rows: List[Dict]) -> Path:
    """Write full run detail to a timestamped JSON file, return its path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = results_dir / f"{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "aggregate": aggregate,
        "cases": rows,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
