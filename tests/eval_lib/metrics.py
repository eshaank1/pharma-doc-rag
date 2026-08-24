from typing import Dict, List, Tuple


def check_answer_match(answer: str, expected_facts: List[str]) -> bool:
    """True if every expected fact appears, case-insensitively, in the answer."""
    answer_lower = answer.lower()
    return all(fact.lower() in answer_lower for fact in expected_facts)


def check_recall_hit(sources: List[Dict], expected_pages: Tuple[int, int]) -> bool:
    """True if any source's page range overlaps expected_pages."""
    exp_start, exp_end = expected_pages
    for source in sources:
        pages = source.get("pages", "")
        if "-" not in pages:
            continue
        start_str, end_str = pages.split("-", 1)
        try:
            start, end = int(start_str), int(end_str)
        except ValueError:
            continue
        if start <= exp_end and end >= exp_start:
            return True
    return False


def classify_bucket(pages_info: List, expected_pages: Tuple[int, int]) -> str:
    """'scanned' if any page in expected_pages went through OCR, else 'digital'."""
    exp_start, exp_end = expected_pages
    for page in pages_info:
        if exp_start <= page.page_num <= exp_end and page.was_ocr:
            return "scanned"
    return "digital"


def aggregate_metrics(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    rows: dicts with keys bucket ('digital'/'scanned'), correct (bool),
    recall_hit (bool), latency (float seconds), error (bool).
    Returns {'digital'|'scanned'|'overall': {'answer_match_pct', 'recall_pct',
    'avg_latency', 'error_rate_pct'}}.
    """
    buckets = {"digital": [], "scanned": [], "overall": []}
    for row in rows:
        buckets[row["bucket"]].append(row)
        buckets["overall"].append(row)

    result = {}
    for bucket, bucket_rows in buckets.items():
        total = len(bucket_rows)
        if total == 0:
            result[bucket] = {
                "answer_match_pct": 0.0,
                "recall_pct": 0.0,
                "avg_latency": 0.0,
                "error_rate_pct": 0.0,
            }
            continue

        errors = [r for r in bucket_rows if r["error"]]
        non_errors = [r for r in bucket_rows if not r["error"]]

        answer_match_pct = (
            100.0 * sum(1 for r in non_errors if r["correct"]) / len(non_errors)
            if non_errors else 0.0
        )
        recall_pct = (
            100.0 * sum(1 for r in non_errors if r["recall_hit"]) / len(non_errors)
            if non_errors else 0.0
        )
        avg_latency = (
            sum(r["latency"] for r in non_errors) / len(non_errors)
            if non_errors else 0.0
        )
        error_rate_pct = 100.0 * len(errors) / total

        result[bucket] = {
            "answer_match_pct": answer_match_pct,
            "recall_pct": recall_pct,
            "avg_latency": avg_latency,
            "error_rate_pct": error_rate_pct,
        }
    return result
