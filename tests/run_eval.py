"""
Runs every ground-truth test case in tests/ground_truth/ through the real
pharma_rag pipeline and reports Answer Match %, Retrieval Recall@k, average
latency, and error rate, broken out by digital vs. scanned documents.

Usage: python tests/run_eval.py
Requires Ollama running (same as the app itself): ollama serve
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TESTS_DIR))

from pharma_rag.document_store import EnhancedDocumentStore  # noqa: E402
from pharma_rag.llm import wait_for_ollama  # noqa: E402

from eval_lib.metrics import (  # noqa: E402
    aggregate_metrics,
    check_answer_match,
    check_recall_hit,
    classify_bucket,
)
from eval_lib.report import format_console_table, write_results_json  # noqa: E402

FIXTURES_DIR = TESTS_DIR / "fixtures"
GROUND_TRUTH_DIR = TESTS_DIR / "ground_truth"
RESULTS_DIR = TESTS_DIR / "results"
DEFAULT_K = 4

# Literal prefix of the fallback answer that generate_answer_with_sources()
# returns when the LLM call itself fails -- it doesn't raise, it just returns
# a normal-looking result dict with this text baked into "answer". Without
# checking for it explicitly, this infrastructure failure silently scores as
# a MISS (wrong answer) rather than an ERROR (pipeline didn't run).
LLM_FAILURE_PREFIX = "The AI answer-generation step failed"


def run_fixture(fixture_path: Path, ground_truth: dict) -> list:
    """Process one fixture and run all its test cases. Returns a list of
    result rows -- one error row per case if processing the PDF itself
    failed, otherwise one scored row per case."""
    doc_store = EnhancedDocumentStore()
    success, stats = doc_store.process_pdf(str(fixture_path), filename=fixture_path.name)

    # tests/generate_fixtures.py names scanned fixtures with a "scanned_"
    # prefix (scanned_certificate.pdf, scanned_bse_tse.pdf) -- use that
    # instead of hardcoding "digital" so a failure on a scanned fixture
    # doesn't get misreported under the digital bucket.
    bucket = "scanned" if "scanned" in fixture_path.stem else "digital"

    rows = []
    if not success:
        for case in ground_truth["cases"]:
            rows.append({
                "fixture": fixture_path.name,
                "question": case["question"],
                "bucket": bucket,
                "correct": False,
                "recall_hit": False,
                "doc_type_hit": False,
                "latency": 0.0,
                "error": True,
                "error_message": stats.get("error", "process_pdf failed"),
            })
        return rows

    for case in ground_truth["cases"]:
        expected_pages = tuple(case["expected_pages"])
        case_bucket = classify_bucket(doc_store.pages_info, expected_pages)

        start = time.perf_counter()
        try:
            result = doc_store.query(case["question"], auto_route=True, k=DEFAULT_K)
            elapsed = time.perf_counter() - start

            if result["answer"].startswith(LLM_FAILURE_PREFIX):
                rows.append({
                    "fixture": fixture_path.name,
                    "question": case["question"],
                    "bucket": case_bucket,
                    "correct": False,
                    "recall_hit": False,
                    "doc_type_hit": False,
                    "latency": elapsed,
                    "error": True,
                    "error_message": result["answer"][:200],
                })
                continue

            doc_type_hit = any(
                s.get("doc_type") == case["expected_doc_type"] for s in result["sources"]
            )
            rows.append({
                "fixture": fixture_path.name,
                "question": case["question"],
                "bucket": case_bucket,
                "correct": check_answer_match(result["answer"], case["expected_facts"]),
                "recall_hit": check_recall_hit(result["sources"], expected_pages),
                "doc_type_hit": doc_type_hit,
                "latency": elapsed,
                "error": False,
                "answer": result["answer"],
                "expected_facts": case["expected_facts"],
            })
        except Exception as e:
            elapsed = time.perf_counter() - start
            rows.append({
                "fixture": fixture_path.name,
                "question": case["question"],
                "bucket": case_bucket,
                "correct": False,
                "recall_hit": False,
                "doc_type_hit": False,
                "latency": elapsed,
                "error": True,
                "error_message": f"{type(e).__name__}: {e}",
            })

    return rows


def main():
    print("Checking Ollama is reachable...")
    if not wait_for_ollama(timeout=15):
        print("ERROR: Ollama isn't reachable at the configured OLLAMA_HOST.")
        print("Start it with: ollama serve")
        sys.exit(1)

    fixture_paths = sorted(FIXTURES_DIR.glob("*.pdf"))
    if not fixture_paths:
        print(f"No fixtures found in {FIXTURES_DIR}. Run tests/generate_fixtures.py first, "
              f"and make sure pharma_blob_sample.pdf has been copied in.")
        sys.exit(1)

    all_rows = []
    for fixture_path in fixture_paths:
        gt_path = GROUND_TRUTH_DIR / f"{fixture_path.stem}.json"
        if not gt_path.exists():
            print(f"Skipping {fixture_path.name}: no matching ground truth at {gt_path}")
            continue

        with open(gt_path) as f:
            ground_truth = json.load(f)

        print(f"Running {len(ground_truth['cases'])} case(s) for {fixture_path.name}...")
        all_rows.extend(run_fixture(fixture_path, ground_truth))

    if not all_rows:
        print("No test cases ran. Nothing to report.")
        sys.exit(1)

    aggregate = aggregate_metrics(all_rows)
    print()
    print(format_console_table(aggregate, k=DEFAULT_K))

    results_path = write_results_json(RESULTS_DIR, aggregate, all_rows)
    print()
    print(f"Saved full results to {results_path}")


if __name__ == "__main__":
    main()
