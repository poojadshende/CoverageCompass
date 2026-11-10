"""Reproduce retrieval and routing metrics for the curated evaluation set."""
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from engine import CoverageEngine

ROOT = Path(__file__).resolve().parent

def main():
    engine = CoverageEngine()
    cases = json.loads((ROOT / "data/evaluation.json").read_text(encoding="utf-8"))
    results = []
    for role, question, expected_route, expected in cases:
        start = perf_counter()
        answer = engine.ask(question, role)
        elapsed = (perf_counter() - start) * 1000
        topics = engine.classify({"question": question})["topics"]
        dense = [doc.id for doc in engine.rank(question, topics, hybrid=False)[:2]]
        hybrid = [doc.id for doc in engine.rank(question, topics)[:2]]
        returned = [card["id"] for card in answer["cards"]]
        results.append({
            "role": role, "question": question, "expected_route": expected_route,
            "actual_route": answer["route"], "expected_evidence": expected, "dense_at_2": dense,
            "hybrid_at_2": hybrid, "returned_evidence": returned, "elapsed_ms": round(elapsed, 2),
        })

    gold_total = sum(len(row["expected_evidence"]) for row in results)

    def recall(field):
        found = sum(len(set(row["expected_evidence"]) & set(row[field])) for row in results)
        return {"found": found, "total": gold_total, "fraction": round(found / gold_total, 3)}

    report = {
        "mode": engine.mode,
        "route_correct": sum(row["expected_route"] == row["actual_route"] for row in results),
        "questions": len(results),
        "dense_recall_at_2": recall("dense_at_2"),
        "hybrid_recall_at_2": recall("hybrid_at_2"),
        "returned_evidence_recall": recall("returned_evidence"),
        "median_ms": round(median(row["elapsed_ms"] for row in results), 2),
        "max_ms": max(row["elapsed_ms"] for row in results),
        "results": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))

if __name__ == "__main__":
    main()
