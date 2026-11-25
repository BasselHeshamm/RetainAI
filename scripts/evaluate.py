"""Run the 20-question benchmark against the live agent graph and report accuracy.

Metrics:
  routing   - every expected agent was called (or a listed alternative route)
  answer    - the answer contains every expected fact. Numeric facts are computed from the
              database at run time ({"sql": ..., "round": n}), so they always match the data.
Results are merged into eval/results.json by question id, so you can re-run a subset.

Usage:
  python scripts/evaluate.py                    # all 20
  python scripts/evaluate.py --ids 4 7 10-20    # a subset
  python scripts/evaluate.py --failed           # only questions that failed last time
  python scripts/evaluate.py --pause 5          # seconds between questions (free-tier limits)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainai import db  # noqa: E402
from retainai.agents import ask, build_graph  # noqa: E402

RESULTS = ROOT / "eval" / "results.json"
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def normalise(text: str) -> str:
    """Lower-case, unify the hyphen characters some models emit, drop thousands separators."""
    return text.translate(_DASHES).lower().replace(",", "")


def resolve(fact) -> str:
    """A fact is a string, or {"sql": ..., "round": n} evaluated against the database."""
    if isinstance(fact, dict):
        value = db.run_select(fact["sql"])["rows"][0][0]
        n = fact.get("round", 1)
        return str(int(round(value))) if n == 0 else f"{value:.{n}f}"
    return fact


def contains_fact(answer: str, fact: str) -> bool:
    """Match as text, or as a number equal after rounding to the fact's precision
    (an expected 11.3 matches an answer of 11.27%)."""
    a, f = normalise(answer), normalise(fact)
    if f in a:
        return True
    if re.fullmatch(r"\d+(\.\d+)?", f):
        decimals = len(f.split(".")[1]) if "." in f else 0
        return any(round(float(n), decimals) == float(f) for n in re.findall(r"\d+(?:\.\d+)?", a))
    return False


def routed_ok(q: dict, used: list[str]) -> bool:
    routes = [q["expected_agents"]] + q.get("alternative_agents", [])
    return any(set(r).issubset(used) for r in routes)


def parse_ids(tokens: list[str]) -> set[int]:
    ids: set[int] = set()
    for t in tokens:
        a, _, b = t.partition("-")
        ids.update(range(int(a), int(b or a) + 1))
    return ids


def is_daily_limit(err: str) -> bool:
    return "429" in err and re.search(r"per day|TPD|RPD|daily", err, re.IGNORECASE) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None, help="e.g. 4 7 10-20")
    ap.add_argument("--failed", action="store_true", help="re-run questions that failed last time")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds to wait between questions")
    args = ap.parse_args()

    questions = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if l.strip()]
    previous = {r["id"]: r for r in json.loads(RESULTS.read_text())["results"]} if RESULTS.exists() else {}
    if args.ids:
        wanted = parse_ids(args.ids)
        questions = [q for q in questions if q["id"] in wanted]
    elif args.failed:
        questions = [q for q in questions
                     if q["id"] not in previous or not (previous[q["id"]]["routing_ok"] and previous[q["id"]]["answer_ok"])]
    print(f"Running {len(questions)} question(s)")

    graph = build_graph()
    for i, q in enumerate(questions):
        if i:
            time.sleep(args.pause)
        facts = [resolve(f) for f in q["must_contain"]]
        t0 = time.time()
        try:
            out, error = ask(q["question"], graph=graph), None
        except Exception as e:  # keep going; count as a failure
            out, error = {"answer": "", "agents_used": [], "tool_calls": []}, str(e)
        dt = time.time() - t0
        routed = routed_ok(q, out["agents_used"])
        answered = error is None and all(contains_fact(out["answer"], f) for f in facts)
        previous[q["id"]] = {**q, "expected_facts": facts, "agents_used": out["agents_used"],
                             "n_tool_calls": len(out["tool_calls"]), "answer": out["answer"],
                             "routing_ok": routed, "answer_ok": answered, "seconds": round(dt, 1), "error": error}
        line = (f"[{q['id']:>2}] routing={'ok' if routed else 'FAIL'} answer={'ok' if answered else 'FAIL'} "
                f"{dt:5.1f}s  agents={out['agents_used']}")
        if error:
            line += f"\n     ERROR: {error[:300]}"
        elif not answered:
            line += f"\n     expected {facts} in: {out['answer'][:300]!r}"
        print(line, flush=True)
        if error and is_daily_limit(error):
            print("\nDaily rate limit reached: stopping. Re-run later with --failed to finish.")
            break

    results = [previous[k] for k in sorted(previous)]
    n = len(results)
    summary = {
        "questions_run": n,
        "routing_accuracy": round(sum(r["routing_ok"] for r in results) / n, 3),
        "answer_accuracy": round(sum(r["answer_ok"] for r in results) / n, 3),
        "avg_tool_calls": round(sum(r["n_tool_calls"] for r in results) / n, 2),
        "avg_seconds": round(sum(r["seconds"] for r in results) / n, 1),
    }
    print(json.dumps(summary, indent=2))
    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
