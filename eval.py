"""
Code Archaeology — Eval Harness

The piece that proves the agent's answers are actually grounded in real
history rather than plausible-sounding hallucination. You hand-curate a set
of questions where you've personally verified the true rationale (usually
by reading the linked issue/PR yourself), then this script:

    1. Runs each question through the agent
    2. Checks whether the agent cited the correct source (PR/issue/commit
       number you identified as ground truth) -- a cheap, objective check
    3. Uses Claude as a judge to score whether the agent's stated rationale
       actually matches your ground-truth rationale -- a semantic check
    4. Produces a report with per-case and aggregate scores

Why both checks: citation-matching alone can be gamed (an agent could cite
the right PR number but describe the wrong reason), and judge-scoring alone
can be fooled by confident, well-written hallucination. Together they're a
much harder bar to fake.

Requires:
    pip install anthropic

Env vars:
    ANTHROPIC_API_KEY, DATABASE_URL

Usage:
    python eval.py --repo owner/name --cases eval_cases.json
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from agent import run_agent

JUDGE_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Eval case format
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    question: str
    ground_truth_rationale: str          # what you, the human, determined the real reason was
    ground_truth_source: str             # e.g. "PR #4821" or "issue #3190" -- for the citation check
    notes: str = ""                      # optional context for why this case is interesting


@dataclass
class EvalResult:
    case: EvalCase
    agent_answer: str
    cited_correct_source: bool
    judge_score: int                     # 1-5
    judge_reasoning: str


def load_cases(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    return [EvalCase(**c) for c in raw]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_citation(agent_answer: str, ground_truth_source: str) -> bool:
    """
    Cheap objective check: does the agent's answer mention the source
    number we independently verified as correct? Normalizes '#4821',
    'PR #4821', 'PR 4821' etc. to the bare number for matching.
    """
    import re
    numbers_in_truth = re.findall(r"\d+", ground_truth_source)
    if not numbers_in_truth:
        return False
    target = numbers_in_truth[0]
    numbers_in_answer = set(re.findall(r"\d+", agent_answer))
    return target in numbers_in_answer


JUDGE_PROMPT_TEMPLATE = """You are scoring whether an AI agent's answer about a code change \
correctly identifies the real rationale behind it.

QUESTION ASKED:
{question}

GROUND TRUTH (verified by a human who read the actual issue/PR):
{ground_truth}

AGENT'S ANSWER:
{agent_answer}

Score the agent's answer from 1-5:
5 = Correctly identifies the core rationale, consistent with ground truth
4 = Mostly correct, minor omission or imprecision
3 = Partially correct -- gets the general area right but misses the specific reason
2 = Mostly wrong -- tangentially related but misses the actual rationale
1 = Wrong or fabricated -- contradicts ground truth or invents an unsupported reason

Respond with ONLY a JSON object, no other text: {{"score": <int>, "reasoning": "<one sentence>"}}"""


def judge_answer(client: anthropic.Anthropic, case: EvalCase, agent_answer: str) -> tuple:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=case.question,
        ground_truth=case.ground_truth_rationale,
        agent_answer=agent_answer,
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    # Defensive parsing in case the judge wraps the JSON in markdown fences
    text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        return int(parsed["score"]), parsed["reasoning"]
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0, f"Judge response unparseable: {text[:200]}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_eval(cases: list, repo: str, dsn: str, api_key: str) -> list:
    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.question}")

        agent_answer = run_agent(case.question, repo, dsn, api_key, verbose=False)
        cited_correct = check_citation(agent_answer, case.ground_truth_source)
        score, reasoning = judge_answer(client, case, agent_answer)

        result = EvalResult(
            case=case,
            agent_answer=agent_answer,
            cited_correct_source=cited_correct,
            judge_score=score,
            judge_reasoning=reasoning,
        )
        results.append(result)

        print(f"    citation match: {cited_correct}  |  judge score: {score}/5")

    return results


def print_report(results: list):
    print("\n" + "=" * 70)
    print("EVAL REPORT")
    print("=" * 70)

    avg_score = sum(r.judge_score for r in results) / len(results)
    citation_accuracy = sum(r.cited_correct_source for r in results) / len(results)

    print(f"\nCases run: {len(results)}")
    print(f"Average judge score: {avg_score:.2f} / 5")
    print(f"Citation accuracy: {citation_accuracy:.0%}")

    print("\n--- Per-case detail ---")
    for i, r in enumerate(results, 1):
        flag = "OK" if r.judge_score >= 4 and r.cited_correct_source else "REVIEW"
        print(f"\n[{i}] {flag}  score={r.judge_score}/5  citation_match={r.cited_correct_source}")
        print(f"    Q: {r.case.question}")
        print(f"    Judge: {r.judge_reasoning}")

    low_scoring = [r for r in results if r.judge_score < 4 or not r.cited_correct_source]
    if low_scoring:
        print(f"\n{len(low_scoring)} case(s) worth manually reviewing -- "
              "these are good material for a 'known limitations' section in your writeup.")


def save_report_json(results: list, path: str):
    out = [
        {
            "question": r.case.question,
            "ground_truth_rationale": r.case.ground_truth_rationale,
            "ground_truth_source": r.case.ground_truth_source,
            "agent_answer": r.agent_answer,
            "cited_correct_source": r.cited_correct_source,
            "judge_score": r.judge_score,
            "judge_reasoning": r.judge_reasoning,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run the Code Archaeology eval suite")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cases", required=True, help="path to eval_cases.json")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    dsn = os.environ["DATABASE_URL"]
    api_key = os.environ["ANTHROPIC_API_KEY"]

    cases = load_cases(args.cases)
    results = run_eval(cases, args.repo, dsn, api_key)
    print_report(results)
    save_report_json(results, args.out)


if __name__ == "__main__":
    main()
