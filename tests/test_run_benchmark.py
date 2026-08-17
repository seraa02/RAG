"""
Tests for eval/run_benchmark.py's grading logic and aggregation.

What these tests prove:
    - An out-of-scope item is graded correct only when the system
      produced zero claims (declined) -- a fabricated claim on an
      out-of-scope question is a miss, even if the claim text happens to
      look plausible.
    - gold_entities is a list of SYNONYM GROUPS: correct requires at
      least one string from EVERY group, not every string in a flat
      list. A group with no matching synonym in the answer is a miss,
      even if other groups matched.
    - SystemResults.accuracy() computes correctly overall and filtered by
      difficulty, and returns None (not a ZeroDivisionError) for an empty
      or non-matching set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_benchmark import QueryOutcome, SystemResults, _grade


def test_out_of_scope_correct_only_when_no_claims():
    item = {"id": "q014", "difficulty": "out_of_scope", "gold_entities": []}
    assert _grade(item, claims=[], answer_text="I can't answer that.") is True
    assert _grade(item, claims=["fabricated"], answer_text="It's sunny.") is False


def test_normal_item_requires_all_gold_entity_groups_present():
    item = {"id": "q001", "difficulty": "single_hop", "gold_entities": [["AMD"], ["Intel"]]}
    assert _grade(item, claims=["x"], answer_text="AMD competes with Intel and NVIDIA.") is True
    assert _grade(item, claims=["x"], answer_text="AMD competes with NVIDIA.") is False  # missing Intel


def test_normal_item_accepts_any_synonym_within_a_group():
    item = {
        "id": "q001",
        "difficulty": "single_hop",
        "gold_entities": [["AMD", "Advanced Micro Devices"], ["NVIDIA", "Nvidia"]],
    }
    # Uses the full legal name, not the abbreviation -- still correct.
    assert _grade(item, claims=["x"], answer_text="Advanced Micro Devices competes with Nvidia.") is True
    # Uses neither synonym for the second group -- a miss.
    assert _grade(item, claims=["x"], answer_text="Advanced Micro Devices competes with a rival.") is False


def test_normal_item_with_no_gold_entities_requires_a_grounded_claim():
    item = {"id": "q099", "difficulty": "single_hop", "gold_entities": []}
    assert _grade(item, claims=["x"], answer_text="anything") is True
    assert _grade(item, claims=[], answer_text="No evidence found.") is False


def _outcome(qid: str, correct: bool, latency: float = 100.0) -> QueryOutcome:
    return QueryOutcome(
        question_id=qid,
        correct=correct,
        route="GRAPH",
        latency_ms=latency,
        answer_text="",
        graph_evidence_count=1,
        vector_evidence_count=0,
    )


def test_system_results_accuracy_overall():
    import eval.run_benchmark as rb

    rb._golden_by_id = {"q1": {"difficulty": "single_hop"}, "q2": {"difficulty": "single_hop"}}
    results = SystemResults(outcomes=[_outcome("q1", True), _outcome("q2", False)])
    assert results.accuracy() == 0.5


def test_system_results_accuracy_by_difficulty():
    import eval.run_benchmark as rb

    rb._golden_by_id = {
        "q1": {"difficulty": "single_hop"},
        "q2": {"difficulty": "aggregation"},
    }
    results = SystemResults(outcomes=[_outcome("q1", True), _outcome("q2", False)])
    assert results.accuracy("single_hop") == 1.0
    assert results.accuracy("aggregation") == 0.0


def test_system_results_accuracy_returns_none_for_empty():
    results = SystemResults(outcomes=[])
    assert results.accuracy() is None
