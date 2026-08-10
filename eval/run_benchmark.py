"""
Benchmark runner.

Purpose:
    Evaluate the hybrid Graph + Vector RAG system against a vector-only
    baseline over eval/questions.json, and report accuracy, accuracy by
    hop count, latency, cost/query, and one-time ingestion cost.

TODO (Phase 5):
    - Load eval/questions.json.
    - Run each question through both the vector-only baseline and the
      full hybrid system.
    - Score answers (accuracy) and record latency/cost per query.
    - Break accuracy down by hop count / question category.
    - Produce a results table suitable for the top of README.md.
"""
