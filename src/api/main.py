"""
API entrypoint.

Purpose:
    Expose the hybrid Graph RAG system (router -> graph/vector retrieval ->
    merge -> answer) over an HTTP interface.

TODO (Phase 4+):
    - Choose a web framework (not yet decided).
    - Expose an endpoint that accepts a question and returns an answer
      with citations.
"""
