# CLAUDE.md

This repo is Tehman's **learning workspace for AI fundamentals**: LLMs, RAG, LangChain, LangGraph and agentic AI.
The goal is very strong fundamentals, not just a working demo.

## Read first, every session
1. Read [LEARNING.md](LEARNING.md). It holds the roadmap, current position, progress log, concepts learned and open questions.
2. Continue from the "Next up" section unless Tehman asks for something else.

## How to teach here
- **Fundamentals before frameworks.** Build each idea by hand first (raw API call, numpy cosine similarity, a hand-written agent loop), then show what LangChain/LangGraph does for you.
- **Explain the "why"**: what problem a technique solves, what it costs, and when *not* to use it.
- **Small, runnable steps.** One concept per lesson, with code Tehman can run and change.
- **Check understanding.** End each lesson with 2–3 questions or a small exercise.
- Use the existing code in `ai_backend/` and `data_ingestion/` as a **case study**. It was written by another author (hash2004) and contains real bugs that make good exercises (listed in LEARNING.md).

## Keeping memory
- At the end of each session, update LEARNING.md: tick the roadmap, add a dated entry to the progress log, record new concepts and open questions, and set "Next up".
- New lessons and exercises go in `learning/NN-topic-name/` (e.g. `learning/01-llm-basics/`).

## Repo layout
- `ai_backend/`: the original LangGraph app (router + 3 RAG sub-graphs)
- `data_ingestion/`: notebooks for OCR / VLM parsing, chunking, embedding into Qdrant
- `data/`: source PDFs · `output/`: ingestion outputs · `assets/`: diagrams + the original author's report
- `learning/`: Tehman's lessons and exercises (created as we go)
