# Learning AI at its Core

My learning journal and roadmap for AI, LangChain, LangGraph and agentic AI.
**Goal:** very strong fundamentals. I should understand every layer well enough to build it by hand, then use the framework because I *choose* to.

> For Claude: this file is the shared memory. Read it at the start of a session and update it at the end (see CLAUDE.md).

---

## Next up
**Phase 1 · Lesson 1: What an LLM call really is.** Make a raw API call with no framework. Cover tokens, the messages format, temperature, max tokens and cost per call.

---

## Roadmap

### Phase 1: LLM fundamentals (no frameworks)
- [ ] What an LLM is: next-token prediction, tokens, tokenizers, context window
- [ ] Raw API call: messages (system/user/assistant), temperature, top-p, max tokens, streaming
- [ ] Prompting basics: instructions, few-shot, delimiters, chain-of-thought, why prompts fail
- [ ] Structured output: JSON mode, schemas, Pydantic validation
- [ ] Cost and latency: counting tokens, prompt caching

### Phase 2: Embeddings & retrieval from scratch
- [ ] Embeddings: what a vector means, cosine similarity in numpy
- [ ] A tiny vector search by hand (list of vectors + top-k)
- [ ] Keyword search: TF-IDF and BM25 by hand
- [ ] Hybrid search and rank fusion (RRF) by hand
- [ ] Chunking: fixed size vs recursive vs semantic vs by heading, and the trade-offs
- [ ] MMR (diversity) vs plain similarity
- [ ] Rerankers (cross-encoders) and why they beat embedding similarity for ordering
- [ ] Vector DBs: Qdrant collections, dense + sparse vectors, metadata filters

### Phase 3: RAG, by hand then with LangChain
- [ ] Naive RAG in ~50 lines of plain Python
- [ ] Evaluating RAG: build a Q&A test set; retrieval recall, faithfulness, answer relevance
- [ ] Long context vs RAG: when you don't need RAG at all
- [ ] LangChain core: Runnables, LCEL (`|`), prompts, models, output parsers, retrievers
- [ ] Advanced retrieval: query rewriting, multi-query / RAG-Fusion, HyDE, step-back, contextual retrieval

### Phase 4: Tools & agents from scratch
- [ ] Tool / function calling: how the model "calls" a function
- [ ] Hand-written agent loop (ReAct): think → act → observe → repeat
- [ ] Stopping conditions, errors and guardrails in agent loops
- [ ] Memory: conversation history and summarisation

### Phase 5: LangGraph
- [ ] StateGraph: state, nodes, edges, START/END
- [ ] Reducers (`Annotated[list, add]`) and why parallel nodes need them
- [ ] Conditional edges and routing
- [ ] Parallel fan-out / fan-in and `Send` (map-reduce)
- [ ] Checkpointers, threads and memory
- [ ] Human-in-the-loop (interrupts)
- [ ] Subgraphs and multi-agent patterns
- [ ] LangGraph Studio and LangSmith tracing

### Phase 6: Agentic RAG patterns
- [ ] Routed RAG vs truly agentic RAG (the difference matters)
- [ ] Adaptive RAG, Corrective RAG (CRAG), Self-RAG
- [ ] Agent decides *whether* and *what* to retrieve, and retries

### Phase 7: Multimodal
- [ ] Vision-language models: image input, captioning, OCR vs VLM parsing
- [ ] Parsing messy PDFs (tables, scans, charts)

### Phase 8: Production fundamentals
- [ ] Evals as a habit (RAGAS / custom eval sets)
- [ ] Observability, cost/latency budgets, caching, retries
- [ ] Safety: prompt injection, grounding, refusal behaviour

### Capstone
- [ ] Rebuild this repo's system correctly, with an eval set proving each technique helps

---

## Progress log

### 2026-10-20: Session 1: Analysis of the existing repo
Analysed the original project (hash2004's "Multimodal Agentic RAG"): code, notebooks and the paper in `assets/report.pdf`.

**Key takeaways**
- The *techniques* are a good 2024–25 toolbox: VLM page parsing, contextual retrieval, hybrid dense+BM25 search, multi-query retrieval, reranking, routing with LangGraph.
- It is **routed RAG, not agentic RAG.** One LLM picks a route, then a fixed pipeline runs. Agentic means the LLM makes decisions in a loop (retrieve or not, is the context good enough, try again).
- **No evaluation**, so no claim can be verified. Lesson: *measure before adding complexity.*
- **Right-size the solution:** the financial document is only ~4k tokens and all three PDFs total ~70k tokens. They would fit in one long-context prompt. RAG pays off at a much larger scale.

---

## Concepts learned
| Concept | One-line meaning |
|---|---|
| RAG | Fetch relevant text from your data and put it in the prompt so the LLM answers from it |
| Chunking | Splitting documents into pieces small enough to embed and retrieve |
| Embedding | A vector that represents meaning; similar texts give nearby vectors |
| BM25 | Classic keyword-matching score; great for exact terms, names, numbers |
| Hybrid search | Combine dense (meaning) and sparse (keyword) search results |
| Reranker | A model that rescores (query, chunk) pairs more accurately than embeddings |
| MMR | Retrieval that balances relevance with diversity, to avoid near-duplicate chunks |
| Contextual retrieval | An LLM writes a short context for each chunk and *prepends* it before embedding (Anthropic, 2024) |
| RAG-Fusion | Several query rewrites → several searches → merge (RRF) / rerank |
| RRF | Reciprocal Rank Fusion: merge ranked lists by summing 1/(k + rank) |
| Router | An LLM classifier that sends a query to the right pipeline/collection |
| LangGraph reducer | A rule for combining a state field when several nodes write to it (e.g. `add` = append) |

---

## Fix-it labs (real bugs in this repo to practise on)
Each one teaches a concept. Do these once the matching phase is covered.

1. **Reranker output ignored**: `rerank_docs` re-searches instead of reranking the 12 results, and `chat_message` sends raw `retrieved_docs` to the LLM. → Phase 3/5 (data flow in graph state). `ai_backend/financial_chat_graph.py`
2. **Two collections, same data**: `contexulized_chunked_docs = chunked_docs` aliases one list, so the in-place edit changes both. → Python references. `data_ingestion/financial/financial_ingestion.ipynb`
3. **Chunker threshold off by 100×**: `breakpoint_threshold_amount=0.95` in percentile mode (expects 0–100). → Phase 2 chunking.
4. **Contextual retrieval done wrong**: the chunk is *replaced* by the LLM output instead of having context prepended. → Phase 3.
5. **Mislabelled search**: the "similarity" nodes for the contextualized store actually run MMR. → Phase 2.
6. **Score threshold on fused scores**: `score_threshold=0.7` on hybrid (rank-fused) scores. → Phase 2 RRF.
7. **Fragile router**: binds a tool but reads `.content`; an empty response breaks the graph. → Phase 1 structured output, Phase 5 conditional edges.
8. **Lossy ingestion**: non-ASCII characters stripped in a text→PDF→text round trip. → data hygiene.
9. **12 copy-pasted search functions**: should be one function plus `Send`. → Phase 5 fan-out.
10. **No memory**: every chat turn starts fresh. → Phase 4/5 memory and checkpointers.

---

## Open questions
- Which model/API keys will I use for lessons (Gemini, OpenAI, Claude, or a local model)?

## Resources
- Anthropic, *Introducing Contextual Retrieval* (2024)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)
- LangGraph docs and LangChain Academy (free LangGraph course)
