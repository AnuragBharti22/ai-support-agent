# AI Support Agent — RAG + Agentic AI

A support assistant that answers questions using retrieval-augmented generation (RAG) over a knowledge base, and autonomously decides when to call external tools instead of just answering from text — including logging questions it can't answer for human follow-up.

## What it does

- Retrieves relevant context from a vector database before answering (RAG)
- Strictly grounds answers in retrieved context — refuses to answer from general knowledge if the answer isn't in the documents
- Autonomously decides whether to call a tool based on the question:
  - `get_current_datetime` — for live date/time queries
  - `log_unanswered_question` — logs unanswerable questions to a file for follow-up instead of hallucinating an answer
- Exposes everything through a FastAPI web API
- Includes an automated evaluation suite using LLM-as-judge scoring

## Architecture

## Tech stack

- **LLM**: Groq API (`openai/gpt-oss-120b`)
- **Vector DB**: ChromaDB (local, persistent)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **API**: FastAPI + Uvicorn
- **Evaluation**: custom LLM-as-judge scoring script

## How to run it

```bash
# Clone and set up
git clone https://github.com/AnuragBharti22/ai-support-agent.git
cd ai-support-agent
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# Add your own Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Build the vector index from docs.txt
python build_index.py

# Run the agent directly in the terminal
python agent.py

# Or run it as a web API
uvicorn main:app --reload
# then visit http://127.0.0.1:8000/docs
```

## Evaluation results

Ran an automated 5-question test suite using an independent LLM-as-judge to score whether each answer was faithful to the retrieved context (or correctly declined to answer when it shouldn't).

**Score: 5/5 (100%)**

Test cases covered: standard document Q&A, tool-triggering questions, and out-of-scope questions that should be correctly declined and logged.

## Known limitations

- Grounding is generally strong but not 100% consistent — the model occasionally adds general knowledge beyond the retrieved context (e.g., mentioning key-rotation practices not present in the source document) despite explicit "use only context" instructions. This is a known limitation of smaller/faster models and could be improved with a stricter system prompt or a larger judge model for validation.
- Chunking is currently paragraph-based (split on blank lines) — works well for structured docs but would need a smarter strategy (recursive or semantic chunking) for messier real-world documents.
- Single-document knowledge base for this demo — architecture supports scaling to many documents without changes.

## What I'd improve with more time

- Hybrid search (keyword + semantic) for more robust retrieval
- Reranking retrieved chunks before passing to the LLM
- Streaming responses for better UX
- A lightweight frontend instead of relying on `/docs`