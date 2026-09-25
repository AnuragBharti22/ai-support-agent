# AI Support Agent — RAG + Agentic AI

A support assistant that answers questions using retrieval-augmented generation (RAG) over a real-world knowledge base, and autonomously decides when to call external tools instead of just answering from text — including logging questions it can't answer for human follow-up.

## What it does

- Retrieves relevant context from a vector database before answering (RAG)
- Strictly grounds answers in retrieved context — refuses to answer from general knowledge if the answer isn't in the documents
- Autonomously decides whether to call a tool based on the question:
  - `get_current_datetime` — for live date/time queries
  - `log_unanswered_question` — logs unanswerable questions to a file for follow-up instead of hallucinating an answer
- Exposes everything through a FastAPI web API and a Streamlit chat UI
- Includes an automated evaluation suite using LLM-as-judge scoring

## Architecture

User question
&nbsp;&nbsp;&nbsp;&nbsp;↓
Retrieve top-8 relevant chunks (ChromaDB + sentence-transformers embeddings)
&nbsp;&nbsp;&nbsp;&nbsp;↓
Build prompt: context + question + tool instructions
&nbsp;&nbsp;&nbsp;&nbsp;↓
LLM (Groq / openai-oss-120b) decides: answer directly, or call a tool?
&nbsp;&nbsp;&nbsp;&nbsp;↓
If tool needed → run real Python function → feed result back to LLM → final answer
&nbsp;&nbsp;&nbsp;&nbsp;↓
Return answer via FastAPI endpoint or Streamlit chat UI

## Tech stack

- **LLM**: Groq API (`openai/gpt-oss-120b`)
- **Vector DB**: ChromaDB (local, persistent)
- **Embeddings**: sentence-transformers (`all-mpnet-base-v2`)
- **PDF ingestion**: pypdf
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit
- **Evaluation**: custom LLM-as-judge scoring script

## Knowledge base

A real-world Wikipedia article on UPI (Unified Payments Interface) — 168,608 characters, chunked into 199 segments using fixed-size character windows with overlap. Not hand-crafted sample data.

## How to run it

\bash
# Clone and set up
git clone https://github.com/AnuragBharti22/ai-support-agent.git
cd ai-support-agent
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# Add your own Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Build the vector index from knowledge.pdf
python build_index_pdf.py

# Run the agent directly in the terminal
python agent.py

# Or run it as a web API
uvicorn main:app --reload
# then visit http://127.0.0.1:8000/docs

# Or run the chat UI
streamlit run app.py
\

## Evaluation results

Ran an automated 5-question test suite using an independent LLM-as-judge to score whether each answer was faithful to the retrieved context (or correctly declined to answer when it shouldn't).

Tested against the real-world UPI knowledge base described above — not sample data written to make the demo work.

**Score: 5/5 (100%)**

Test cases covered: factual questions requiring retrieval, a tool-triggering question, and an out-of-scope question that should be correctly declined and logged.

## Known limitations

- Grounding is generally strong but not 100% consistent — the model occasionally adds general knowledge beyond the retrieved context despite explicit "use only context" instructions. This is a known limitation of smaller/faster models and could be improved with a stricter system prompt or a larger judge model for validation.
- Initial retrieval testing on the real-world UPI dataset revealed that the default lightweight embedding model (`all-MiniLM-L6-v2`) struggled with short factual queries like "Who developed X?" — the correct chunk existed but didn't surface in the top-5 matches (similarity distance ~0.9-1.0). Diagnosed this by directly inspecting retrieved chunks and their distances, then fixed it by switching to a stronger embedding model (`all-mpnet-base-v2`) and increasing retrieved chunks from 3 to 8 — improving match quality significantly (distance ~0.45).
- Chunking uses fixed-size character windows with overlap, which is more robust than paragraph splitting for messy real-world PDFs, but still isn't as accurate as semantic or recursive chunking — a fact can still occasionally be split awkwardly across chunk boundaries.
- Single-document knowledge base for this demo — architecture supports scaling to many documents without changes.

## What I'd improve with more time

- Hybrid search (keyword + semantic) for more robust retrieval
- Reranking retrieved chunks before passing to the LLM
- Streaming responses for better UX
- Multi-document support with source citation in answers