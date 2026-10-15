# AI Support Agent — RAG + Tool-Calling Assistant

A general-purpose AI assistant built from scratch with Python. It answers general questions from the model's own knowledge, searches **your documents** (RAG) when a question is covered by them, and looks up **live news and web results** for current events. The agent decides for itself which tool to use for each question.

## What it does

- **Answers general questions directly** (facts, explanations, coding, math) without unnecessary tool calls
- **Searches your uploaded documents** (PDF or text) using vector search and cites the source file
- **Searches news and the web** for current information, with dated results and source URLs
- **Remembers the conversation**, so follow-up questions like "when was it launched?" work
- **Runs three ways**: terminal, FastAPI web API, or a Streamlit chat UI

## How it works

```
User question (+ earlier chat history)
        |
        v
LLM (Groq, openai/gpt-oss-120b) decides what to do
        |
        +--> answer directly (general knowledge)
        +--> search_documents  -> vector search over your files (ChromaDB)
        +--> news_search       -> dated news results, newest first
        +--> web_search        -> general web results
        +--> get_current_datetime
        |
        v
Tool results go back to the LLM -> final answer (with sources)
```

The agent runs in a loop: it can call a tool, read the result, and call another before answering. Limits are enforced in code, not just in the prompt: at most 4 tool rounds and 3 searches per question.

## Tools

| Tool | When the agent uses it |
|---|---|
| `search_documents` | Question may be covered by the uploaded documents |
| `news_search` | Latest news, "this week", recent events (results sorted newest first) |
| `web_search` | Other current facts, prices, verification |
| `get_current_datetime` | Exact current time |

## Tech stack

- **LLM**: Groq API (`openai/gpt-oss-120b`)
- **Vector database**: ChromaDB (local, persistent)
- **Embeddings**: sentence-transformers (`all-mpnet-base-v2`)
- **Document ingestion**: pypdf, fixed-size chunking with overlap
- **Search**: `ddgs` (text and news search, no API key needed)
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit

## Project structure

```
agent.py                  # the agent: tools, system prompt, tool loop
build_knowledge_base.py   # reads documents/ and builds the vector index
check_index.py            # shows what is stored in the vector database
evaluate.py               # automated behavior tests
app.py                    # Streamlit chat UI
main.py                   # FastAPI web API
documents/                # your PDFs / text files (not included in this repo)
```

## How to run it

```bash
# 1. Clone and set up
git clone https://github.com/AnuragBharti22/ai-support-agent.git
cd ai-support-agent
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Add your Groq API key (free at console.groq.com)
echo GROQ_API_KEY=your_key_here > .env

# 3. Add documents: create a documents/ folder and put PDF or .txt files in it
#    (the demo used the Wikipedia article on UPI saved as documents/upi.pdf)

# 4. Build the vector index (takes a few minutes the first time)
python build_knowledge_base.py

# 5. Run it
python agent.py                # quick terminal test
streamlit run app.py           # chat UI
uvicorn main:app --reload      # web API, docs at http://127.0.0.1:8000/docs
```

Adding a new document later is just: drop the file in `documents/` and re-run `python build_knowledge_base.py`.

## Evaluation

`evaluate.py` runs 8 behavior tests and checks two things for each: **did the agent pick the right tool**, and **does the answer contain the expected fact**. It doesn't use an LLM judge, so results are fast and repeatable.

| Test group | What it checks |
|---|---|
| 3 document questions | Uses `search_documents` and returns the correct fact from the PDF |
| 2 general-knowledge questions | Answers directly with **no** tool calls |
| 1 time question | Uses the datetime tool |
| 1 news question | Uses a search tool and includes source links |
| 1 follow-up question | Resolves "it" correctly using chat history |

**Latest run: 8/8.** The first run scored 5/8, and the failures were useful: the agent sometimes searched the web for facts that were in the documents (fixed by tightening the routing instructions), a burst of requests hit the API's tokens-per-minute limit and crashed the agent (fixed with retry and backoff), and one test was itself wrong. Model behavior varies slightly between runs, and 8 tests is a small suite.

## Engineering notes: problems found and fixed

- **Retrieval missed a fact that was in the data.** With the default embedding model and small chunks, "Who developed UPI?" did not surface the correct chunk (similarity distance around 0.9-1.0). Inspecting the raw retrieved chunks showed the data was fine and retrieval was the problem. Switching to `all-mpnet-base-v2`, larger chunks (1000 characters, 150 overlap) and more retrieved results brought the distance to about 0.45 and fixed it.
- **The model sometimes produced malformed tool calls**, which crashed the API request. The agent now catches this, asks the model to answer from what it already has, and caps tool rounds so it can never loop forever.
- **Web search returned stale or wrong-topic results.** The news query "UPI" matched United Press International, and plain web search ranked old articles above new ones. Fixes: a dedicated news tool that returns dated results newest first, today's date in the system prompt, and instructions to expand acronyms and to check whether "proposed" or "possible" items have since been decided.
- **API rate limits**: the free tier caps tokens per minute, so LLM calls now retry with backoff.

## Known limitations

- **Web answers are only as reliable as their sources.** The agent can repeat a weak or misleading source with the same confidence as a good one (during testing it once reported an RBI announcement that I could not find in any reputable source). Search results are snippets, not full articles, so details should be verified against primary sources.
- **Model behavior is not fully deterministic.** The same question can be routed slightly differently between runs.
- **Chunking is fixed-size**, which can split a fact across chunk boundaries; semantic or recursive chunking would be more accurate.
- **Free-tier limits**: Groq's free tier restricts tokens per minute, and the search library depends on public search endpoints that can throttle.
- The evaluation suite is small, and the news test checks that a search happened and links were returned, not that the story is correct.
- No user authentication, streaming responses, or per-passage citations yet.

## What I'd improve next

- Hybrid search (keyword + semantic) and reranking of retrieved chunks
- Source-quality filtering for web results, and fetching full pages instead of snippets
- Streaming responses in the chat UI
- Passage-level citations (which part of which document)
- A larger evaluation set, including adversarial and multi-document questions