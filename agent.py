import chromadb
from chromadb.utils import embedding_functions
from groq import Groq, BadRequestError, RateLimitError
from dotenv import load_dotenv
from ddgs import DDGS
import os
import json
import time
from datetime import datetime

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 4     # how many times the agent may loop through tools
MAX_SEARCHES = 3        # hard cap on web/news searches per question
SEARCH_TOOLS = {"web_search", "news_search"}

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn
)


def chat_completion(**kwargs):
    """Call the LLM, waiting and retrying if the API rate limit is hit."""
    for attempt in range(1, 6):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait = 2 * attempt
            print(f"[Rate limited by the API - waiting {wait}s and retrying]")
            time.sleep(wait)
    return groq_client.chat.completions.create(**kwargs)  # last attempt; raises if still limited


# ---------- TOOLS (real Python functions) ----------

def get_current_datetime():
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


def search_documents(query):
    """Search the user's uploaded documents (vector database)."""
    results = collection.query(query_texts=[query], n_results=6)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "No relevant passages found in the documents."
    return "\n\n".join(f"[Source: {m['source']}]\n{d}" for d, m in zip(docs, metas))


def web_search(query):
    """Search the live web for current facts."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Web search failed: {e}"
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"[{r.get('title', '')}] ({r.get('href', '')})\n{r.get('body', '')}"
        for r in results
    )


def news_search(query):
    """Search news articles. Results are dated and sorted newest first."""
    try:
        results = DDGS().news(query, max_results=10)
    except Exception as e:
        return f"News search failed: {e}"
    if not results:
        return "No news results found."
    results.sort(key=lambda r: r.get("date") or "", reverse=True)
    return "\n\n".join(
        f"[{r.get('date') or 'no date'}] {r.get('title', '')} - {r.get('source', '')}\n"
        f"{r.get('body', '')}\n{r.get('url', '')}"
        for r in results[:6]
    )


TOOL_FUNCTIONS = {
    "get_current_datetime": lambda args: get_current_datetime(),
    "search_documents": lambda args: search_documents(args["query"]),
    "web_search": lambda args: web_search(args["query"]),
    "news_search": lambda args: news_search(args["query"]),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date, day of week, and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the user's uploaded knowledge base documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A focused search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": "Search recent news articles. Returns dated results, newest first. Use for latest news and recent events.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A short news search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current facts, prices, or to verify something.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A focused search query."}},
                "required": ["query"],
            },
        },
    },
]


def list_document_names():
    """Names of the documents currently in the knowledge base."""
    try:
        metas = collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})
    except Exception:
        return []


def build_system_prompt():
    today = datetime.now().strftime("%A, %d %B %Y")
    docs = list_document_names()
    doc_line = ", ".join(docs) if docs else "none uploaded yet"
    return f"""You are a helpful, knowledgeable AI assistant. Today's date is {today}.

You can answer general questions on any topic from your own knowledge, and you have tools:

- search_documents: the user's knowledge base currently contains these documents: {doc_line}. If a question could plausibly be answered by them (judge by their names and topics), ALWAYS call this tool first, even for who/when/what questions you think you already know, and do not use web_search for facts the documents may contain. Cite the document name in plain parentheses, like (Source: upi.pdf).
- news_search: use for latest news, "this week", "what happened", or anything recent. Results are dated and sorted newest first. Report the latest status: if a newer result says something was postponed, changed or cancelled, say so and give the date.
- web_search: use for other current facts, prices, or to verify something.
- You may make at most 3 searches in total (news_search and web_search combined), then answer. Make queries specific: expand acronyms and add distinguishing context (for example "Unified Payments Interface NPCI India", not "UPI", which can also mean United Press International). If the first results are about something else, search again with a better query. If a result describes something as possible, proposed, expected or under review, make one more search to check whether it has since been decided, and report the latest status. State the date of the newest result you found instead of claiming nothing newer exists. Use today's date above when forming queries. Only state facts that appear in the search results; do not add details from memory or guess, and never attribute a decision to an organization the results don't name. If results conflict or are thin, say so. End with the source URLs. Never use citation markers like 【5†L1-L5】. Keep news answers short, in plain sentences, with no large tables.
- get_current_datetime: use when the exact current time matters.

For general knowledge, reasoning, writing, coding and explanations, answer directly without tools. Be clear and concise. If you are unsure, say so honestly instead of guessing."""


def run_tool(name, args):
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    try:
        return func(args)
    except Exception as e:
        return f"Tool error: {e}"


def force_final_answer(messages, last_tool_output):
    """Last resort: ask the model to answer now, without calling any tools."""
    final_messages = messages + [{
        "role": "user",
        "content": "Write your final answer to my question now, using any information gathered above. Do not call any tools."
    }]
    try:
        response = chat_completion(model=MODEL, messages=final_messages)
        return response.choices[0].message.content
    except BadRequestError:
        return "Sorry, I had trouble putting together a final answer. Here is what I found:\n\n" + last_tool_output[:1500]


def ask(query, history=None):
    """history: optional list of earlier {'role': 'user'|'assistant', 'content': str} messages."""
    messages = [{"role": "system", "content": build_system_prompt()}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    searches_used = 0
    last_tool_output = ""

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = chat_completion(model=MODEL, messages=messages, tools=TOOLS)
        except BadRequestError:
            # The model produced a malformed tool call
            print("[Malformed tool call from model - asking it to answer directly]")
            return force_final_answer(messages, last_tool_output)

        message = response.choices[0].message

        # No tool requested -> this is the final answer
        if not message.tool_calls:
            return message.content

        # Run every tool the model asked for, then loop back with the results
        messages.append(message)
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name in SEARCH_TOOLS and searches_used >= MAX_SEARCHES:
                print(f"[Blocked extra search: limit of {MAX_SEARCHES} reached]")
                result = "Search limit reached. Do not search again. Answer now using the results already gathered."
            else:
                print(f"[Agent called tool: {name} {args}]")
                if name in SEARCH_TOOLS:
                    searches_used += 1
                result = run_tool(name, args)

            last_tool_output = result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return force_final_answer(messages, last_tool_output)


if __name__ == "__main__":
    print("Documents in knowledge base:", list_document_names())

    questions = [
        "Who developed UPI?",
        "When was UPI launched?",
    ]
    for q in questions:
        print(f"\nQuestion: {q}")
        print(f"Answer: {ask(q)}")
        print("-" * 50)