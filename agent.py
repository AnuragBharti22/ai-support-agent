import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

collection = chroma_client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_fn
)


# --- Tool 1: get current date/time ---
def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Tool 2: log a question we couldn't answer ---
def log_unanswered_question(question):
    with open("unanswered_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()}: {question}\n")
    return "Question has been logged for follow-up."


# --- Tool schemas ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_unanswered_question",
            "description": "Log a question that could not be answered from the available context, so it can be followed up on later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The original question that could not be answered."
                    }
                },
                "required": ["question"]
            }
        }
    }
]


def retrieve_context(query):
    results = collection.query(query_texts=[query], n_results=8)
    return "\n\n".join(results["documents"][0])


def ask(query):
    context = retrieve_context(query)

    messages = [
        {
            "role": "user",
            "content": f"""Answer the question using ONLY the context below.
Use the get_current_datetime tool only if the question is specifically asking about the current date or time.
If the answer isn't in the context and doesn't need the datetime tool, use the log_unanswered_question tool to log it, then tell the user you don't know but have logged it for follow-up.

Context:
{context}

Question: {query}"""
        }
    ]

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        print(f"[Agent decided to call tool: {tool_name}]")

        if tool_name == "get_current_datetime":
            result = get_current_datetime()
        elif tool_name == "log_unanswered_question":
            args = json.loads(tool_call.function.arguments)
            result = log_unanswered_question(args["question"])
        else:
            result = "Unknown tool."

        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

        final_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )
        return final_response.choices[0].message.content

    return message.content


if __name__ == "__main__":
    questions = [
        "What is UPI?",
        "Who developed UPI?",
        "What is the current date and time?"
    ]

    for q in questions:
        answer = ask(q)
        print(f"\nQuestion: {q}")
        print(f"Answer: {answer}")
        print("-" * 50)