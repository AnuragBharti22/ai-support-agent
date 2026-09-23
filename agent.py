import chromadb
from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")


# --- Tool definition: a real Python function ---
def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Tool schema: tells the LLM this tool exists and how to use it ---
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
    }
]


def retrieve_context(query):
    results = collection.query(query_texts=[query], n_results=3)
    return "\n\n".join(results["documents"][0])


def ask(query):
    context = retrieve_context(query)

    messages = [
        {
            "role": "user",
            "content": f"""Answer the question using ONLY the context below.
Only use the get_current_datetime tool if the question is specifically asking about the current date or time.
If the answer isn't in the context and doesn't need the datetime tool, say "I don't know based on the provided documents."


Context:
{context}

Question: {query}"""
        }
    ]

    # Step 1: Let the model decide if it needs a tool
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    # Step 2: Check if the model wants to call a tool
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        print(f"[Agent decided to call tool: {tool_call.function.name}]")

        # Step 3: Actually run the real Python function
        result = get_current_datetime()

        # Step 4: Send the tool's result back to the model for a final answer
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

    # No tool needed — just return the direct answer
    return message.content


## Test it
questions = [
    "How do I keep my API key safe?",       # should use context, no tool
    "What is the current date and time?",     # should use tool
    "What's the capital of France?"           # should say "I don't know"
]

for q in questions:
    answer = ask(q)
    print(f"\nQuestion: {q}")
    print(f"Answer: {answer}")
    print("-" * 50)