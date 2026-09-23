import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

def ask(query):
    # Step 1: Retrieve relevant chunks
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_chunks = results["documents"][0]

    # Step 2: Build context from retrieved chunks
    context = "\n\n".join(retrieved_chunks)

    # Step 3: Build the prompt with context + question
    prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}

Answer:"""

    # Step 4: Send to the LLM
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Test it
question = "What's the capital of France?"
answer = ask(question)
print(f"Question: {question}\n")
print(f"Answer: {answer}")