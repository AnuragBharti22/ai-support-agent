import streamlit as st
from agent import ask

st.set_page_config(page_title="AI Support Agent", page_icon="🤖")

st.title("🤖 AI Support Agent")
st.caption("RAG + Agentic AI — retrieves from documents, calls tools when needed")

# Keep chat history across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input box at the bottom
user_question = st.chat_input("Ask a question...")

if user_question:
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Get and show the agent's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(user_question)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})