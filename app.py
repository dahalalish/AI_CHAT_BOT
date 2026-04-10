import streamlit as st
from hybrid import hybrid_execute

st.set_page_config(page_title="Payer Specific Chatbot")

st.title("Payer Specific Chatbot!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    result = hybrid_execute(prompt)
    answer = result["answer"]

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

    # Optional debug info
    with st.expander("Debug Info"):
        st.write("Type:", result["type"])

        if result["type"] == "HYBRID":
            st.write("SQL:", result["sql"])
            st.write("RAG:", result["rag"])
