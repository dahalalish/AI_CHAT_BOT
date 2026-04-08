import streamlit as st
from hybrid import hybrid_execute

st.set_page_config(page_title="Offline Payer Chatbot")

st.title("💬 Offline US Payer Chatbot (SQL + RAG)")

query = st.text_input("Ask your question:")

if query:
    result = hybrid_execute(query)

    st.write("### Answer")
    st.write(result["answer"])

    with st.expander("Debug Info"):
        st.write("Type:", result["type"])

        if result["type"] == "HYBRID":
            st.write("SQL:", result["sql"])
            st.write("RAG:", result["rag"])