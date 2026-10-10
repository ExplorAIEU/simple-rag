from pathlib import Path

import streamlit as st

from rag import LocalRAG


ROOT = Path(__file__).parent
DOCUMENTS_DIR = ROOT / "documents"
INDEX_DIR = ROOT / ".rag_index"

st.set_page_config(
    page_title="Local Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Local Document Assistant")
st.caption(
    "Ask questions about your documents. Files, embeddings, and conversations stay on this computer."
)

with st.sidebar:
    st.header("Settings")
    chat_model = st.text_input("Chat model", value="qwen3:4b")
    embedding_model = st.text_input("Embedding model", value="bge-m3")
    top_k = st.slider("Sources per answer", 2, 10, 5)
    st.divider()
    st.markdown(
        "**Document folder**  \n"
        f"`{DOCUMENTS_DIR}`\n\n"
        "Supported: PDF, Word, TXT, Markdown, CSV, JSON"
    )

rag = LocalRAG(
    documents_dir=DOCUMENTS_DIR,
    index_dir=INDEX_DIR,
    chat_model=chat_model,
    embedding_model=embedding_model,
)
files = rag.document_files()

with st.sidebar:
    st.metric("Documents found", len(files))
    index_current = False
    try:
        index_current = rag.index_is_current()
    except Exception as error:
        st.warning(f"Could not inspect the index: {error}")

    if index_current:
        st.success("Index is up to date")
    elif files:
        st.warning("Documents changed — rebuild the index")
    else:
        st.info("Add documents to the folder, then build the index")

    if st.button("Build / refresh index", type="primary", use_container_width=True):
        try:
            with st.spinner("Reading and embedding documents…"):
                count = rag.build_index()
            st.success(f"Indexed {count} text chunks.")
            st.rerun()
        except Exception as error:
            st.error(str(error))
            st.info(
                "Make sure Ollama is running and the embedding model is installed:\n\n"
                "`ollama pull bge-m3`"
            )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['number']}. {source['name']}** "
                        f"— {source['location']} · relevance {source['score']:.0%}"
                    )
                    st.caption(source["preview"])

question = st.chat_input(
    "Ask a question in English or Chinese…",
    disabled=not index_current,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1]
        if item["role"] in {"user", "assistant"}
    ]

    with st.chat_message("assistant"):
        try:
            stream, results = rag.answer(question, history=history, top_k=top_k)
            answer = st.write_stream(stream)
            sources = [
                {
                    "number": number,
                    "name": result.chunk.source,
                    "location": result.chunk.location,
                    "score": result.score,
                    "preview": (
                        result.chunk.text[:300] + "…"
                        if len(result.chunk.text) > 300
                        else result.chunk.text
                    ),
                }
                for number, result in enumerate(results, start=1)
            ]
            with st.expander("Sources used"):
                for source in sources:
                    st.markdown(
                        f"**{source['number']}. {source['name']}** "
                        f"— {source['location']} · relevance {source['score']:.0%}"
                    )
                    st.caption(source["preview"])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as error:
            st.error(str(error))
            st.info(
                "Make sure Ollama is running and the models are installed:\n\n"
                "`ollama pull qwen3:4b`\n\n`ollama pull bge-m3`"
            )
