import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Basic RAG",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Basic RAG")
st.caption("Ask questions about your indexed documents.")

query = st.text_area(
    "Your question",
    placeholder="What is DSPy?",
)

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"query": query},
                    timeout=180.0,
                )
                response.raise_for_status()
                result = response.json()

            except httpx.HTTPError:
                st.error("Unable to get an answer from the RAG API.")
            else:
                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")
                for source in result["sources"]:
                    st.write(
                        f"[Source {source['id']}] "
                        f"{source['file']} — Page {source['page']}"
                    )

st.divider()
st.subheader("Upload PDF")

st.info(
    "Uploading a new PDF replaces the active knowledge base "
    "after successful indexing."
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)

if st.button(
    "Upload and Index",
    disabled=uploaded_file is None,
):
    with st.spinner("Uploading and indexing document..."):
        try:
            response = httpx.post(
                f"{API_URL}/documents/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                },
                timeout=600.0,
            )
            response.raise_for_status()

        except httpx.HTTPError:
            st.error("Document upload or indexing failed.")
        else:
            result = response.json()
            st.success("Document indexed successfully!")
            st.json(result)