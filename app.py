# app.py
import streamlit as st
from utils import extract_text, preprocess, compute_similarity, find_similar_lines

# Streamlit UI setup
st.set_page_config(page_title="Document Similarity Checker", layout="wide")

st.title("📄 Document Similarity Checker (PDF / TXT)")

col1, col2 = st.columns(2)

# File upload for both documents
with col1:
    file1 = st.file_uploader("Upload Document 1", type=["pdf", "txt"], key="doc1")

with col2:
    file2 = st.file_uploader("Upload Document 2", type=["pdf", "txt"], key="doc2")

# Check if both files are uploaded
if file1 and file2:

    # Extract text from both documents
    with st.status("Extracting text from documents...", expanded=True) as status:
        raw1 = extract_text(file1)
        raw2 = extract_text(file2)
        status.update(label="✅ Text extracted successfully!", state="complete")

    # Preprocess the extracted text
    with st.spinner("Preprocessing text..."):
        processed1 = " ".join(preprocess(raw1))
        processed2 = " ".join(preprocess(raw2))

    # Display the raw text for both documents
    st.subheader("📄 Raw Text Preview")
    with st.expander("Document 1"):
        st.text_area("Text from Document 1", raw1, height=200)
    with st.expander("Document 2"):
        st.text_area("Text from Document 2", raw2, height=200)

    # Compute similarity score between the two preprocessed documents
    with st.status("Generating similarity score...", expanded=True) as status:
        similarity = compute_similarity(processed1, processed2)
        status.update(label=f"✅ Similarity score computed: **{similarity:.2f}**", state="complete")

    # Display the overall similarity score
    st.success(f"Overall Similarity Score: **{similarity:.2f}**")

    # Find and highlight similar lines
    with st.status("Finding similar lines...", expanded=True) as status:
        similar_lines = find_similar_lines(raw1, raw2, threshold=0.4)
        status.update(label="✅ Similar lines found and highlighted!", state="complete")

    # Display the similar lines with highlights
    st.subheader("🔍 Similar Lines Highlight")
    if similar_lines:
        for i, (hl1, hl2, score) in enumerate(similar_lines):
            st.markdown(f"**Match {i+1}** (Score: `{score:.2f}`):", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-left: 1rem;'><b>📝 Doc 1:</b> {hl1}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-left: 1rem;'><b>📝 Doc 2:</b> {hl2}</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.info("No similar lines above threshold found.")
