"""Streamlit UI: chat with any YouTube video."""
import os
import streamlit as st
from dotenv import load_dotenv

from rag.transcript import extract_video_id, get_transcript, load_manual, fmt_time
from rag.indexer import segments_to_documents, build_vector_store
from rag.chain import build_chain

load_dotenv()
st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬")
st.title("🎬 YouTube Video Q&A Chatbot")
st.caption("RAG with LangChain · FAISS · HuggingFace embeddings · Gemini")

with st.sidebar:
    # Owner's key comes from .env (local) or Streamlit Secrets (deployed) and is never shown.
    if not os.getenv("GOOGLE_API_KEY"):
        api_key = st.text_input("Google API key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.success("API key loaded ✅")
    url = st.text_input("YouTube URL or video ID", "https://www.youtube.com/watch?v=LPZh9BOjkQs")
    uploaded = st.file_uploader("Manual transcript (JSON) – used if YouTube blocks access", type="json")
    use_sample = st.checkbox("Use bundled sample transcript", value=True)
    go = st.button("Process video", type="primary")


@st.cache_resource(show_spinner=False)
def index_video(video_id, segments_key, _segments):
    docs = segments_to_documents(_segments, video_id)
    return build_vector_store(docs), len(docs)


if go:
    vid = extract_video_id(url)
    sample = f"sample_data/{vid}.json"
    try:
        if uploaded:
            segments, source = load_manual(uploaded), "manual_upload"
        else:
            fallback = sample if use_sample and os.path.exists(sample) else None
            segments, source = get_transcript(vid, fallback)
        with st.spinner("Embedding transcript..."):
            store, n = index_video(vid, len(segments), segments)
        st.session_state.update(chain=build_chain(store), video_id=vid, messages=[])
        st.success(f"Indexed {n} chunks from {len(segments)} segments (source: {source})")
    except Exception as e:
        st.error(str(e))

if "chain" in st.session_state:
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    if q := st.chat_input("Ask something about the video..."):
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    out = st.session_state.chain.invoke(q)
                    vid = st.session_state.video_id
                    srcs = " · ".join(
                        f"[{fmt_time(d.metadata['start'])}](https://youtu.be/{vid}?t={int(d.metadata['start'])})"
                        for d in out["docs"])
                    reply = f"{out['answer']}\n\n**Sources:** {srcs}"
                except Exception as e:
                    reply = f"⚠️ Generation failed: {e}"
            st.markdown(reply)
        st.session_state.messages += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
