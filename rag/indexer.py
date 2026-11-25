"""Chunking + embeddings + FAISS, keeping timestamps as metadata."""
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def segments_to_documents(segments, video_id: str, chunk_size=500, overlap_segments=2) -> List[Document]:
    """Group transcript segments into ~chunk_size-char chunks.
    Unlike plain text splitting, every chunk remembers WHEN it was said (start time)."""
    docs, buf, i = [], [], 0
    while i < len(segments):
        buf.append(segments[i])
        if sum(len(s["text"]) for s in buf) >= chunk_size or i == len(segments) - 1:
            start = buf[0]["start"]
            end = buf[-1]["start"] + buf[-1]["duration"]
            docs.append(Document(
                page_content=" ".join(s["text"] for s in buf),
                metadata={"video_id": video_id, "start": start, "end": end},
            ))
            buf = buf[-overlap_segments:] if i < len(segments) - 1 else []
        i += 1
    return docs


def build_vector_store(docs: List[Document]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(docs, embeddings)
