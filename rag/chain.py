"""LangChain RAG chain (LCEL) with Gemini, returning answer + timestamped sources."""
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.transcript import fmt_time

PROMPT = PromptTemplate(
    template="""You are a helpful assistant that answers questions about a YouTube video.
Answer ONLY from the transcript context below. If the context is insufficient, say you don't know.
Cite the timestamps (e.g. [1:23]) you used.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)


def format_docs(docs) -> str:
    return "\n\n".join(f"[{fmt_time(d.metadata['start'])}] {d.page_content}" for d in docs)


def build_chain(vector_store, model="gemini-3.1-flash-lite", k=4, temperature=0.2):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # retrieve once, reuse docs for both the answer and the source list
    chain = RunnableParallel(
        docs=retriever, question=RunnablePassthrough()
    ).assign(
        answer=(
            {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
            | PROMPT | llm | StrOutputParser()
        )
    )
    return chain
