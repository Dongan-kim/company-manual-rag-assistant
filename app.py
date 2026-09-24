import streamlit as st
import re

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# -------------------------
# Configuration
# -------------------------

CHROMA_PATH = "./chroma_langchain"
COLLECTION_NAME = "company_manual"
THRESHOLD = 1.30
TOP_K = 3


load_dotenv()


# -------------------------
# Streamlit page
# -------------------------

st.set_page_config(
    page_title="Company Manual Assistant",
    page_icon="📄"
)

st.title("📄 Company Manual Assistant")

st.write(
    "Ask a question about the company manual."
)


# -------------------------
# RAG components
# -------------------------

@st.cache_resource
def load_rag():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    llm = ChatOpenAI(
        model="gpt-5.6-luna"
    )

    return vector_store, llm


vector_store, llm = load_rag()


# -------------------------
# Prompt
# -------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a company manual assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
respond with exactly:

NOT_FOUND

Do not invent company policies.

When answering, cite the source page directly after the information
using this format:

[Page X]

Only cite pages that directly support the answer.

Context:
{context}

Question:
{question}
"""
)


# -------------------------
# User input
# -------------------------

question = st.text_input(
    "Question",
    placeholder="e.g. How much vacation do I get each year?"
)


if st.button("Ask") and question:

    with st.spinner("Searching the company manual..."):

        # -------------------------
        # Retrieval
        # -------------------------

        results = vector_store.similarity_search_with_score(
            question,
            k=TOP_K
        )

        best_score = results[0][1]


        # -------------------------
        # Threshold check
        # -------------------------

        if best_score > THRESHOLD:

            st.subheader("Answer")

            st.info(
                "I could not find that information "
                "in the company manual."
            )

        else:

            documents = [
                document
                for document, score in results
            ]


            # -------------------------
            # Build context
            # -------------------------

            context_parts = []

            for document in documents:

                source = document.metadata["source"]
                page = document.metadata["page"]

                context_parts.append(
                    f"Source: {source}\n"
                    f"Page: {page}\n"
                    f"{document.page_content}"
                )

            context = "\n\n---\n\n".join(
                context_parts
            )


            # -------------------------
            # LLM
            # -------------------------

            messages = prompt.invoke({
                "context": context,
                "question": question
            })

            response = llm.invoke(messages)

            answer = response.content.strip()


            # -------------------------
            # Display answer
            # -------------------------

            st.subheader("Answer")

            if answer == "NOT_FOUND":

                st.info(
                    "I could not find that information "
                    "in the company manual."
                )

            else:

                st.write(answer)

                # -------------------------
                # Extract cited pages
                # -------------------------

                cited_pages = re.findall(
                    r"\[Page (\d+)\]",
                    answer
                )

                cited_pages = list(dict.fromkeys(cited_pages))


                # -------------------------
                # Sources
                # -------------------------

                if cited_pages:

                    st.subheader("Sources")

                    for page in cited_pages:

                        st.write(
                            f"• company_manual.pdf - Page {page}"
                        )

