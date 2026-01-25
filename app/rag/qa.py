from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {question}
    """
        )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
