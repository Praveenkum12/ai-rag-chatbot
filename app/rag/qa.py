from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4.1-nano",
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

    # Use RunnableParallel to pass research (context) through to the final output
    chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        }).assign(
            answer=(
                prompt
                | llm
                | StrOutputParser()
            )
        )
    )

    return chain
