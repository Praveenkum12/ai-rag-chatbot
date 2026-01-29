from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.2
    )

def get_prompt():
    return ChatPromptTemplate.from_template(
        """
        You are a helpful and professional AI assistant.
        
        Guidelines:
        1. If the user's message is a greeting (like 'hello', 'hi', 'hey') or a general pleasantry, respond warmly and invite them to ask questions about their documents.
        2. For factual questions, use the provided Context strictly to answer.
        3. If a question is specifically about the documents but the information is missing from the Context, say "I don't know based on the provided documents."
        4. Always maintain a helpful tone.

        Context:
        {context}

        Question:
        {question}
        """
    )


def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()
    prompt = get_prompt()

    # This is the "Full Auto" chain used at startup
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
