from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the VEX Knowledge Base (the ingested data).

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions from teachers/educators about how to effectively teach STEM with VEX educational robots and VEX platforms. Provide the best answer in the fewest words you think is necessary, but no more than four sentences.
If you don't know the answer with high certainty say, "Hmm, I've not been trained yet to give a reliable answer. But would you like me to speculate from the information that I do have?" If the user replies, "Yes", then you may give your best answer based on the ingested data. If the user replies, "No", then you may say, "OK I'll ask my human colleagues and get back to you."
If the question is not about the VEX continuum or VEX robots or the ingested data, politely reply that you're supposed to talk only about VEX and ask if they have a question about VEX.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
