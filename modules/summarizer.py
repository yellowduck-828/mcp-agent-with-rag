from langchain_core.prompts import ChatPromptTemplate
from config import llm

summary_prompt = ChatPromptTemplate.from_template("""
问题：{question}
搜索内容：
{content}

只提取有用信息，避免编造。
""")

summary_chain = summary_prompt | llm


def summarize(question, content):
    return summary_chain.invoke({
        "question": question,
        "content": content
    }).content