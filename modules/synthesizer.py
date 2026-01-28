from langchain_core.prompts import ChatPromptTemplate
from config import llm

synth_prompt = ChatPromptTemplate.from_template("""
以下是针对问题的多条搜索总结，请综合生成最终答案：

{points}
""")

synth_chain = synth_prompt | llm


def synthesize(points):
    return synth_chain.invoke({"points": "\n".join(points)}).content
