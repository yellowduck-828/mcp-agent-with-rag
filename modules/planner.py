from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import llm
import json

planner_prompt = ChatPromptTemplate.from_template("""
请把问题拆分成3-5个子问题，并给出搜索关键词。
问题：{query}

输出 JSON：
{{
 "sub_questions": [...],
 "keywords": [...]
}}
""")

planner_chain = planner_prompt | llm | JsonOutputParser()

def make_plan(query):
    resp = planner_chain.invoke({"query": query})
    if not isinstance(resp, dict):
        raise ValueError("规划结果解析失败：非字典")
    subs = resp.get("sub_questions") or []
    kws = resp.get("keywords") or []
    if not subs or not kws:
        raise ValueError("规划结果缺少 sub_questions 或 keywords")
    return {"sub_questions": subs, "keywords": kws}
