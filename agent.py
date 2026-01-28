import os
import re
import json
import time
import threading
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import DEEPSEEK_API_KEY
from multi_mcp_client import MultiMCPClient

# 每轮最多允许的工具调用次数，超出将被截断以避免重复浪费
MAX_TOOL_CALLS_PER_ROUND = 3

SYSTEM_PROMPT = """
你是一个可靠的智能助手，需要用“思考→行动→观察→总结”的 ReAct 流程解决问题。
你需要解决一个问题。为此，你需要将问题分解为多个步骤。
对于每个步骤，首先使用 <thought> 思考要做什么，然后使用可用工具之一决定一个 <action>。
接着，你将根据你的行动从环境/工具中收到一个 <observation>。
持续这个思考和行动的过程，直到你有足够的信息来提供 <final_answer>。

所有步骤请严格使用以下 XML 标签格式输出：
- <question> 用户问题
- <thought> 思考
- <action> 采取的工具操作
- <observation> 工具或环境返回的结果
- <final_answer> 最终答案

⸻

例子 1:

<question>埃菲尔铁塔有多高？</question>
<thought>我需要找到埃菲尔铁塔的高度。可以使用搜索工具。</thought>
<action>get_height("埃菲尔铁塔")</action>
<observation>埃菲尔铁塔的高度约为330米（包含天线）。</observation>
<thought>搜索结果显示了高度。我已经得到答案了。</thought>
<final_answer>埃菲尔铁塔的高度约为330米。</final_answer>

⸻

例子 2:

<question>帮我找一个简单的番茄炒蛋食谱，并看看家里的冰箱里有没有西红柿。</question>
<thought>这个任务分两步。第一步，找到番茄炒蛋的食谱。第二步，检查冰箱里是否有西红柿。我先用 find_recipe 工具找食谱。</thought>
<action>find_recipe(dish="番茄炒蛋")</action>
<observation>简单的番茄炒蛋食谱：将2个鸡蛋打散，2个番茄切块。热油，先炒鸡蛋，盛出。再热油，炒番茄至软烂，加入鸡蛋，放盐调味即可。</observation>
<thought>好的，我已经有食谱了。食谱需要西红柿。现在我需要用 check_fridge 工具看看冰箱里有没有西红柿。</thought>
<action>check_fridge(item="西红柿")</action>
<observation>冰箱检查结果：有3个西红柿。</observation>
<thought>我找到了食谱，并且确认了冰箱里有西红柿。可以回答问题了。</thought>
<final_answer>简单的番茄炒蛋食谱是：鸡蛋打散，番茄切块。先炒鸡蛋，再炒番茄，混合后加盐调味。冰箱里有3个西红柿。</final_answer>

⸻

请严格遵守：
- 你每次回答都必须包括两个标签，第一个是 <thought>，第二个是 <action> 或 <final_answer>
- 输出 <action> 后立即停止生成，等待真实的 <observation>，擅自生成 <observation> 将导致错误
- <thought> / <action> / <observation> 的内容必须真实可验证，禁止臆造；<observation> 只能使用工具真实返回值，未收到工具结果时不要编写 <observation>
- 每条回复只能包含 1 个 <thought> 和 1 个 <action>/<final_answer>；禁止在同一条消息里输出多个 <thought> 或多个 <action>
- <thought> 仅允许写“接下来要调用哪个工具/准备做什么”，不得描述任何事实、搜索结果或结论；若未收到 <observation>，不得出现“根据搜索结果/根据工具返回”等措辞
- <action> 只写真实的工具调用及参数，不要夹带解释或结论
- 如果 <action> 中的某个工具参数有多行的话，请使用 \n 来表示，如：<action>write_to_file("/tmp/test.txt", "a\nb\nc")</action>


工具可用：
- 通过 MCP Server 暴露的工具集合（启动时自动获取工具列表与参数模式）

安全与真实性：
- 只能访问 workspace 目录内文件。
- 工具无结果就如实说明，禁止臆造。
- 有歧义要说明不确定性。

【重要规则】
- <action> 必须准确写出将要调用的真实工具名称和目的，不能编造不存在的调用。
- <observation> 必须忠实呈现最新一次工具真实返回结果的关键信息，禁止编造或改写与返回值不符的内容；如无调用则写“无”。未收到工具返回时不得输出“观察”。
- 一旦调用工具，必须以工具返回结果为唯一事实来源。
- 禁止自行推测、假设或修改工具返回的任何信息。
- 若工具返回与模型直觉冲突，以工具返回为准。
- 未完成工具调用时禁止给出 <final_answer>；工具返回后再总结。
- <final_answer> 只能基于已获取的 <observation> 信息，不得引入 observation 之外的新事实；如信息不足请说明不足或再次调用工具。
- 如需多次调用工具，必须按顺序完整输出每一次的 <thought>、<action>、<observation>，不得省略任何一步，也不得把多次调用合并成一条。
- 若调用了多个工具，所有调用及其 observation 必须体现在最终输出中（折叠区），最终回答前必须包含全部调用结果。
- 思考只写“当前一步”，禁止一次性预演多步；每次只能给出一个 <action> 并立即触发对应工具调用，拿到 observation 后再进行下一步思考。

输出规范（前端约定）：
- 中间推理（thought/action/observation）会折叠显示，请严格用 <thought> / <action> / <observation>。
- 最终对用户的回答放在 <final_answer>，不要再用 <final> 标签。
- 不输出原始 JSON；要简明、可验证。

当用户纠错时：
- 重新用工具验证，承认并修正，简要说明原因。

【时间相关的强制规则】
1. 只要问题中涉及以下任何内容：
   - 年、月、日、具体日期
   - “今天 / 昨天 / 前天 / 去年 / 明年 / 现在 / 当前”
   - “几年前 / 几天后 / 最近 / 此刻 / 当前时间 / 星期几”
   - 或任何需要基于“当前时间”进行判断、推理、换算的情况
   
   你【必须】先调用 get_current_datetime 工具获取当前系统时间。

2. 在调用 get_current_datetime 之前：
   - 严禁自行猜测当前年份、日期或时间
   - 严禁使用“我认为现在是…”、“假设现在是…”之类的表述

3. 一旦 get_current_datetime 返回结果：
   - 该结果是【唯一、不可质疑、不可修改的事实来源】
   - 直接使用返回的字段（如 weekday / weekday_cn / readable / iso / offset），禁止自行推算日期或星期
   - 后续所有推理、判断、年份/日期换算（如“去年”“明年”“星期几”）必须严格基于该返回值
   - 禁止使用与工具返回不一致的时间信息

4. 如果未调用 get_current_datetime 就涉及时间判断，视为严重错误。
"""


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "deepseek-chat",
        mcp_client: Optional[MultiMCPClient] = None,
        tool_call_timeout: int = 20,
        verbose: bool = False,
        max_rounds: int = 10,
    ):
        self.client = client
        # 默认接入本地 MCP server 和外部 fetch server
        self.mcp_client = mcp_client or MultiMCPClient(
            servers=[
                {"name": "local", "command": "python", "args": ["mcp_server.py"]},
                {
                    "name": "fetch",
                    "command": "uvx",
                    "args": ["mcp-server-fetch"],
                    # 使用项目内可写缓存目录，避免 ~/.cache/uv 权限/锁问题
                    "env": {"UV_CACHE_DIR": "/Users/wangluyao/Desktop/myagentbymcp/.uv-cache"},
                },
                {
                    "name": "amap",
                    "command": "npx",
                    "args": ["-y", "@amap/amap-maps-mcp-server"],
                    # 从环境变量读取高德 Key，需在启动前 source .env
                    "env": {"AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY", "")},
                },
            ]
        )
        self.tool_call_timeout = tool_call_timeout
        # 预取 MCP 工具 schema，失败自动重试以避免空列表
        self.tools_schema = self._fetch_tools_with_retry()
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose
        self.max_rounds = max_rounds

    def _fetch_tools_with_retry(self, retries: int = 3, delay: float = 1.0) -> List[Dict[str, Any]]:
        """获取工具 schema，失败时重试，避免启动时偶发空列表。"""
        last_exc: Optional[Exception] = None
        for _ in range(retries):
            try:
                tools = self.mcp_client.get_openai_tools()
                if tools:
                    return tools
            except Exception as exc:  # noqa: PERF203 - 捕获记录后重试
                last_exc = exc
            time.sleep(delay)
        if self.verbose:
            print(f"⚠️ 获取 MCP 工具列表失败：{last_exc}")
        return []

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式；若缓存为空则尝试刷新一次
        if not self.tools_schema:
            self.tools_schema = self._fetch_tools_with_retry()
        return self.tools_schema

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments or "{}")
        function_id = tool_call.id

        result = self.mcp_client.call_tool(
            function_name,
            function_args,
            timeout=self.tool_call_timeout,
        )
        function_call_content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt, return_details: bool = False, stop_event: Optional["threading.Event"] = None):
        """支持多轮工具调用的对话流程。
        return_details=True 时返回 dict，包含回复与本轮用到的工具列表。
        stop_event 用于外部请求中断。
        """
        self.messages.append({"role": "user", "content": prompt})

        round_idx = 0
        tool_log: List[str] = []
        tool_results: List[str] = []
        while True:
            if stop_event and stop_event.is_set():
                final = "对话已中断。"
                if return_details:
                    return {"content": final, "tools": tool_log, "tool_results": tool_results}
                return final
            round_idx += 1
            if round_idx > self.max_rounds:
                final = "对话已达最大轮次，可能存在工具请求超时或依赖外部网络不可达，请稍后重试或检查网络/代理。"
                if return_details:
                    return {"content": final, "tools": tool_log, "tool_results": tool_results}
                return final

            try:
                if stop_event and stop_event.is_set():
                    final = "对话已中断。"
                    if return_details:
                        return {"content": final, "tools": tool_log, "tool_results": tool_results}
                    return final
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.get_tool_schema(),
                    stream=False,
                    timeout=30,  # 放宽超时，减少长工具流程被中断
                )
                # DEBUG: 若无 tool_calls 也无内容，打印日志，避免静默结束
                choice_msg = response.choices[0].message
                if not choice_msg.tool_calls and not (choice_msg.content or "").strip():
                    if self.verbose:
                        print("⚠️ 模型返回空消息，无 tool_calls、无 content")
                    # 继续下一轮，尝试引导模型给出调用或回答
                    self.messages.append({"role": "assistant", "content": ""})
                    continue
            except Exception as exc:
                err_msg = f"模型请求超时或失败：{exc}"
                if return_details:
                    return {"content": err_msg, "tools": tool_log, "tool_results": tool_results}
                return err_msg

            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            # 先把带 tool_calls 的 assistant 消息放入历史
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
            }
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            self.messages.append(assistant_entry)

            # 如果已包含最终答案，则直接返回
            content_text = msg.content or ""
            has_final = bool(re.search(r"<final_answer>|<final>", content_text, re.IGNORECASE))

            # 如果没有工具调用，检查是否需要继续循环或返回
            if not tool_calls:
                # 1) 有最终答案，直接返回；若缺少 observation 则补全，方便前端展示完整工具结果
                if has_final:
                    final_content = content_text
                    if tool_results and not re.search(r"<observation>", content_text, re.IGNORECASE):
                        observations_block = "\n".join(f"<observation>{obs}</observation>" for obs in tool_results)
                        final_content = f"{content_text}\n{observations_block}"
                    if return_details:
                        return {"content": final_content, "tools": tool_log, "tool_results": tool_results}
                    return final_content

                # 2) 有 action 文本或调用提示，但模型未返回 tool_calls，继续请求下一轮
                has_action_tag = bool(re.search(r"<action>", content_text, re.IGNORECASE))
                if has_action_tag:
                    if self.verbose:
                        print("⚠️ 模型输出了 action/调用文本但未返回 tool_calls，继续请求下一轮。")
                    continue

                # 3) 既无工具调用也无最终答案，继续下一轮
                if self.verbose:
                    print("⚠️ 模型无 tool_calls 且无 final_answer，继续请求下一轮。")
                continue

            if tool_calls:
                # 去重并限制调用次数，避免无效重复消耗
                filtered_calls = []
                seen = set()
                for call in tool_calls:
                    key = (call.function.name, call.function.arguments or "")
                    if key in seen:
                        continue
                    seen.add(key)
                    filtered_calls.append(call)
                    if len(filtered_calls) >= MAX_TOOL_CALLS_PER_ROUND:
                        break

                # 仅打印模型调用了哪些工具及其参数，不展示工具结果
                for call in filtered_calls:
                    print(f"🔧 模型调用工具：{call.function.name}，参数：{call.function.arguments}")
                    tool_log.append(call.function.name)

                # 处理每个工具调用，并把结果加入消息
                for call in filtered_calls:
                    if stop_event and stop_event.is_set():
                        final = "对话已中断。"
                        if return_details:
                            return {"content": final, "tools": tool_log, "tool_results": tool_results}
                        return final
                    tool_msg = self.handle_tool_call(call)
                    self.messages.append(tool_msg)
                    tool_results.append(tool_msg.get("content", ""))
                    if self.verbose:
                        content_preview = tool_msg["content"]
                        # 展示更长的预览，避免换乘信息被截断；如仍嫌长可再调大
                        if len(content_preview) > 2000:
                            content_preview = content_preview[:2000].rstrip() + "..."
                        print(f"📦 工具结果：{content_preview}")

                # 继续循环，再问模型
                continue

            # 没有工具调用，表示模型已给出最终答案
            if return_details:
                return {"content": msg.content, "tools": tool_log, "tool_results": tool_results}
            return msg.content


    def stream_completion(self, prompt, stop_event: Optional["threading.Event"] = None):
        """简化版流式输出（不走工具），用于前端实时显示；支持 stop_event 中断。"""
        self.messages.append({"role": "user", "content": prompt})
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
            timeout=30,
        )
        full_text = ""
        for chunk in stream:
            if stop_event and stop_event.is_set():
                break
            delta = chunk.choices[0].delta
            content_piece = delta.content or ""
            if content_piece:
                full_text += content_piece
                yield content_piece
        # 将完整 assistant 消息记录到历史
        self.messages.append({"role": "assistant", "content": full_text})
        yield None


def run_agent(query: str):
    """使用 MCP 工具的 Agent 进行对话/查询。"""
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=base_url)
    agent = Agent(client=client, verbose=True)
    return agent.get_completion(query)

