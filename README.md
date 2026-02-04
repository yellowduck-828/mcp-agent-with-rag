# MCP Agent with RAG Tool
  
本项目实现了一个 **基于 MCP（Model Context Protocol）的可扩展 Agent 系统**，  
核心目标是展示 **Agent 如何在统一协议下调度本地工具与外部工具，并完成复杂任务**。

---

## ✨ 项目特点

- 基于 **MCP 的工具调度型 Agent 架构**
- 同时支持 **本地工具（Local Tools）与远程 MCP 工具**
- RAG 作为 **可调用工具** 而非内嵌逻辑

---

## 🔧 Tool 体系说明

本项目采用 MCP（Model Context Protocol）风格的工具机制，
Agent 可以在对话过程中按需调用不同类型的工具来完成任务。

目前支持两类工具：
- **本地工具（Local Tools）**：项目内直接实现
- **远程工具（Remote MCP Tools）**：通过 MCP 协议接入第三方工具服务

### 1️⃣ 本地工具（Local Tools）

Agent 可以直接调用项目中实现的本地工具，统一放置在 `tools/` 目录下，由 MCP Server 注册后暴露给 Agent 使用。

当前已实现的本地工具有：

- `cookbook_rag.py`  
  基于本地向量索引的 RAG 检索工具（以菜谱数据为示例）。
  
  主要用于演示完整的 RAG 流程，包括：
  - 文档加载与切分
  - 向量化（Embedding）
  - 相似度检索
  - 检索结果返回给 Agent 作为上下文

- `web_search.py`  
  基于 **Tavily API** 的网页搜索工具。  
  Agent 可以通过该工具进行实时网页搜索，用于补充最新信息或外部知识。

- `datetime.py`  
  获取当前系统时间，用于时间相关的推理或回答。

- `file.py`  
  本地文件读写工具，支持的操作包括：

  - 列出目录内容
  - 读取文本文件
  - 覆盖写入文件
  - 追加写入文件
  - 删除文件
  - 重命名 / 移动文件
  - 创建目录

  **所有文件操作仅允许在 `workspace/` 目录下进行。** 因此，即使 Agent 被错误调用或输入异常，也无法访问或修改宿主系统的任何其他文件，从而保证文件系统的安全性。

  另外，项目支持**从前端上传文件**，并将文件保存至 `workspace/` 目录中。


---

### 2️⃣ 远程工具（Remote MCP Tools）

除了本地工具外，本项目也支持通过 MCP 协议接入第三方工具服务。Agent 可以调用部署在远端的 MCP Server，例如来自 mcp.so 等平台的工具。目前示例中接入了：

- 网页搜索类工具
- 高德地图 MCP 工具

用于演示 Agent 同时使用本地工具与远程工具的能力。

若希望接入更多第三方 MCP 工具，请在 `agent.py` 的 `MultiMCPClient` servers 列表中新增/调整对应的 MCP server 配置，或改造为读取自定义配置源。

---

## 📦 数据说明

RAG 使用的数据量较大，因此 **未随代码一同提交**。

示例数据来源：
- https://github.com/Anduin2017/HowToCook

你可以自行下载并放置到：

rag/data/

随后执行索引构建流程即可。

在实际使用中可以根据自己的需求替换为任意参考文档或领域数据，例如个人笔记、项目文档、知识库或其他结构化/非结构化文本。

---

## 使用指南
### 运行环境
- Python 3.10+
- 可选：Node/npm（npx，用于 `@amap/amap-maps-mcp-server`）、uv/uvx（可选加速）、`faiss-cpu`（向量检索加速）

### 安装依赖
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
# 可选 FAISS 加速
pip install faiss-cpu
```

### 环境变量（根目录 `.env` 示例）
```env
DEEPSEEK_API_KEY=你的key          # 必填
TAVILY_API_KEY=你的key            # 必填
DEEPSEEK_API_BASE=https://api.deepseek.com/v1   # 可选，自定义网关
AMAP_MAPS_API_KEY=你的高德key      # 可选，用于 amap mcp server
```

### 运行步骤
1. 创建/激活虚拟环境并安装依赖。
2. 配置 `.env`。
3. 构建菜谱 RAG 索引（首次或数据更新后）：
   ```bash
   python -m rag.index_construction
   ```
   产物：`rag/index/index.json`（必备），可选 `faiss.index`、`meta.json`。
   系统会优先使用 FAISS 索引查询，若 FAISS 文件缺失或损坏，将自动回退使用 `index.json` 做纯 Python 检索。

5. 运行方式（二选一）：
   - 命令行对话：`python main.py`
   - API 服务：`uvicorn backend.server:app --reload --port 8000`
     - 主要接口：`/chat`、`/chat/stream`、`/chat/session/{id}/cancel`
     - 沙箱接口：`/workspace/list`、`/workspace/upload`

### 常见问题
- 检索为空：先执行索引构建；确认 `rag/data` 存在。
- 调用超时/失败：检查网络代理，可调高 `agent.py` 的 `tool_call_timeout`。
- 路径错误：文件操作仅限 `workspace/`，越界会被拒绝。
- amap 工具不可用：需 npm/npx 并配置 `AMAP_MAPS_API_KEY`。

### 其他
- Agent 默认连接本地 MCP server、`mcp-server-fetch`（uvx）和 `@amap/amap-maps-mcp-server`（npx）；可在 `agent.py` 中调整。
- 需要流式输出可用 `/chat/stream` 或 `Agent.stream_completion`。
- 日志/调试：将 `Agent(verbose=True)` 以打印工具调用与结果裁剪预览。


