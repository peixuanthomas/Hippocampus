# Hippocampus

Hippocampus 是一个完全使用 Rust 实现的本地 Ollama 会话客户端。无参数启动时进入 Ratatui TUI；`serve` 提供本地 Web UI；`ask` 子命令适合脚本和其他程序进行单次调用。

项目保存每轮原始输入、模型正文、thinking、权威 token usage、知识证据、联网工具步骤和上下文裁剪轨迹。thinking 只用于当前工具循环的展示与审计，绝不会重新注入未来对话轮次。当前会话格式为 `schema_version=3`；v1/v2 会话可直接读取并在下次保存时迁移，旧会话的 AI 名称固定迁移为 `LLM`。

会话 JSON、知识快照和原始文档是事实来源。SQLite FTS5 仅是可删除、可重建的派生索引；系统不使用生成摘要、embedding 或图数据库替代原文。

## 构建

需要 Rust 2024 edition 工具链和已经运行的 Ollama：

```bash
ollama serve
ollama pull qwen3.5:9b
cargo build --release
```

本仓库交付的可执行文件位于 `build/hippocampus`。

## 配置

仓库根目录的 [`config.toml`](config.toml) 是默认配置。未传 `--config` 时，程序可选读取当前工作目录下的 `config.toml`；当前目录没有配置时使用安全回退：名称为 `LLM`、联网和自动知识同步关闭。显式传入的文件不存在或配置含未知字段、重复来源 ID、空名称或越界预算时会直接报错。

```toml
ai_name = "hippocampus"
system_prompt = """
你是一个乐于助人的AI助手，你的任务是解决用户的问题或者与用户对话。
"""

[web_search]
enabled = true
max_results = 5
max_tool_rounds = 4
max_tool_calls = 8
max_injected_chars_per_fetch = 12000

[knowledge]
auto_sync = true
candidate_limit = 64
max_selected = 4
evidence_char_budget = 3200
```

所有命令都支持全局配置路径，例如：

```bash
./build/hippocampus --config ./config.toml new
```

知识源中的相对文件路径以配置文件目录为基准。`--system-prompt` 和 `--system-prompt-file` 对新会话及无状态 `ask` 优先于配置。AI 名称和 system prompt 写入会话后冻结，恢复旧会话不会被新配置改名或改写提示词。

## 联网搜索

联网只对有会话的请求开放；模型可在有界循环中自主调用 `web_search` 和 `web_fetch`。无会话 `ask` 始终是纯模型调用，不提供搜索或知识检索。

优先设置环境变量，由程序直连 Ollama Cloud：

```bash
export OLLAMA_API_KEY="..."
```

未设置密钥时，程序通过本地 Ollama 的登录状态访问代理接口：

```bash
ollama signin
```

API key 不会写入配置、日志或会话。每轮最多执行配置的工具轮次与调用数；搜索失败、认证失败、超时或达到上限后，程序只再执行一次禁用工具的尽力回答，并明确显示“未完成实时核验”。抓取限制为 HTTP(S)，会拒绝 localhost、私网 IP、私网 DNS 解析结果及带凭据 URL，单次完整响应最多 1 MiB。

完整工具响应和截断后的模型注入内容都会进入对应轮次的 append-only trace。CLI、TUI 与 Web UI 的来源列表由程序从真实 trace 生成，而不是采信模型自行声称的来源；模型正文中的未返回 URL 会被移除。

接口行为基于 Ollama 官方 [Web Search/Fetch API](https://docs.ollama.com/capabilities/web-search) 与 [Tool Calling](https://docs.ollama.com/capabilities/tool-calling)。

## TUI

直接运行会创建一个默认会话并进入 TUI：

```bash
./build/hippocampus
./build/hippocampus new --model qwen3.5:9b --no-think
./build/hippocampus resume 20260811-abcdef12
./build/hippocampus --sessions-dir ./sessions resume 20260811-abcdef12
```

界面顶部显示模型、thinking、上下文与会话状态；中间是对话；底部是可编辑的多行输入框。

- `Enter`：发送
- `Ctrl+J`、`Shift+Enter` 或 `Alt+Enter`：换行
- `↑` / `↓`：浏览输入历史
- 鼠标滚轮、`PageUp` / `PageDown`：滚动对话（输入或发送后自动回到最新消息）
- `Ctrl+C`：中断生成并保存已收到内容；空闲时退出
- `/list`：列出可切换的会话
- `/session <id>`：按完整 ID 或唯一前缀切换会话
- `/debug`、`/debug on|off`：查看或切换本次 TUI 的上下文组装调试输出（默认关闭）
- `/budget`、`/think on|off`、`/save`、`/help`、`/exit`

上下文达到 90% 警戒线时，TUI 会要求明确选择：裁剪最旧的完整轮次后继续，或暂停会话。裁剪只改变后续请求的活动起点，不删除原始记录。

## Web UI

`serve` 会保持 HTTP 服务运行。默认只监听本机 `127.0.0.1:31415`，然后在浏览器打开显示的地址：

```bash
./build/hippocampus serve
./build/hippocampus serve --session 20260811-abcdef12
./build/hippocampus serve --port 8080 --no-think
```

不传 `--session` 时会创建新会话；传入后会继续指定会话。页面包含与 TUI 相近的顶部状态栏、对话区、多行输入框、thinking 开关、原子保存和停止生成按钮。流式正文实时出现，完成后 Markdown 会渲染成标题、列表、表格、引用、代码块和链接等富文本；原始 HTML 与危险内容会在 Rust 服务端清洗。

网页端同样保留上下文临界决策：达到 90% 后会弹出“裁剪并继续”或“暂停会话”。所有静态资源都编译进可执行文件，不依赖 CDN 或外部前端服务。

如需保持在 shell 后台运行，可以使用操作系统自己的进程管理方式，例如：

```bash
./build/hippocampus serve >hippocampus-web.log 2>&1 &
```

默认回环地址没有跨设备访问能力。`--bind 0.0.0.0` 可以开放局域网访问，但当前版本没有用户认证，不应暴露到不可信网络或公网。

## `ask` 单次调用

不传 `--session` 时是无状态调用：不会读取历史、不会创建会话文件、不会自动同步知识，也不会向模型提供联网工具，只发送 system prompt、独立身份指令和当前问题。

```bash
./build/hippocampus ask "只回答一个词：天空是什么颜色？"
./build/hippocampus ask --no-think --system-prompt "简洁回答" "你好"
./build/hippocampus ask --json "你好"
```

传入 `--session` 时才会加载该会话上下文，并把新一轮原子保存回同一个会话：

```bash
./build/hippocampus ask --session 20260811-abcdef12 "继续刚才的话题"
./build/hippocampus ask --session 20260811-abcdef12 --trim "继续"
```

当有会话的上下文达到警戒线时，非交互调用默认暂停并报错；`--trim` 表示授权自动丢弃最旧完整轮次后继续。

## 可更新知识库

在 `config.toml` 中添加稳定资料来源：

```toml
[[knowledge.sources]]
id = "local-notes"
kind = "path"
location = "./knowledge"

[[knowledge.sources]]
id = "project-docs"
kind = "url"
location = "https://example.com/docs.txt"
```

`path` 支持单个 UTF-8 `.txt`/`.md` 文件或递归目录，忽略符号链接和其他格式；`url` 通过 Ollama Web Fetch 获取标题、正文和链接。管理命令如下：

```bash
./build/hippocampus knowledge sync
./build/hippocampus knowledge list
./build/hippocampus knowledge search "查询词"
./build/hippocampus knowledge rebuild
```

当 `knowledge.auto_sync=true` 时，启动 TUI、Web、新建或恢复会话以及带 `--session` 的 `ask` 前都会同步；`list`、`show` 与无状态 `ask` 不触发同步。URL 更新失败时继续使用最近成功 revision，并在回答和界面中显示过期警告。

每次内容变化会在 `<sessions-dir>/.knowledge/snapshots/` 写入不可变 JSON revision；相同内容不重复写入，配置中移除的来源不再检索，但历史 revision 保留。当前来源的最新成功 revision 会派生到 `.knowledge/index.sqlite3`。索引使用 Jieba 字段、CJK 2/3-gram、完整文档以及 240 Unicode 字符、40 字符重叠的 passage，并在返回前核对精确 span 与 SHA-256。

## 会话管理

```bash
./build/hippocampus list
./build/hippocampus show 20260811-abcdef12
./build/hippocampus show 20260811-abcdef12 --json
```

## 事件检索与溯源 API

会话 JSON 始终是唯一事实来源。每次 JSON 原子保存成功后，`SessionStore` 会同步更新同目录下的 `.hippocampus-index.sqlite3`；该 SQLite 文件只是一层可删除、可重建的派生索引。

会话长期回忆同样使用完整消息及 240 字符重叠 passage，并以 Jieba 与 CJK n-gram 做确定性词法检索。`RetrievalTrace` 记录查询词、BM25 原始排名、排除原因、最终原文 span 和独立预算。`answer_context` 同时返回身份指令、知识证据与完整 `WebTrace`，因此可以重建最终模型请求和每个工具步骤。

Rust 调用方可以通过 `SessionStore::retrieval()` 获取 `RetrievalStore`：

```rust,no_run
let store = hippocampus::SessionStore::new("sessions")?;
let retrieval = store.retrieval();

let events = retrieval.replay_session("20260811-abcdef12")?;
let event = retrieval.get_event(&events[1].id)?;
let fragment = retrieval.resolve_span(&hippocampus::SourceSpan {
    event_id: event.id.clone(),
    start_char: 0,
    end_char: event.content.chars().count(),
})?;
let trace = retrieval.answer_context(&events.last().unwrap().id)?;

// SQLite 被删除、清空或修改后，从全部会话 JSON 重建。
retrieval.rebuild()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

事件 ID 与内容无关并可确定性重建；原文片段使用从 0 开始、右开区间的 Unicode 字符偏移。thinking 不进入检索事件。assistant 的 `token_count` 只在最终模型 usage 可证明时记录权威生成数，user 与 system 保持未知。所有读取都会核对原始 JSON 的 SHA-256，源文件变化时拒绝返回过期结果。

如果 JSON 已经安全落盘、但 SQLite 同步失败，保存会返回 `IndexSyncAfterSourceCommit`。此时原始会话没有丢失，可重试保存或调用 `RetrievalStore::rebuild()` 恢复派生层。

知识 SQLite 可通过 `knowledge rebuild` 从状态与不可变快照重建；会话 SQLite 可通过 `RetrievalStore::rebuild()` 从全部会话 JSON 重建。所有读取都会校验原始内容、Unicode span、revision 与派生行，发现索引被修改时拒绝使用。

默认上下文窗口为 32768，输出预留 4096，安全余量 512，输入预算因此为 28160。80% 起执行精确 probe，90% 起要求裁剪决策。流中的 logprob 数只用于实时显示；最终 `prompt_eval_count` 和 `eval_count` 才是权威值。没有收到最终事件时，未知计数保持为 `null`，不会用 probe 或估算值冒充。
