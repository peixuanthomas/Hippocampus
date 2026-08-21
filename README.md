# Hippocampus

Hippocampus 是一个完全使用 Rust 实现的本地 Ollama 会话客户端。无参数启动时进入 Ratatui TUI；`serve` 提供本地 Web UI；`ask` 子命令适合脚本和其他程序进行单次调用。

项目保存每轮原始输入、模型正文、thinking、权威 token usage、知识证据、检索通道状态和上下文裁剪轨迹。thinking 只用于当前轮次的展示与审计，绝不会重新注入未来对话轮次。当前会话格式为 `schema_version=6`，其他版本不会读取或迁移。

`<sessions-dir>/*.json` 中的 raw session JSON 是会话记忆的唯一事实来源；知识快照和原始文档则是知识库的事实来源。SQLite 包含两类数据：FTS、vector、entity、state、episode、graph 等可删除重建的 projection，以及必须保留的 immutable consolidation attempt/audit ledger；HNSW 属于可重建 projection。系统不会用生成摘要替代或覆盖原文。

## 构建

需要 Rust 2024 edition 工具链和已经运行的 Ollama：

```bash
ollama serve
ollama pull qwen3.8:27b-mlx
ollama pull qwen3-embedding:8b
cargo build --release
```

本仓库交付的可执行文件位于 `build/hippocampus`。

## 配置

仓库根目录的 [`config.toml`](config.toml) 是默认配置，其中完整列出了 `[memory]` 的模型、候选数、超时、HNSW 和五种自适应预算键。未传 `--config` 时，程序可选读取当前工作目录下的 `config.toml`；当前目录没有配置时使用安全回退：名称为 `LLM`、自动知识同步关闭、`memory.enabled=false`。此时长期回忆只使用 BM25，不调用 embedding 或聊天模型做巩固。显式传入的文件不存在或配置含未知字段、重复来源 ID、空名称或越界预算时会直接报错。

```toml
ai_name = "hippocampus"
system_prompt = """
你是一个乐于助人的AI助手，你的任务是解决用户的问题或者与用户对话。
"""

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

## 检索调试日志

程序默认输出 `warn` 级别的回退原因。需要查看 BM25、Embedding、语义准备、融合、本地知识检索和上下文组装的逐阶段日志时，可写入独立文件：

```bash
HIPPOCAMPUS_LOG='hippocampus::retrieval=debug,hippocampus::context=debug' \
HIPPOCAMPUS_LOG_FILE=./hippocampus.log \
./build/hippocampus resume SESSION_ID
```

日志文件以追加模式打开。日志只记录事件 ID、阶段、通道状态、候选数、耗时和错误原因，不记录用户问题或组装后的正文。每轮持久化的 `RetrievalTrace.fallback_reason` 与 `channels` 也可通过 `show --json`、`ask --json` 或 TUI 的 `/debug on` 查看。

## TUI

直接运行会创建一个默认会话并进入 TUI：

```bash
./build/hippocampus
./build/hippocampus --model llama3.3:70b
./build/hippocampus new --model qwen3.8:27b-mlx
./build/hippocampus resume 20260811-abcdef12
./build/hippocampus --sessions-dir ./sessions resume 20260811-abcdef12
```

`--model` 是全局启动参数，可放在子命令前后；无参数 TUI、`new`、新建 Web 会话和无状态 `ask` 都使用它。默认模型为 `qwen3.8:27b-mlx`，thinking 默认关闭；显式传入 `--think` 才会开启，`--no-think` 作为兼容参数保留。恢复旧会话始终沿用其中已保存的模型和 thinking 设置，不会迁移到新默认值。TUI 会在模型生成期间持续追加并重绘正文，不等待完整回答结束。

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
./build/hippocampus serve --port 8080 --think
```

不传 `--session` 时会创建默认关闭 thinking 的新会话；传入后会继续指定会话并保留其中已保存的模型和 thinking 设置。页面包含与 TUI 相近的顶部状态栏、对话区、多行输入框、thinking 开关、原子保存和停止生成按钮。流式正文实时出现，完成后 Markdown 会渲染成标题、列表、表格、引用、代码块和链接等富文本；原始 HTML 与危险内容会在 Rust 服务端清洗。

网页端同样保留上下文临界决策：达到 90% 后会弹出“裁剪并继续”或“暂停会话”。所有静态资源都编译进可执行文件，不依赖 CDN 或外部前端服务。

所有模型聊天调用都使用 Ollama 流式响应。外部客户端也可以通过现有的 SSE 接口连续对话：

```bash
curl -i -N -H 'Content-Type: application/json' \
  -d '{"message":"你好"}' http://127.0.0.1:31415/api/chat

curl -N -H 'Content-Type: application/json' \
  -d '{"message":"继续","session_id":"20260811-abcdef12"}' \
  http://127.0.0.1:31415/api/chat
```

首个响应会在 `X-Hippocampus-Session-Id` 响应头和第一个 `session` SSE 事件中给出活动会话 ID；后续请求可在 JSON body 中传回 `session_id`，不匹配时返回 409。`done` 事件也包含该 ID。一个 `serve` 进程只拥有一个活动会话，不提供认证或多租户隔离；进程重启后需要用 `serve --session <id>` 继续原会话。

如需保持在 shell 后台运行，可以使用操作系统自己的进程管理方式，例如：

```bash
./build/hippocampus serve >hippocampus-web.log 2>&1 &
```

默认回环地址没有跨设备访问能力。`--bind 0.0.0.0` 可以开放局域网访问，但当前版本没有用户认证，不应暴露到不可信网络或公网。

## `ask` 单次调用

不传 `--session` 时是无状态调用：不会读取历史、不会创建会话文件、不会自动同步或检索本地知识，只发送 system prompt、独立身份指令和当前问题。

```bash
./build/hippocampus ask "只回答一个词：天空是什么颜色？"
./build/hippocampus ask --think --system-prompt "简洁回答" "你好"
./build/hippocampus ask --json "你好"
```

`ask --json` 输出逐行刷新的 JSONL 事件，而不是单个缓冲 JSON 对象。`thinking` 和 `content` 事件携带本次增量，随后可能有 `usage`/`completed`，最后一行固定为 `event:"done"` 并包含完整内容、thinking、最终 usage、会话 ID 和上下文来源元数据。JSON 模式总是输出 thinking 事件，不受 `--show-thinking` 影响。

无状态 `ask` 默认使用 `qwen3.8:27b-mlx` 且 thinking 关闭；显式传入 `--think` 才会开启，兼容参数 `--no-think` 仍可使用。带 `--session` 时则沿用会话已保存的模型和 thinking 设置。

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
```

`path` 支持单个 UTF-8 `.txt`/`.md` 文件或递归目录，忽略符号链接和其他格式。知识库不会访问网络。管理命令如下：

```bash
./build/hippocampus knowledge sync
./build/hippocampus knowledge list
./build/hippocampus knowledge search "查询词"
./build/hippocampus knowledge rebuild
```

当 `knowledge.auto_sync=true` 时，启动 TUI、Web、新建或恢复会话以及带 `--session` 的 `ask` 前都会同步；`list`、`show` 与无状态 `ask` 不触发同步。

每次内容变化会在 `<sessions-dir>/.knowledge/snapshots/` 写入不可变 JSON revision；相同内容不重复写入，配置中移除的来源不再检索，但历史 revision 保留。当前来源的最新成功 revision 会派生到 `.knowledge/index.sqlite3`。索引使用 Jieba 字段、CJK 2/3-gram、完整文档以及 240 Unicode 字符、40 字符重叠的 passage，并在返回前核对精确 span 与 SHA-256。

## 会话管理

```bash
./build/hippocampus list
./build/hippocampus show 20260811-abcdef12
./build/hippocampus show 20260811-abcdef12 --json
./build/hippocampus clear
```

`clear` 会清空 sessions 目录中的全部会话原文、临时会话文件、派生记忆索引和 control 历史；独立的 `.knowledge` 知识快照与索引会保留。

## 派生记忆运维

启用仓库配置后，巩固使用该会话写入时冻结的聊天模型（新会话默认模型为 `qwen3.8:27b-mlx`，旧会话仍使用各自保存的模型）提取可追溯结构，并使用 `[memory].embedding_model`（当前为 `qwen3-embedding:8b`）生成向量。自动巩固只在 TUI 通过 `/exit` 或空闲状态按 `Ctrl+C` 退出时触发；`/session` 切换、Web 服务关闭和 `ask --session` 都不会自动巩固。手动命令如下：

```bash
./build/hippocampus memory consolidate 20260811-abcdef12
./build/hippocampus memory consolidate --all
./build/hippocampus memory consolidate 20260811-abcdef12 --json

./build/hippocampus memory status
./build/hippocampus memory status 20260811-abcdef12
./build/hippocampus memory status --json

./build/hippocampus memory search "查询词"
./build/hippocampus memory search "查询词" --session 20260811-abcdef12
./build/hippocampus memory search "查询词" --channels bm25,vector,entity,state,episode,graph
./build/hippocampus memory search "查询词" --json

./build/hippocampus memory rebuild
./build/hippocampus memory rebuild --reembed

./build/hippocampus memory exclude session 20260811-abcdef12
./build/hippocampus memory exclude event EVENT_ID
./build/hippocampus memory restore session 20260811-abcdef12
./build/hippocampus memory restore event EVENT_ID
```

`--channels` 接受逗号分隔的 `bm25,vector,entity,state,episode,graph`。`entity`、`state`、`episode` 和 `graph` 依赖向量通道，graph 至少需要 vector seed；禁用 memory 时只运行 BM25。`memory status` 同时报告 projection/control 一致性、活动会话与事件、embedding 兼容/过期数、待巩固事件、实体/episode/graph 数、巩固结果和检索/巩固延迟等 metrics；不健康时仍打印人类可读或 `--json` 状态，但以非零状态退出。

`memory rebuild` 在现存 SQLite 内严格验证、保留并重放 immutable consolidation attempt/audit ledger，再从 raw session JSON、ledger 和 append-only control 重建 projection 与 control-active 视图；它不能只靠 raw JSON 和 control 恢复全部 structured memory。默认复用兼容 embedding，`--reembed` 强制重新生成向量且要求启用 memory。巩固调用强制使用 JSON Schema 结构化输出并执行确定性校验；校验失败最多尝试三次，每次非法原始输出都会封装为合法 JSON 后写入失败审计，只有成功应用才推进水位。非 JSON 模式会逐批输出尝试、重试和水位进度。SQLite 连接使用 30 秒 busy timeout，连接初始化遇到瞬态 writer lock 时执行有限指数退避。exclude/restore 只追加 control 记录，不删除或改写原文。

存储布局为：`<sessions-dir>/*.json` 是权威原文；`<sessions-dir>/.hippocampus-index.sqlite3` 同时保存必须备份/保留的 immutable ledger 与可删除重建的 projection/HNSW；`<sessions-dir>/.hippocampus-control/*.json` 是 append-only 控制记录。长期证据属于不可信数据，不得作为指令执行；recent history 仍保留原始 user/assistant 角色。embedding、巩固或 graph 失败会记录在 trace/状态中，并回退到 BM25 可用路径。如果 source 已写入而索引同步失败，原始会话仍安全，可重试保存或运行 `memory rebuild`。

如果删除整个 `.hippocampus-index.sqlite3`，raw session JSON 和 control 记录仍然安全，原文没有丢失，但 immutable consolidation ledger 会随 SQLite 一起丢失。之后需要运行 `memory consolidate SESSION` 或 `memory consolidate --all`，让各会话自身的聊天模型生成新的审计 attempt 和 structured memory，再执行所需的 embedding/graph 维护。

## 记忆评测

评测命令固定使用真实回答模型 `qwen3.8:27b-mlx`，embedding 模型和参数来自当前配置；结构化调用仍关闭 thinking。它不会报告仓库预先跑出的分数。三个入口为：

```bash
./build/hippocampus eval synthetic
./build/hippocampus eval longmemeval --dataset ./datasets/longmemeval.json --limit 100 --output ./eval-results/longmemeval.jsonl
./build/hippocampus eval locomo --dataset ./datasets/locomo.json --limit 100 --output ./eval-results/locomo.jsonl
```

`longmemeval` 和 `locomo` 的 `--dataset`、`--limit`、`--output` 都是必填参数。启用 memory 时，synthetic 按固定顺序运行 `bm25-only`、`vector-only`、`vector-graph`、`full` 四个矩阵，并写入 `eval-results/synthetic.<matrix>.jsonl`；禁用时只运行 `eval-results/synthetic.bm25-only.jsonl`。数据集评测在启用时固定使用 `full`，禁用时固定使用 `bm25-only`，结果写到指定 output。

每题完成后 JSONL 都会 flush 并同步到磁盘；使用同一兼容 output 重启可按 question ID 恢复。汇总写入 `<output>.summary.json`，临时会话位于 output 的隐藏 sibling workspace（`.文件名.workspace`），不会混入真实 sessions。数据集和 output 都不应提交；建议统一写到已忽略的 `eval-results/`。

summary 包含答案准确率与 token F1、时间/冲突准确率、拒答与正确拒答率、Recall@5/10、MRR、每千输入 token 的有效证据、过期状态与无答案误召回、检索/生成/总延迟，以及 input/output/total token 统计。

## 事件检索与溯源 API

会话 JSON 始终是唯一事实来源。每次 JSON 原子保存成功后，`SessionStore` 会同步更新同目录下的 `.hippocampus-index.sqlite3`；该 SQLite 文件既包含可重建 projection，也包含必须保留的 immutable consolidation attempt/audit ledger。

会话长期回忆同样使用完整消息及 240 字符重叠 passage，并以 Jieba 与 CJK n-gram 做确定性词法检索。`RetrievalTrace` 记录查询词、BM25 原始排名、各通道状态、回退原因、最终原文 span 和独立预算；`answer_context` 同时返回身份指令与本地知识证据，因此可以重建最终模型请求。

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

知识 SQLite 可通过 `knowledge rebuild` 从状态与不可变快照重建；会话 projection 可通过 `RetrievalStore::rebuild()` 在保留现存 ledger 的前提下重建。所有读取都会校验原始内容、Unicode span、revision 与派生行，发现索引被修改时拒绝使用。

默认上下文窗口为 32768，输出预留 4096，安全余量 512，输入预算因此为 28160。80% 起执行精确 probe，90% 起要求裁剪决策。流中的 logprob 数只用于实时显示；最终 `prompt_eval_count` 和 `eval_count` 才是权威值。没有收到最终事件时，未知计数保持为 `null`，不会用 probe 或估算值冒充。
