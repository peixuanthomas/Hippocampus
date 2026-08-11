# Hippocampus

## Long-term lexical evidence (MVP)

Hippocampus indexes immutable user/assistant originals in SQLite FTS5 after each source-session
commit.  The derived index has a full message document and, above 240 Unicode characters,
overlapping 240-character fragments (40-character overlap).  Jieba word fields and CJK 2/3
character n-grams provide deterministic keyword recall; embeddings and generated summaries are
not used.

`Session::retrieval` controls candidate, core-evidence, and independent expansion budgets.  The
public `RetrievalStore::keyword_recall` returns exact spans plus a `RetrievalTrace` recording
quoted query terms, lower-is-better SQLite BM25 scores, raw ranks, every exclusion, and selected
evidence.  Evidence is inserted as an independent contiguous original-role region between the
original system message and normal recent history.  Completed answer context exposes the same
trace via `answer_context`.

Diagnosis is mechanical: a fact absent from bounded candidates is a retrieval failure; a
candidate marked unselected with its reason is a selection failure; selected exact evidence in
the persisted answer context followed by a wrong answer is a generation failure.

Hippocampus 是一个完全使用 Rust 实现的本地 Ollama 会话客户端。无参数启动时进入 Ratatui TUI；`serve` 提供本地 Web UI；`ask` 子命令适合脚本和其他程序进行单次调用。

项目保存每轮原始输入、模型正文、thinking、权威 token usage 和上下文裁剪轨迹。thinking 只用于当前轮展示和审计，绝不会重新注入后续模型上下文。当前会话格式为 `schema_version=2`；旧 Python 版本产生的 v1 JSON 可以直接读取，并在下一次保存时迁移，无法从旧格式证明的历史溯源会明确标记为 `legacy_inferred`。

## 构建

需要 Rust 2024 edition 工具链和已经运行的 Ollama：

```bash
ollama serve
ollama pull qwen3.5:9b
cargo build --release
```

本仓库交付的可执行文件位于 `build/hippocampus`。

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
- `PageUp` / `PageDown`：滚动对话
- `Ctrl+C`：中断生成并保存已收到内容；空闲时退出
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

不传 `--session` 时是无状态调用：不会读取历史、不会创建会话文件，只向 Ollama 发送 `system prompt + 当前问题`。

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

## 会话管理

```bash
./build/hippocampus list
./build/hippocampus show 20260811-abcdef12
./build/hippocampus show 20260811-abcdef12 --json
```

## 事件检索与溯源 API

会话 JSON 始终是唯一事实来源。每次 JSON 原子保存成功后，`SessionStore` 会同步更新同目录下的 `.hippocampus-index.sqlite3`；该 SQLite 文件只是一层可删除、可重建的派生索引。

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

默认上下文窗口为 32768，输出预留 4096，安全余量 512，输入预算因此为 28160。80% 起执行精确 probe，90% 起要求裁剪决策。流中的 logprob 数只用于实时显示；最终 `prompt_eval_count` 和 `eval_count` 才是权威值。没有收到最终事件时，未知计数保持为 `null`，不会用 probe 或估算值冒充。

## 验证

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```
