# Hippocampus

Hippocampus 是一个完全使用 Rust 实现的本地 Ollama 会话客户端。无参数启动时进入 Ratatui TUI；`serve` 提供本地 Web UI；`ask` 子命令适合脚本和其他程序进行单次调用。

项目保存每轮原始输入、模型正文、thinking、权威 token usage 和上下文裁剪轨迹。thinking 只用于当前轮展示和审计，绝不会重新注入后续模型上下文。旧 Python 版本产生的 `schema_version=1` 会话 JSON 可以直接继续使用。

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

默认上下文窗口为 32768，输出预留 4096，安全余量 512，输入预算因此为 28160。80% 起执行精确 probe，90% 起要求裁剪决策。流中的 logprob 数只用于实时显示；最终 `prompt_eval_count` 和 `eval_count` 才是权威值。没有收到最终事件时，未知计数保持为 `null`，不会用 probe 或估算值冒充。

## 验证

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```
