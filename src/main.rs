use std::io::{self, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use hippocampus::engine::PreparationStatus;
use hippocampus::model::{
    BudgetConfig, ChatEventKind, ChatMessage, DEFAULT_SYSTEM_PROMPT, Session, TokenUsage,
};
use hippocampus::ollama::{ChatBackend, ChatRequest};
use hippocampus::{ChatEngine, LimitAction, OllamaClient, SessionStore};
use serde_json::json;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Parser)]
#[command(
    name = "hippocampus",
    version,
    about = "本地 Ollama 会话客户端：无参数进入 TUI，ask 用于脚本调用"
)]
struct Cli {
    #[arg(long, global = true, default_value = "sessions")]
    sessions_dir: PathBuf,
    #[arg(long, global = true, default_value = "http://127.0.0.1:11434")]
    host: String,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 创建会话并进入 TUI
    New(NewArgs),
    /// 恢复已有会话并进入 TUI（支持唯一前缀）
    Resume { identifier: String },
    /// 列出保存的会话
    List,
    /// 只读查看会话
    Show {
        identifier: String,
        #[arg(long)]
        json: bool,
    },
    /// 单次调用；带 --session 使用会话上下文，否则只发送 system prompt + 当前问题
    Ask(AskArgs),
    /// 启动本地 Web UI 并保持服务运行
    Serve(ServeArgs),
}

#[derive(Debug, Clone, Args)]
struct NewArgs {
    #[arg(long, default_value = "qwen3.5:9b")]
    model: String,
    #[arg(long, default_value_t = 32_768)]
    context_window: u64,
    #[arg(long, default_value_t = 4_096)]
    max_output_tokens: u64,
    #[arg(long, default_value_t = 512)]
    safety_margin_tokens: u64,
    #[arg(long, conflicts_with = "no_think")]
    think: bool,
    #[arg(long = "no-think", conflicts_with = "think")]
    no_think: bool,
    #[arg(long, conflicts_with = "system_prompt_file")]
    system_prompt: Option<String>,
    #[arg(long, conflicts_with = "system_prompt")]
    system_prompt_file: Option<PathBuf>,
}

impl Default for NewArgs {
    fn default() -> Self {
        Self {
            model: "qwen3.5:9b".into(),
            context_window: 32_768,
            max_output_tokens: 4_096,
            safety_margin_tokens: 512,
            think: true,
            no_think: false,
            system_prompt: None,
            system_prompt_file: None,
        }
    }
}

impl NewArgs {
    fn budget(&self) -> BudgetConfig {
        BudgetConfig {
            context_window: self.context_window,
            max_output_tokens: self.max_output_tokens,
            safety_margin_tokens: self.safety_margin_tokens,
            ..BudgetConfig::default()
        }
    }

    fn read_prompt(&self) -> Result<Option<String>> {
        if let Some(path) = &self.system_prompt_file {
            return Ok(Some(std::fs::read_to_string(path).with_context(|| {
                format!("无法读取系统提示文件 {}", path.display())
            })?));
        }
        Ok(self.system_prompt.clone())
    }

    fn thinking_enabled(&self) -> bool {
        self.think || !self.no_think
    }
}

#[derive(Debug, Args)]
struct AskArgs {
    /// 当前问题
    prompt: String,
    /// 使用该会话的历史上下文；不传则无历史且不创建会话
    #[arg(long)]
    session: Option<String>,
    #[arg(long, default_value = "qwen3.5:9b")]
    model: String,
    #[arg(long, default_value_t = 32_768)]
    context_window: u64,
    #[arg(long, default_value_t = 4_096)]
    max_output_tokens: u64,
    #[arg(long, default_value_t = 512)]
    safety_margin_tokens: u64,
    #[arg(long, conflicts_with = "no_think")]
    think: bool,
    #[arg(long = "no-think", conflicts_with = "think")]
    no_think: bool,
    #[arg(long, conflicts_with = "system_prompt_file")]
    system_prompt: Option<String>,
    #[arg(long, conflicts_with = "system_prompt")]
    system_prompt_file: Option<PathBuf>,
    /// 会话上下文达到警戒线时自动裁剪最旧完整轮次
    #[arg(long)]
    trim: bool,
    /// 将 thinking 流输出到 stderr
    #[arg(long)]
    show_thinking: bool,
    /// 输出单个 JSON 对象，不实时打印正文
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// 加载已有会话；不传则按下列参数创建新会话
    #[arg(long)]
    session: Option<String>,
    /// HTTP 监听地址；默认仅本机可访问
    #[arg(long, default_value = "127.0.0.1")]
    bind: IpAddr,
    /// HTTP 监听端口
    #[arg(long, default_value_t = 31_415)]
    port: u16,
    #[command(flatten)]
    new: NewArgs,
}

impl AskArgs {
    fn thinking_enabled(&self) -> bool {
        self.think || !self.no_think
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("错误：{error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let store = SessionStore::new(&cli.sessions_dir)?;
    match cli.command {
        None => run_new_tui(store, &cli.host, NewArgs::default()).await,
        Some(Command::New(args)) => run_new_tui(store, &cli.host, args).await,
        Some(Command::Resume { identifier }) => run_resume_tui(store, &identifier).await,
        Some(Command::List) => list_sessions(&store),
        Some(Command::Show { identifier, json }) => show_session(&store, &identifier, json),
        Some(Command::Ask(args)) => run_ask(store, &cli.host, args).await,
        Some(Command::Serve(args)) => run_serve(store, &cli.host, args).await,
    }
}

async fn run_serve(store: SessionStore, host: &str, args: ServeArgs) -> Result<()> {
    let mut session = if let Some(identifier) = &args.session {
        let mut session = store.load(identifier)?;
        store.reopen(&mut session)?;
        session
    } else {
        let prompt = args.new.read_prompt()?;
        store.create(
            &args.new.model,
            host,
            prompt.as_deref(),
            args.new.budget(),
            args.new.thinking_enabled(),
        )?
    };
    let client = OllamaClient::new(&session.ollama_host)?;
    let info = client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::new(store.clone(), client);
    let address = SocketAddr::new(args.bind, args.port);
    hippocampus::web::serve(engine, session.clone(), info, address).await?;
    session = store.load(&session.id)?;
    store.save(&mut session)?;
    Ok(())
}

async fn run_new_tui(store: SessionStore, host: &str, args: NewArgs) -> Result<()> {
    let prompt = args.read_prompt()?;
    let mut session = store.create(
        &args.model,
        host,
        prompt.as_deref(),
        args.budget(),
        args.thinking_enabled(),
    )?;
    let client = OllamaClient::new(&session.ollama_host)?;
    let info = client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::new(store.clone(), client);
    session = hippocampus::tui::run(engine, session, info).await?;
    store.save(&mut session)?;
    Ok(())
}

async fn run_resume_tui(store: SessionStore, identifier: &str) -> Result<()> {
    let mut session = store.load(identifier)?;
    store.reopen(&mut session)?;
    let client = OllamaClient::new(&session.ollama_host)?;
    let info = client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::new(store.clone(), client);
    session = hippocampus::tui::run(engine, session, info).await?;
    store.save(&mut session)?;
    Ok(())
}

async fn run_ask(store: SessionStore, host: &str, args: AskArgs) -> Result<()> {
    if let Some(identifier) = args.session.clone() {
        return run_contextual_ask(store, &identifier, args).await;
    }
    run_stateless_ask(host, args).await
}

async fn run_contextual_ask(store: SessionStore, identifier: &str, args: AskArgs) -> Result<()> {
    let mut session = store.load(identifier)?;
    store.reopen(&mut session)?;
    let client = OllamaClient::new(&session.ollama_host)?;
    client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::new(store, client);
    let mut prepared = engine.prepare_turn(&mut session, args.prompt).await?;
    if prepared.needs_limit_decision() {
        if !args.trim {
            engine
                .resolve_limit(&mut session, prepared, LimitAction::EndSession)
                .await?;
            bail!("上下文达到临界阈值；如需自动裁剪后继续，请重新运行并添加 --trim");
        }
        prepared = engine
            .resolve_limit(&mut session, prepared, LimitAction::ContinueWithTrim)
            .await?;
    }
    if prepared.status == PreparationStatus::Blocked {
        bail!(prepared.message);
    }

    let cancellation = cancellation_on_ctrl_c();
    let show_thinking = args.show_thinking;
    let json_output = args.json;
    engine
        .stream_turn(&mut session, &prepared, cancellation, move |event| {
            if !json_output && event.kind == ChatEventKind::Content {
                print!("{}", event.text);
                let _ = io::stdout().flush();
            } else if !json_output && show_thinking && event.kind == ChatEventKind::Thinking {
                eprint!("{}", event.text);
                let _ = io::stderr().flush();
            }
        })
        .await?;
    if args.json {
        let turn = session
            .turns
            .last()
            .context("生成完成但会话中没有对应轮次")?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "session_id": session.id,
                "turn_id": turn.id,
                "status": turn.status,
                "thinking": turn.thinking,
                "content": turn.assistant_content,
                "usage": turn.usage,
            }))?
        );
    } else {
        println!();
        io::stdout().flush()?;
        print_usage(session.turns.last().map(|turn| turn.usage));
    }
    Ok(())
}

async fn run_stateless_ask(host: &str, args: AskArgs) -> Result<()> {
    let system_prompt = read_ask_prompt(&args)?;
    let budget = BudgetConfig {
        context_window: args.context_window,
        max_output_tokens: args.max_output_tokens,
        safety_margin_tokens: args.safety_margin_tokens,
        ..BudgetConfig::default()
    };
    budget.validate()?;
    let client = OllamaClient::new(host)?;
    client
        .check_model(&args.model, budget.context_window)
        .await?;
    let think = args.thinking_enabled();
    let mut messages = Vec::new();
    if !system_prompt.is_empty() {
        messages.push(ChatMessage {
            role: "system".into(),
            content: system_prompt,
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: args.prompt,
    });
    let request = ChatRequest {
        model: args.model,
        messages,
        think,
        num_ctx: budget.context_window,
        num_predict: budget.max_output_tokens,
    };
    let mut thinking = String::new();
    let mut content = String::new();
    let mut usage = None;
    let json_output = args.json;
    let show_thinking = args.show_thinking;
    client
        .stream_chat(
            request,
            cancellation_on_ctrl_c(),
            &mut |event| match event.kind {
                ChatEventKind::Thinking => {
                    thinking.push_str(&event.text);
                    if !json_output && show_thinking {
                        eprint!("{}", event.text);
                        let _ = io::stderr().flush();
                    }
                }
                ChatEventKind::Content => {
                    content.push_str(&event.text);
                    if !json_output {
                        print!("{}", event.text);
                        let _ = io::stdout().flush();
                    }
                }
                ChatEventKind::Completed => usage = event.usage,
                ChatEventKind::Usage => {}
            },
        )
        .await?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "session_id": null,
                "stateless": true,
                "thinking": thinking,
                "content": content,
                "usage": usage,
            }))?
        );
    } else {
        println!();
        io::stdout().flush()?;
        print_usage(usage);
    }
    Ok(())
}

fn cancellation_on_ctrl_c() -> CancellationToken {
    let token = CancellationToken::new();
    let signal_token = token.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_token.cancel();
        }
    });
    token
}

fn read_ask_prompt(args: &AskArgs) -> Result<String> {
    if let Some(path) = &args.system_prompt_file {
        return std::fs::read_to_string(path)
            .with_context(|| format!("无法读取系统提示文件 {}", path.display()));
    }
    Ok(args
        .system_prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.into()))
}

fn list_sessions(store: &SessionStore) -> Result<()> {
    let sessions = store.list_sessions()?;
    if sessions.is_empty() {
        println!("没有保存的会话。");
        return Ok(());
    }
    for session in sessions {
        println!(
            "{}｜{}｜{}｜{}｜{} 轮｜{}",
            session.id,
            session_status(&session),
            session.updated_at,
            session.model,
            session.turns.len(),
            session.title
        );
    }
    Ok(())
}

fn show_session(store: &SessionStore, identifier: &str, json_output: bool) -> Result<()> {
    let session = store.load(identifier)?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&session)?);
        return Ok(());
    }
    println!(
        "会话 {}｜{}｜{}\n模型：{}｜Ollama：{}｜thinking：{}\n系统提示：{}",
        session.id,
        session_status(&session),
        session.title,
        session.model,
        session.ollama_host,
        if session.think { "on" } else { "off" },
        session.system_prompt
    );
    let budget = &session.budget;
    println!(
        "预算：context={}, input={}, output reserve={}, safety={}\n阈值：80%={}, 90%={}；活动起点={}",
        budget.context_window,
        budget.input_budget(),
        budget.max_output_tokens,
        budget.safety_margin_tokens,
        budget.probe_threshold(),
        budget.warning_threshold(),
        session.active_context_start_index,
    );
    for (index, turn) in session.turns.iter().enumerate() {
        println!(
            "第 {} 轮 [{}] 用户：{}\n  正文：{}\n  精确：input={}, output={}, total={}\n  probe：input={}, output={}, total={}",
            index + 1,
            turn.status.as_str(),
            compact(&turn.user_content, 80),
            compact(&turn.assistant_content, 80),
            optional(turn.usage.input_tokens),
            optional(turn.usage.output_tokens),
            optional(turn.usage.total_tokens),
            optional(turn.probe_usage.input_tokens),
            optional(turn.probe_usage.output_tokens),
            optional(turn.probe_usage.total_tokens),
        );
    }
    Ok(())
}

fn session_status(session: &Session) -> &'static str {
    session.status.as_str()
}

fn compact(value: &str, limit: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let body = compact.chars().take(limit).collect::<String>();
    if compact.chars().count() > limit {
        format!("{body}…")
    } else if body.is_empty() {
        "（无）".into()
    } else {
        body
    }
}

fn optional(value: Option<u64>) -> String {
    value.map_or_else(|| "未知".into(), |value| value.to_string())
}

fn print_usage(usage: Option<TokenUsage>) {
    if let Some(usage) = usage {
        eprintln!(
            "token：input={}, output={}, total={}",
            optional(usage.input_tokens),
            optional(usage.output_tokens),
            optional(usage.total_tokens)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ask_session_is_optional_and_no_think_is_supported() {
        let stateless = Cli::try_parse_from(["hippocampus", "ask", "hello"]).unwrap();
        let Some(Command::Ask(args)) = stateless.command else {
            panic!("expected ask command");
        };
        assert!(args.session.is_none());
        assert!(args.thinking_enabled());

        let contextual = Cli::try_parse_from([
            "hippocampus",
            "ask",
            "--session",
            "20260811-abc",
            "--no-think",
            "hello",
        ])
        .unwrap();
        let Some(Command::Ask(args)) = contextual.command else {
            panic!("expected ask command");
        };
        assert_eq!(args.session.as_deref(), Some("20260811-abc"));
        assert!(!args.thinking_enabled());
    }

    #[test]
    fn legacy_new_flags_still_parse() {
        let cli = Cli::try_parse_from([
            "hippocampus",
            "new",
            "--model",
            "qwen",
            "--context-window",
            "2048",
            "--max-output-tokens",
            "128",
            "--safety-margin-tokens",
            "32",
            "--no-think",
            "--system-prompt",
            "system",
        ])
        .unwrap();
        let Some(Command::New(args)) = cli.command else {
            panic!("expected new command");
        };
        assert_eq!(args.model, "qwen");
        assert_eq!(args.context_window, 2048);
        assert!(!args.thinking_enabled());
        assert_eq!(args.system_prompt.as_deref(), Some("system"));
    }

    #[test]
    fn serve_defaults_to_loopback_and_supports_session() {
        let cli = Cli::try_parse_from([
            "hippocampus",
            "serve",
            "--session",
            "20260811-abc",
            "--port",
            "8080",
        ])
        .unwrap();
        let Some(Command::Serve(args)) = cli.command else {
            panic!("expected serve command");
        };
        assert_eq!(args.session.as_deref(), Some("20260811-abc"));
        assert!(args.bind.is_loopback());
        assert_eq!(args.port, 8080);
    }
}
