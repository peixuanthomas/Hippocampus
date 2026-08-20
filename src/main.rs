use std::ffi::OsString;
use std::io::{self, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use hippocampus::config::AppConfig;
use hippocampus::engine::PreparationStatus;
use hippocampus::model::{
    BudgetConfig, ChatEvent, ChatEventKind, ChatMessage, RetrievalConfig, Session, TokenUsage,
    Turn, identity_instruction, utc_now,
};
use hippocampus::ollama::{ChatBackend, ChatRequest};
use hippocampus::{
    ChatEngine, ConsolidationRunReport, ConsolidationRunStatus, ConsolidationTrigger,
    ControlRecord, ControlTarget, ControlTargetKind, EmbeddingRefreshReport, EvalBenchmark,
    EvalRunOptions, EvalRunReport, GraphMaterializationReport, HybridRecallOptions,
    KnowledgeSyncReport, LimitAction, MemoryStatus, OllamaClient, RebuildOptions, RebuildReport,
    RecallChannels, RecallQueryOrigin, RecallResult, SessionStore, load_eval_corpus,
    run_evaluation, validate_eval_paths,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Parser)]
#[command(
    name = "hippocampus",
    version,
    about = "本地 Ollama 会话客户端：无参数进入 TUI，ask 用于脚本调用"
)]
struct Cli {
    /// 配置文件；未指定时可选读取当前目录 config.toml
    #[arg(long, global = true)]
    config: Option<PathBuf>,
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
    /// 清空全部会话历史及其派生记忆，保留知识库
    Clear,
    /// 单次调用；--json 输出逐行刷新的 JSONL 事件
    Ask(AskArgs),
    /// 启动本地 Web UI 并保持服务运行
    Serve(ServeArgs),
    /// 管理版本化本地知识库
    Knowledge(KnowledgeArgs),
    /// 管理派生记忆
    Memory(MemoryArgs),
    /// 运行可恢复的记忆评测
    Eval(EvalArgs),
}

#[derive(Debug, Args)]
struct EvalArgs {
    #[command(subcommand)]
    command: EvalCommand,
}

#[derive(Debug, Subcommand)]
enum EvalCommand {
    Synthetic,
    Longmemeval {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        limit: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Locomo {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        limit: usize,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Args)]
struct MemoryArgs {
    #[command(subcommand)]
    command: MemoryCommand,
}

#[derive(Debug, Subcommand)]
enum MemoryCommand {
    /// 从会话原始事件构建派生记忆
    Consolidate {
        #[arg(required_unless_present = "all", conflicts_with = "all")]
        session: Option<String>,
        #[arg(long, required_unless_present = "session", conflicts_with = "session")]
        all: bool,
        #[arg(long)]
        json: bool,
    },
    /// 查看派生记忆健康状态
    Status {
        session: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// 检索派生记忆，不生成回答
    Search {
        query: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long, value_enum, value_delimiter = ',')]
        channels: Vec<MemoryChannel>,
        #[arg(long)]
        json: bool,
    },
    /// 从原始会话重建派生记忆
    Rebuild {
        #[arg(long)]
        reembed: bool,
    },
    /// 排除会话或事件
    Exclude {
        #[command(subcommand)]
        target: MemoryTarget,
    },
    /// 恢复会话或事件
    Restore {
        #[command(subcommand)]
        target: MemoryTarget,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MemoryChannel {
    Bm25,
    Vector,
    Entity,
    State,
    Episode,
    Graph,
}

#[derive(Debug, Subcommand)]
enum MemoryTarget {
    Session { id: String },
    Event { id: String },
}

#[derive(Debug, Args)]
struct KnowledgeArgs {
    #[command(subcommand)]
    command: KnowledgeCommand,
}

#[derive(Debug, Subcommand)]
enum KnowledgeCommand {
    /// 同步当前配置的全部知识源
    Sync,
    /// 列出当前配置来源与最近同步状态
    List {
        #[arg(long)]
        json: bool,
    },
    /// 对当前知识索引执行确定性词法检索
    Search {
        query: String,
        #[arg(long)]
        json: bool,
    },
    /// 删除派生索引并从不可变快照重建
    Rebuild,
}

#[derive(Debug, Clone, Args)]
struct NewArgs {
    #[arg(long, default_value = "qwen3.8:27b-mlx")]
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
            model: "qwen3.8:27b-mlx".into(),
            context_window: 32_768,
            max_output_tokens: 4_096,
            safety_margin_tokens: 512,
            think: false,
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

    fn read_prompt(&self, config: &AppConfig) -> Result<String> {
        if let Some(path) = &self.system_prompt_file {
            return std::fs::read_to_string(path)
                .with_context(|| format!("无法读取系统提示文件 {}", path.display()));
        }
        Ok(self
            .system_prompt
            .clone()
            .unwrap_or_else(|| config.system_prompt.clone()))
    }

    fn thinking_enabled(&self) -> bool {
        self.think && !self.no_think
    }
}

#[derive(Debug, Args)]
struct AskArgs {
    /// 当前问题
    prompt: String,
    /// 使用该会话的历史上下文；不传则无历史且不创建会话
    #[arg(long)]
    session: Option<String>,
    #[arg(long, default_value = "qwen3.8:27b-mlx")]
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
    /// 输出逐行刷新、包含增量和最终元数据的 JSONL 事件
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
        self.think && !self.no_think
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        if error.is::<SilentCliExit>() {
            std::process::exit(1);
        }
        eprintln!("错误：{error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let loaded = AppConfig::load(cli.config.as_deref())?;
    if let Some(Command::Eval(args)) = cli.command {
        return run_eval(
            args,
            &cli.host,
            &cli.sessions_dir,
            loaded.path.as_deref(),
            loaded.config,
        )
        .await;
    }
    let config = loaded.config;
    if !config.memory.enabled
        && matches!(
            &cli.command,
            Some(Command::Memory(MemoryArgs {
                command: MemoryCommand::Rebuild { reembed: true }
            }))
        )
    {
        bail!("memory rebuild --reembed requires memory.enabled=true");
    }
    let store = SessionStore::new(&cli.sessions_dir)?;
    match cli.command {
        None => run_new_tui(store, &cli.host, NewArgs::default(), &config).await,
        Some(Command::New(args)) => run_new_tui(store, &cli.host, args, &config).await,
        Some(Command::Resume { identifier }) => run_resume_tui(store, &identifier, &config).await,
        Some(Command::List) => list_sessions(&store),
        Some(Command::Show { identifier, json }) => show_session(&store, &identifier, json),
        Some(Command::Clear) => clear_history(&store),
        Some(Command::Ask(args)) => run_ask(store, &cli.host, args, &config).await,
        Some(Command::Serve(args)) => run_serve(store, &cli.host, args, &config).await,
        Some(Command::Knowledge(args)) => run_knowledge(store, &cli.host, args, &config).await,
        Some(Command::Memory(args)) => run_memory(store, &cli.host, args, &config).await,
        Some(Command::Eval(_)) => unreachable!("evaluation returned before opening session store"),
    }
}

async fn run_eval(
    args: EvalArgs,
    host: &str,
    sessions_dir: &Path,
    config_path: Option<&Path>,
    config: AppConfig,
) -> Result<()> {
    let matrices = if config.memory.enabled {
        vec![
            ("bm25-only", RecallChannels::bm25_only()),
            (
                "vector-only",
                RecallChannels {
                    bm25: false,
                    vector: true,
                    entity: false,
                    state: false,
                    episode: false,
                    graph: false,
                },
            ),
            (
                "vector-graph",
                RecallChannels {
                    bm25: false,
                    vector: true,
                    entity: false,
                    state: false,
                    episode: false,
                    graph: true,
                },
            ),
            ("full", RecallChannels::all()),
        ]
    } else {
        vec![("bm25-only", RecallChannels::bm25_only())]
    };
    let client = OllamaClient::new(host)?;
    let ollama_host = host.trim_end_matches('/').to_owned();
    match args.command {
        EvalCommand::Synthetic => {
            println!(
                "evaluation matrix: {}",
                matrices
                    .iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let total = matrices.len();
            for (matrix, channels) in matrices {
                let output = PathBuf::from(format!("eval-results/synthetic.{matrix}.jsonl"));
                run_single_evaluation(
                    client.clone(),
                    config.clone(),
                    EvalBenchmark::Synthetic,
                    matrix,
                    None,
                    None,
                    output,
                    channels,
                    sessions_dir,
                    config_path,
                    &ollama_host,
                )
                .await?;
            }
            println!("completed matrices: {total}/{total}");
        }
        EvalCommand::Longmemeval {
            dataset,
            limit,
            output,
        } => {
            let (matrix, channels) = if config.memory.enabled {
                ("full", RecallChannels::all())
            } else {
                ("bm25-only", RecallChannels::bm25_only())
            };
            println!("evaluation matrix: {matrix}");
            run_single_evaluation(
                client,
                config,
                EvalBenchmark::LongMemEval,
                matrix,
                Some(dataset),
                Some(limit),
                output,
                channels,
                sessions_dir,
                config_path,
                &ollama_host,
            )
            .await?;
        }
        EvalCommand::Locomo {
            dataset,
            limit,
            output,
        } => {
            let (matrix, channels) = if config.memory.enabled {
                ("full", RecallChannels::all())
            } else {
                ("bm25-only", RecallChannels::bm25_only())
            };
            println!("evaluation matrix: {matrix}");
            run_single_evaluation(
                client,
                config,
                EvalBenchmark::Locomo,
                matrix,
                Some(dataset),
                Some(limit),
                output,
                channels,
                sessions_dir,
                config_path,
                &ollama_host,
            )
            .await?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_single_evaluation(
    client: OllamaClient,
    config: AppConfig,
    benchmark: EvalBenchmark,
    matrix: &str,
    dataset: Option<PathBuf>,
    limit: Option<usize>,
    output: PathBuf,
    channels: RecallChannels,
    sessions_dir: &Path,
    config_path: Option<&Path>,
    ollama_host: &str,
) -> Result<()> {
    let context = format!(
        "evaluation failed: benchmark={benchmark:?}, matrix={matrix}, output={}; 已持久化记录保留，可用相同命令恢复",
        output.display()
    );
    run_single_evaluation_inner(
        client,
        config,
        benchmark,
        matrix,
        dataset,
        limit,
        output,
        channels,
        sessions_dir,
        config_path,
        ollama_host,
    )
    .await
    .with_context(|| context)
}

#[allow(clippy::too_many_arguments)]
async fn run_single_evaluation_inner(
    client: OllamaClient,
    config: AppConfig,
    benchmark: EvalBenchmark,
    matrix: &str,
    dataset: Option<PathBuf>,
    limit: Option<usize>,
    output: PathBuf,
    channels: RecallChannels,
    sessions_dir: &Path,
    config_path: Option<&Path>,
    ollama_host: &str,
) -> Result<()> {
    let corpus = load_eval_corpus(benchmark, dataset.as_deref(), limit)?;
    if config.memory.candidate_limit < 10 {
        bail!("evaluation requires memory.candidate_limit >= 10");
    }
    let filename = output
        .file_name()
        .context("evaluation output filename missing")?;
    let mut workspace_name = OsString::from(".");
    workspace_name.push(filename);
    workspace_name.push(".workspace");
    let workspace = output.with_file_name(workspace_name);
    validate_eval_paths(dataset.as_deref(), &output, &workspace)?;
    validate_eval_store_isolation(&output, &workspace, sessions_dir, config_path)?;
    let options = EvalRunOptions {
        dataset_path: dataset,
        output: output.clone(),
        workspace,
        answer_model: "qwen3.8:27b-mlx".into(),
        ollama_host: ollama_host.to_owned(),
        channels,
        num_ctx: 32_768,
        num_predict: 4_096,
        selected_evidence_limit: 10,
    };
    let report = run_evaluation(client, config, corpus, options).await?;
    print_eval_report(benchmark, matrix, &report);
    Ok(())
}

fn print_eval_report(benchmark: EvalBenchmark, matrix: &str, report: &EvalRunReport) {
    println!("benchmark: {benchmark:?}");
    println!("matrix: {matrix}");
    println!("output: {}", report.output.display());
    println!("summary_path: {}", report.summary_path.display());
    println!("resumed_records: {}", report.resumed_records);
    println!("appended_records: {}", report.appended_records);
    println!(
        "completed/requested: {}/{}",
        report.summary.completed_questions, report.summary.requested_questions
    );
    println!("run_fingerprint: {}", report.summary.run_fingerprint);
}

fn validate_eval_store_isolation(
    output: &Path,
    workspace: &Path,
    sessions_dir: &Path,
    config_path: Option<&Path>,
) -> Result<()> {
    let mut summary_name = OsString::from(
        output
            .file_name()
            .context("evaluation output filename missing")?,
    );
    summary_name.push(".summary.json");
    let summary = output.with_file_name(summary_name);
    reject_parent_path(output, "evaluation output")?;
    reject_parent_path(&summary, "evaluation summary")?;
    reject_parent_path(workspace, "evaluation workspace")?;
    reject_parent_path(sessions_dir, "sessions directory")?;
    let output = resolve_path_components(output)?;
    let summary = resolve_path_components(&summary)?;
    let workspace = resolve_path_components(workspace)?;
    let sessions = resolve_path_components(sessions_dir)?;
    ensure_disjoint(
        &workspace,
        &sessions,
        "evaluation workspace",
        "sessions directory",
    )?;
    if let Some(config_path) = config_path {
        reject_parent_path(config_path, "config path")?;
        let config = resolve_path_components(config_path)?;
        if output == config {
            bail!("evaluation output and config path must be distinct");
        }
        if summary == config {
            bail!("evaluation summary and config path must be distinct");
        }
        ensure_disjoint(&workspace, &config, "evaluation workspace", "config path")?;
    }
    Ok(())
}

fn reject_parent_path(path: &Path, label: &str) -> Result<()> {
    if path
        .components()
        .any(|component| component == Component::ParentDir)
    {
        bail!("{label} must not contain parent-directory components");
    }
    Ok(())
}

fn resolve_path_components(path: &Path) -> Result<PathBuf> {
    let mut resolved = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::fs::canonicalize(std::env::current_dir()?)?
    };
    for component in path.components() {
        match component {
            Component::RootDir => resolved.push(Path::new("/")),
            Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => unreachable!("parent components rejected before resolution"),
            Component::Normal(part) => {
                let candidate = resolved.join(part);
                match std::fs::symlink_metadata(&candidate) {
                    Ok(_) => {
                        resolved = std::fs::canonicalize(&candidate).with_context(|| {
                            format!("cannot resolve path {}", candidate.display())
                        })?;
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => resolved.push(part),
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("cannot inspect path {}", candidate.display())
                        });
                    }
                }
            }
        }
    }
    Ok(resolved)
}

fn ensure_disjoint(a: &Path, b: &Path, a_label: &str, b_label: &str) -> Result<()> {
    if a == b || a.starts_with(b) || b.starts_with(a) {
        bail!("{a_label} and {b_label} must be disjoint");
    }
    Ok(())
}

async fn run_serve(
    store: SessionStore,
    host: &str,
    args: ServeArgs,
    config: &AppConfig,
) -> Result<()> {
    let sync_host = args
        .session
        .as_deref()
        .map(|identifier| store.load(identifier).map(|session| session.ollama_host))
        .transpose()?
        .unwrap_or_else(|| host.to_owned());
    auto_sync_knowledge(&store, &sync_host, config).await?;
    let mut session = if let Some(identifier) = &args.session {
        let mut session = store.load(identifier)?;
        store.reopen(&mut session)?;
        session
    } else {
        let prompt = args.new.read_prompt(config)?;
        store.create_named(
            &args.new.model,
            host,
            config.ai_name(),
            Some(&prompt),
            args.new.budget(),
            args.new.thinking_enabled(),
        )?
    };
    let client = OllamaClient::new(&session.ollama_host)?;
    let info = client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::with_config(store.clone(), client, config.clone());
    let address = SocketAddr::new(args.bind, args.port);
    hippocampus::web::serve(engine, session.clone(), info, address).await?;
    session = store.load(&session.id)?;
    store.save(&mut session)?;
    Ok(())
}

async fn run_new_tui(
    store: SessionStore,
    host: &str,
    args: NewArgs,
    config: &AppConfig,
) -> Result<()> {
    auto_sync_knowledge(&store, host, config).await?;
    let prompt = args.read_prompt(config)?;
    let session = store.create_named(
        &args.model,
        host,
        config.ai_name(),
        Some(&prompt),
        args.budget(),
        args.thinking_enabled(),
    )?;
    let client = OllamaClient::new(&session.ollama_host)?;
    let info = client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::with_config(store.clone(), client, config.clone());
    let outcome = hippocampus::tui::run(engine, session, info).await?;
    let mut session = outcome.session;
    store.save(&mut session)?;
    run_tui_exit_consolidation(
        &store,
        &session,
        config,
        consolidation_trigger_for_tui_exit(outcome.exit_reason),
    )
    .await;
    Ok(())
}

async fn run_resume_tui(store: SessionStore, identifier: &str, config: &AppConfig) -> Result<()> {
    let mut session = store.load(identifier)?;
    auto_sync_knowledge(&store, &session.ollama_host, config).await?;
    store.reopen(&mut session)?;
    let client = OllamaClient::new(&session.ollama_host)?;
    let info = client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::with_config(store.clone(), client, config.clone());
    let outcome = hippocampus::tui::run(engine, session, info).await?;
    let mut session = outcome.session;
    store.save(&mut session)?;
    run_tui_exit_consolidation(
        &store,
        &session,
        config,
        consolidation_trigger_for_tui_exit(outcome.exit_reason),
    )
    .await;
    Ok(())
}

async fn run_ask(store: SessionStore, host: &str, args: AskArgs, config: &AppConfig) -> Result<()> {
    if let Some(identifier) = args.session.clone() {
        return run_contextual_ask(store, &identifier, args, config).await;
    }
    run_stateless_ask(host, args, config).await
}

async fn run_contextual_ask(
    store: SessionStore,
    identifier: &str,
    args: AskArgs,
    config: &AppConfig,
) -> Result<()> {
    let mut session = store.load(identifier)?;
    auto_sync_knowledge(&store, &session.ollama_host, config).await?;
    store.reopen(&mut session)?;
    let client = OllamaClient::new(&session.ollama_host)?;
    client
        .check_model(&session.model, session.budget.context_window)
        .await?;
    let engine = ChatEngine::with_config(store, client, config.clone());
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
            if json_output {
                print_json_stream_event(&event);
            } else if event.kind == ChatEventKind::Content {
                print!("{}", event.text);
                let _ = io::stdout().flush();
            } else if show_thinking && event.kind == ChatEventKind::Thinking {
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
                "event": "done",
                "session_id": session.id,
                "stateless": false,
                "turn_id": turn.id,
                "status": turn.status,
                "thinking": turn.thinking,
                "content": turn.assistant_content,
                "usage": turn.usage,
                "knowledge": turn.context_trace.knowledge,
                "web": turn.context_trace.web,
                "knowledge_sources": turn.context_trace.knowledge.selected_evidence,
                "web_sources": turn.context_trace.web.sources,
                "warnings": turn_warnings(turn),
            }))?
        );
    } else {
        println!();
        io::stdout().flush()?;
        print_usage(session.turns.last().map(|turn| turn.usage));
        if let Some(turn) = session.turns.last() {
            print_turn_sources(turn);
        }
    }
    Ok(())
}

async fn run_knowledge(
    store: SessionStore,
    host: &str,
    args: KnowledgeArgs,
    config: &AppConfig,
) -> Result<()> {
    match args.command {
        KnowledgeCommand::Sync => {
            let client = OllamaClient::new(host)?;
            let report = store.knowledge().sync(&config.knowledge, &client).await?;
            print_sync_report(&report);
        }
        KnowledgeCommand::List { json } => {
            let statuses = store.knowledge().list(&config.knowledge)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&statuses)?);
            } else if statuses.is_empty() {
                println!("当前配置没有知识来源。");
            } else {
                for status in statuses {
                    println!(
                        "{}｜{:?}｜{}｜{} 篇｜成功={}｜错误={}",
                        status.id,
                        status.kind,
                        if status.enabled { "active" } else { "inactive" },
                        status.active_documents,
                        status.last_success_at.as_deref().unwrap_or("从未"),
                        status.last_error.as_deref().unwrap_or("无"),
                    );
                    println!("  {}", status.location);
                }
            }
        }
        KnowledgeCommand::Search { query, json } => {
            let recall = store.knowledge().recall(&query, &config.knowledge)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&recall.trace)?);
            } else {
                println!("状态：{}", recall.trace.status);
                for (index, evidence) in recall.trace.selected_evidence.iter().enumerate() {
                    println!(
                        "[K{}] {}｜{}｜{}..{}\n  source={}\n  revision={}",
                        index + 1,
                        evidence.title,
                        evidence.fetched_at,
                        evidence.start_char,
                        evidence.end_char,
                        evidence.source_location,
                        evidence.revision_id,
                    );
                }
                for warning in &recall.trace.warnings {
                    eprintln!("警告：{warning}");
                }
            }
        }
        KnowledgeCommand::Rebuild => {
            let documents = store.knowledge().rebuild_for_config(&config.knowledge)?;
            println!("知识索引已从原始快照重建：{documents} 个 passage。");
        }
    }
    Ok(())
}

#[derive(Debug)]
struct SilentCliExit;

impl std::fmt::Display for SilentCliExit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("command failed after printing its report")
    }
}

impl std::error::Error for SilentCliExit {}

#[derive(Serialize)]
struct MemorySearchOutput<'a> {
    query: &'a str,
    session_id: Option<&'a str>,
    reference_time: &'a str,
    channels: &'a [&'static str],
    recall: &'a RecallResult,
}

struct MemoryMaintenanceReport {
    rebuild: RebuildReport,
    embedding: Option<EmbeddingRefreshReport>,
    graph: Option<GraphMaterializationReport>,
}

async fn run_memory(
    store: SessionStore,
    host: &str,
    args: MemoryArgs,
    config: &AppConfig,
) -> Result<()> {
    match args.command {
        MemoryCommand::Consolidate { session, all, json } => {
            run_memory_consolidate(store, session, all, json, config).await
        }
        MemoryCommand::Status { session, json } => {
            run_memory_status(&store, session.as_deref(), json, config)
        }
        MemoryCommand::Search {
            query,
            session,
            channels,
            json,
        } => {
            run_memory_search(
                &store,
                host,
                &query,
                session.as_deref(),
                &channels,
                json,
                config,
            )
            .await
        }
        MemoryCommand::Rebuild { reembed } => {
            if reembed && !config.memory.enabled {
                bail!("memory rebuild --reembed requires memory.enabled=true");
            }
            let rebuild = rebuild_projection(&store, !reembed).await?;
            let report = refresh_after_rebuild(&store, host, config, rebuild)
                .await
                .map_err(|error| {
                    anyhow::anyhow!(
                        "raw data is unchanged and the rebuild commit succeeded, but derived refresh is incomplete: {error:#}"
                    )
                })?;
            print_memory_maintenance(&report);
            Ok(())
        }
        MemoryCommand::Exclude { target } => {
            run_memory_control(&store, host, target, true, config).await
        }
        MemoryCommand::Restore { target } => {
            run_memory_control(&store, host, target, false, config).await
        }
    }
}

fn run_memory_status(
    store: &SessionStore,
    session: Option<&str>,
    json_output: bool,
    config: &AppConfig,
) -> Result<()> {
    let status = store.retrieval().memory_status(&config.memory, session)?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        print_memory_status(&status);
    }
    if status.healthy {
        Ok(())
    } else {
        Err(SilentCliExit.into())
    }
}

fn print_memory_status(status: &MemoryStatus) {
    println!(
        "session_id: {}",
        status.session_id.as_deref().unwrap_or("all")
    );
    println!("healthy: {}", status.healthy);
    println!("memory_enabled: {}", status.memory_enabled);
    println!("projection_current: {}", status.projection_current);
    println!(
        "expected_control_generation_sha256: {}",
        status.expected_control_generation_sha256
    );
    println!(
        "indexed_control_generation_sha256: {}",
        status
            .indexed_control_generation_sha256
            .as_deref()
            .unwrap_or("none")
    );
    println!(
        "validation_error: {}",
        status.validation_error.as_deref().unwrap_or("none")
    );
    if let Some(metrics) = &status.metrics {
        println!("active_sessions: {}", metrics.active_sessions);
        println!("active_events: {}", metrics.active_events);
        println!("embedding_total: {}", metrics.embedding_total);
        println!("embedding_compatible: {}", metrics.embedding_compatible);
        println!("embedding_stale: {}", metrics.embedding_stale);
        println!(
            "pending_consolidation_events: {}",
            metrics.pending_consolidation_events
        );
        println!("entity_count: {}", metrics.entity_count);
        println!("episode_count: {}", metrics.episode_count);
        println!("graph_current: {}", metrics.graph_current);
        println!(
            "graph_node_count: {}",
            display_optional(metrics.graph_node_count)
        );
        println!(
            "graph_edge_count: {}",
            display_optional(metrics.graph_edge_count)
        );
        println!(
            "consolidation_attempts: applied={} rejected={} model_error={} cancelled={} failed={}",
            metrics.consolidation_attempts.applied,
            metrics.consolidation_attempts.rejected,
            metrics.consolidation_attempts.model_error,
            metrics.consolidation_attempts.cancelled,
            metrics.consolidation_attempts.failed
        );
        println!(
            "consolidation_latency_ms: samples={} p50={} p95={}",
            metrics.consolidation_latency_ms.samples,
            display_optional(metrics.consolidation_latency_ms.p50_ms),
            display_optional(metrics.consolidation_latency_ms.p95_ms)
        );
        println!("retrieval_runs: {}", metrics.retrieval_runs);
        println!("retrieval_failures: {}", metrics.retrieval_failures);
        println!(
            "retrieval_latency_ms: samples={} p50={} p95={}",
            metrics.retrieval_latency_ms.samples,
            display_optional(metrics.retrieval_latency_ms.p50_ms),
            display_optional(metrics.retrieval_latency_ms.p95_ms)
        );
    }
}

fn display_optional<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "none".into(), |value| value.to_string())
}

async fn run_memory_search(
    store: &SessionStore,
    host: &str,
    query: &str,
    session: Option<&str>,
    requested_channels: &[MemoryChannel],
    json_output: bool,
    config: &AppConfig,
) -> Result<()> {
    let (session_id, retrieval_config) = if let Some(identifier) = session {
        let session = store.load(identifier)?;
        (Some(session.id), session.retrieval)
    } else {
        (None, RetrievalConfig::default())
    };
    let channels = recall_channels(requested_channels, config.memory.enabled)?;
    let channel_names = enabled_channel_names(channels);
    let reference_time = utc_now();
    let sentinel = memory_search_sentinel(query, session_id.as_deref(), &channel_names);
    let client = OllamaClient::new(host)?;
    let recall = store
        .retrieval()
        .hybrid_recall_with_options(
            &client,
            query,
            &sentinel,
            &[],
            session_id.as_deref(),
            retrieval_config,
            &config.memory,
            HybridRecallOptions {
                channels,
                query_origin: RecallQueryOrigin::Synthetic {
                    reference_time: reference_time.clone(),
                },
            },
        )
        .await?;
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&MemorySearchOutput {
                query,
                session_id: session_id.as_deref(),
                reference_time: &reference_time,
                channels: &channel_names,
                recall: &recall,
            })?
        );
    } else {
        for channel in &recall.trace.channels {
            println!(
                "{:?}: status={} count={} error={}",
                channel.channel,
                channel.status,
                channel.candidate_count,
                channel.error.as_deref().unwrap_or("none")
            );
        }
        for warning in &recall.trace.warnings {
            println!("warning: {warning}");
        }
        for (index, evidence) in recall.evidence.iter().enumerate() {
            println!(
                "[{}] {}:{}..{} role={:?} kind={:?} reason={}\n{}",
                index + 1,
                evidence.selected.span.event_id,
                evidence.selected.span.start_char,
                evidence.selected.span.end_char,
                evidence.selected.role,
                evidence.selected.kind,
                evidence.selected.reason,
                evidence.content
            );
        }
    }
    Ok(())
}

fn recall_channels(requested: &[MemoryChannel], memory_enabled: bool) -> Result<RecallChannels> {
    let mut channels = if requested.is_empty() {
        if memory_enabled {
            RecallChannels::all()
        } else {
            RecallChannels::bm25_only()
        }
    } else {
        RecallChannels {
            bm25: false,
            vector: false,
            entity: false,
            state: false,
            episode: false,
            graph: false,
        }
    };
    for channel in requested {
        match channel {
            MemoryChannel::Bm25 => channels.bm25 = true,
            MemoryChannel::Vector => channels.vector = true,
            MemoryChannel::Entity => channels.entity = true,
            MemoryChannel::State => channels.state = true,
            MemoryChannel::Episode => channels.episode = true,
            MemoryChannel::Graph => channels.graph = true,
        }
    }
    channels.validate().map_err(anyhow::Error::msg)?;
    if !memory_enabled && channels != RecallChannels::bm25_only() {
        bail!("memory.enabled=false only supports BM25-only recall");
    }
    Ok(channels)
}

fn enabled_channel_names(channels: RecallChannels) -> Vec<&'static str> {
    [
        (channels.bm25, "bm25"),
        (channels.vector, "vector"),
        (channels.entity, "entity"),
        (channels.state, "state"),
        (channels.episode, "episode"),
        (channels.graph, "graph"),
    ]
    .into_iter()
    .filter_map(|(enabled, name)| enabled.then_some(name))
    .collect()
}

fn memory_search_sentinel(query: &str, session_id: Option<&str>, channels: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"hippocampus-memory-search-sentinel-v1\0");
    hasher.update(query.as_bytes());
    hasher.update(b"\0");
    hasher.update(session_id.unwrap_or("<all-sessions>").as_bytes());
    for channel in channels {
        hasher.update(b"\0");
        hasher.update(channel.as_bytes());
    }
    format!("synthetic-memory-search-{:x}", hasher.finalize())
}

async fn rebuild_and_refresh(
    store: &SessionStore,
    host: &str,
    config: &AppConfig,
    reuse_compatible_embeddings: bool,
) -> Result<MemoryMaintenanceReport> {
    let rebuild = rebuild_projection(store, reuse_compatible_embeddings).await?;
    refresh_after_rebuild(store, host, config, rebuild).await
}

async fn rebuild_projection(
    store: &SessionStore,
    reuse_compatible_embeddings: bool,
) -> Result<RebuildReport> {
    let retrieval = store.retrieval().clone();
    tokio::task::spawn_blocking(move || {
        retrieval.rebuild_with_options(RebuildOptions {
            reuse_compatible_embeddings,
        })
    })
    .await
    .context("rebuild task failed")?
    .map_err(Into::into)
}

async fn refresh_after_rebuild(
    store: &SessionStore,
    host: &str,
    config: &AppConfig,
    rebuild: RebuildReport,
) -> Result<MemoryMaintenanceReport> {
    if !config.memory.enabled {
        return Ok(MemoryMaintenanceReport {
            rebuild,
            embedding: None,
            graph: None,
        });
    }
    let client = OllamaClient::new(host)?;
    let engine = ChatEngine::with_config(store.clone(), client, config.clone());
    let embedding = engine
        .refresh_embeddings(cancellation_on_ctrl_c())
        .await
        .context("embedding and episode refresh failed")?;
    let retrieval = store.retrieval().clone();
    let memory = config.memory.clone();
    let graph = tokio::task::spawn_blocking(move || retrieval.refresh_graph(&memory))
        .await
        .context("graph refresh task failed")??;
    Ok(MemoryMaintenanceReport {
        rebuild,
        embedding: Some(embedding),
        graph: Some(graph),
    })
}

async fn run_memory_control(
    store: &SessionStore,
    host: &str,
    target: MemoryTarget,
    exclude: bool,
    config: &AppConfig,
) -> Result<()> {
    let target = match target {
        MemoryTarget::Session { id } => ControlTarget {
            kind: ControlTargetKind::Session,
            id,
        },
        MemoryTarget::Event { id } => ControlTarget {
            kind: ControlTargetKind::Event,
            id,
        },
    };
    let record = if exclude {
        store.exclude(target)?
    } else {
        store.restore(target)?
    };
    let report = rebuild_and_refresh(store, host, config, true)
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "control record sequence {} is committed, but projection/refresh is incomplete: {error:#}",
                record.sequence
            )
        })?;
    print_control_record(&record);
    print_memory_maintenance(&report);
    Ok(())
}

fn print_control_record(record: &ControlRecord) {
    let action = match record.action {
        hippocampus::ControlAction::Exclude => "exclude",
        hippocampus::ControlAction::Restore => "restore",
    };
    let target_kind = match record.target_kind {
        ControlTargetKind::Session => "session",
        ControlTargetKind::Event => "event",
    };
    println!(
        "control: action={} target={}:{} sequence={} hash={}",
        action, target_kind, record.target_id, record.sequence, record.record_sha256
    );
}

fn print_memory_maintenance(report: &MemoryMaintenanceReport) {
    println!(
        "rebuild: sessions={} events={} spans={} answer_contexts={} documents={} control_generation={} ledger_preserved={} attempts_replayed={} skipped_inactive={} skipped_dependency={} embeddings_reused={}",
        report.rebuild.sync.sessions,
        report.rebuild.sync.events,
        report.rebuild.sync.spans,
        report.rebuild.sync.answer_contexts,
        report.rebuild.sync.documents,
        report.rebuild.control_generation_sha256,
        report.rebuild.ledger_attempts_preserved,
        report.rebuild.consolidation_attempts_replayed,
        report.rebuild.consolidation_attempts_skipped_inactive,
        report.rebuild.consolidation_attempts_skipped_dependency,
        report.rebuild.embeddings_reused
    );
    if let Some(embedding) = report.embedding {
        println!(
            "embedding: leaf_documents={} leaf_reused={} leaf_embedded_inputs={} backend_batches={} aggregate_documents={} leaf_committed={}",
            embedding.leaf_documents,
            embedding.leaf_reused,
            embedding.leaf_embedded_inputs,
            embedding.backend_batches,
            embedding.aggregate_documents,
            embedding.leaf_committed
        );
    } else {
        println!("embedding: disabled");
    }
    if let Some(graph) = &report.graph {
        println!(
            "graph: changed={} nodes={} edges={} source_sha256={} catalog_sha256={} vector_index_fingerprint={}",
            graph.changed,
            graph.node_count,
            graph.edge_count,
            graph.source_sha256,
            graph.catalog_sha256,
            graph.vector_index_fingerprint
        );
    } else {
        println!("graph: disabled");
    }
}

async fn run_memory_consolidate(
    store: SessionStore,
    identifier: Option<String>,
    all: bool,
    json_output: bool,
    config: &AppConfig,
) -> Result<()> {
    let sessions = if all {
        let mut sessions = store.list_sessions()?;
        order_sessions_for_bulk_consolidation(&mut sessions);
        sessions
    } else {
        vec![
            store.load(
                identifier
                    .as_deref()
                    .expect("clap requires a session when --all is absent"),
            )?,
        ]
    };
    let cancellation = cancellation_on_ctrl_c();
    let mut reports = Vec::with_capacity(sessions.len());
    for session in sessions {
        if cancellation.is_cancelled() {
            break;
        }
        if !json_output && config.memory.enabled {
            eprintln!("巩固中：会话 {}（模型 {}）…", session.id, session.model);
        }
        let report = consolidate_for_session(
            &store,
            &session,
            config,
            ConsolidationTrigger::Manual,
            cancellation.clone(),
        )
        .await;
        let stop = report.status == ConsolidationRunStatus::Cancelled;
        reports.push(report);
        if stop || cancellation.is_cancelled() {
            break;
        }
    }
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&ConsolidationReports { reports: &reports })?
        );
    } else {
        for report in &reports {
            println!("{}", format_consolidation_report(report));
            for warning in &report.warnings {
                println!("警告：{warning}");
            }
        }
    }
    Ok(())
}

fn order_sessions_for_bulk_consolidation(sessions: &mut [Session]) {
    sessions.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.id.cmp(&right.id))
    });
}

fn consolidation_trigger_for_tui_exit(
    exit_reason: hippocampus::tui::TuiExitReason,
) -> ConsolidationTrigger {
    match exit_reason {
        hippocampus::tui::TuiExitReason::ExitCommand => ConsolidationTrigger::TuiExit,
        hippocampus::tui::TuiExitReason::IdleCtrlC => ConsolidationTrigger::TuiIdleCtrlC,
    }
}

async fn run_tui_exit_consolidation(
    store: &SessionStore,
    session: &Session,
    config: &AppConfig,
    trigger: ConsolidationTrigger,
) {
    if config.memory.enabled {
        eprintln!("巩固中：会话 {}（模型 {}）…", session.id, session.model);
    }
    let report =
        consolidate_for_session(store, session, config, trigger, cancellation_on_ctrl_c()).await;
    eprintln!("{}", format_consolidation_report(&report));
    for warning in &report.warnings {
        eprintln!("警告：{warning}");
    }
}

async fn consolidate_for_session(
    store: &SessionStore,
    session: &Session,
    config: &AppConfig,
    trigger: ConsolidationTrigger,
    cancellation: CancellationToken,
) -> ConsolidationRunReport {
    if !config.memory.enabled {
        return synthetic_consolidation_report(
            session,
            trigger,
            ConsolidationRunStatus::Disabled,
            Vec::new(),
        );
    }
    if cancellation.is_cancelled() {
        return synthetic_consolidation_report(
            session,
            trigger,
            ConsolidationRunStatus::Cancelled,
            Vec::new(),
        );
    }
    let client = match OllamaClient::new(&session.ollama_host) {
        Ok(client) => client,
        Err(error) => {
            return synthetic_consolidation_report(
                session,
                trigger,
                ConsolidationRunStatus::Failed,
                vec![format!("无法创建 Ollama 客户端：{error}")],
            );
        }
    };
    ChatEngine::with_config(store.clone(), client, config.clone())
        .consolidate_session(session, trigger, cancellation)
        .await
}

fn synthetic_consolidation_report(
    session: &Session,
    trigger: ConsolidationTrigger,
    status: ConsolidationRunStatus,
    warnings: Vec<String>,
) -> ConsolidationRunReport {
    ConsolidationRunReport {
        session_id: session.id.clone(),
        trigger,
        model: session.model.clone(),
        status,
        batches_attempted: 0,
        batches_applied: 0,
        events_attempted: 0,
        events_applied: 0,
        entities_attempted: 0,
        entities_applied: 0,
        claims_attempted: 0,
        claims_applied: 0,
        boundaries_attempted: 0,
        boundaries_applied: 0,
        watermark_before: 0,
        watermark_after: 0,
        warnings,
    }
}

#[derive(Serialize)]
struct ConsolidationReports<'a> {
    reports: &'a [ConsolidationRunReport],
}

fn format_consolidation_report(report: &ConsolidationRunReport) -> String {
    format!(
        "会话 {}｜模型 {}｜{}｜批次 {}/{}｜事件 {}/{}｜实体 {}/{}｜声明 {}/{}｜边界 {}/{}｜水位 {}→{}",
        report.session_id,
        report.model,
        consolidation_status_label(report.status),
        report.batches_applied,
        report.batches_attempted,
        report.events_applied,
        report.events_attempted,
        report.entities_applied,
        report.entities_attempted,
        report.claims_applied,
        report.claims_attempted,
        report.boundaries_applied,
        report.boundaries_attempted,
        report.watermark_before,
        report.watermark_after,
    )
}

const fn consolidation_status_label(status: ConsolidationRunStatus) -> &'static str {
    match status {
        ConsolidationRunStatus::Disabled => "已禁用",
        ConsolidationRunStatus::UpToDate => "已是最新",
        ConsolidationRunStatus::Completed => "已完成",
        ConsolidationRunStatus::Partial => "部分完成",
        ConsolidationRunStatus::Failed => "失败",
        ConsolidationRunStatus::Cancelled => "已取消",
    }
}

async fn auto_sync_knowledge(store: &SessionStore, host: &str, config: &AppConfig) -> Result<()> {
    if !config.knowledge.auto_sync {
        return Ok(());
    }
    let client = OllamaClient::new(host)?;
    let report = store.knowledge().sync(&config.knowledge, &client).await?;
    for warning in &report.warnings {
        eprintln!("警告：{warning}；继续使用最近成功的知识版本（如有）。");
    }
    Ok(())
}

fn print_sync_report(report: &KnowledgeSyncReport) {
    println!(
        "知识同步完成：来源 {}，成功 {}，失败 {}，活动文档 {}，新增 revision {}。",
        report.configured_sources,
        report.successful_sources,
        report.failed_sources,
        report.active_documents,
        report.new_revisions,
    );
    for warning in &report.warnings {
        eprintln!("警告：{warning}；继续使用最近成功的知识版本（如有）。");
    }
}

async fn run_stateless_ask(host: &str, args: AskArgs, config: &AppConfig) -> Result<()> {
    let system_prompt = read_ask_prompt(&args, config)?;
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
        role: "system".into(),
        content: identity_instruction(config.ai_name()),
    });
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
                    if json_output {
                        print_json_stream_event(&event);
                    } else if show_thinking {
                        eprint!("{}", event.text);
                        let _ = io::stderr().flush();
                    }
                }
                ChatEventKind::Content => {
                    content.push_str(&event.text);
                    if json_output {
                        print_json_stream_event(&event);
                    } else {
                        print!("{}", event.text);
                        let _ = io::stdout().flush();
                    }
                }
                ChatEventKind::Completed => {
                    usage = event.usage;
                    if json_output {
                        print_json_stream_event(&event);
                    }
                }
                ChatEventKind::Usage => {
                    if json_output {
                        print_json_stream_event(&event);
                    }
                }
            },
        )
        .await?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "done",
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

fn json_stream_event(event: &ChatEvent) -> Value {
    match event.kind {
        ChatEventKind::Thinking => json!({
            "event": "thinking",
            "delta": event.text,
            "live_output_tokens": event.live_output_tokens,
        }),
        ChatEventKind::Content => json!({
            "event": "content",
            "delta": event.text,
            "live_output_tokens": event.live_output_tokens,
        }),
        ChatEventKind::Usage => json!({
            "event": "usage",
            "live_output_tokens": event.live_output_tokens,
        }),
        ChatEventKind::Completed => json!({
            "event": "completed",
            "live_output_tokens": event.live_output_tokens,
            "usage": event.usage,
            "done_reason": event.done_reason,
        }),
    }
}

fn print_json_stream_event(event: &ChatEvent) {
    if let Ok(line) = serde_json::to_string(&json_stream_event(event)) {
        println!("{line}");
        let _ = io::stdout().flush();
    }
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

fn read_ask_prompt(args: &AskArgs, config: &AppConfig) -> Result<String> {
    if let Some(path) = &args.system_prompt_file {
        return std::fs::read_to_string(path)
            .with_context(|| format!("无法读取系统提示文件 {}", path.display()));
    }
    Ok(args
        .system_prompt
        .clone()
        .unwrap_or_else(|| config.system_prompt.clone()))
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

fn clear_history(store: &SessionStore) -> Result<()> {
    let report = store.clear_history()?;
    println!(
        "已清空全部历史记录：{} 个会话，{} 个临时文件，{} 个记忆索引文件；知识库已保留。",
        report.sessions_removed, report.temporary_files_removed, report.index_files_removed
    );
    Ok(())
}

fn show_session(store: &SessionStore, identifier: &str, json_output: bool) -> Result<()> {
    let session = store.load(identifier)?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&session)?);
        return Ok(());
    }
    println!(
        "会话 {}｜{}｜{}\nAI：{}｜模型：{}｜Ollama：{}｜thinking：{}\n系统提示：{}",
        session.id,
        session_status(&session),
        session.title,
        session.ai_name,
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
        print_turn_sources(turn);
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

fn turn_warnings(turn: &Turn) -> Vec<String> {
    let mut warnings = turn.context_trace.knowledge.warnings.clone();
    warnings.extend(turn.context_trace.web.warnings.clone());
    warnings.sort();
    warnings.dedup();
    warnings
}

fn print_turn_sources(turn: &Turn) {
    if !turn.context_trace.knowledge.selected_evidence.is_empty() {
        eprintln!("知识来源：");
        for evidence in &turn.context_trace.knowledge.selected_evidence {
            eprintln!(
                "  - {}｜{}｜revision={}｜{}..{}",
                evidence.title,
                evidence.source_location,
                evidence.revision_id,
                evidence.start_char,
                evidence.end_char,
            );
        }
    }
    if !turn.context_trace.web.sources.is_empty() {
        eprintln!("实时来源：");
        for source in &turn.context_trace.web.sources {
            eprintln!("  - {}｜{}｜{}", source.kind, source.title, source.url);
        }
    }
    for warning in turn_warnings(turn) {
        eprintln!("警告：{warning}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_stream_events_use_exact_deltas_and_stable_event_names() {
        let thinking = ChatEvent::text(ChatEventKind::Thinking, "think".into(), 1);
        let content = ChatEvent::text(ChatEventKind::Content, "delta".into(), 2);
        let completed = ChatEvent {
            kind: ChatEventKind::Completed,
            text: String::new(),
            live_output_tokens: Some(2),
            usage: Some(TokenUsage::new(Some(3), Some(2))),
            done_reason: Some("stop".into()),
        };
        assert_eq!(json_stream_event(&thinking)["event"], "thinking");
        assert_eq!(json_stream_event(&thinking)["delta"], "think");
        assert_eq!(json_stream_event(&content)["event"], "content");
        assert_eq!(json_stream_event(&content)["delta"], "delta");
        assert_eq!(json_stream_event(&completed)["event"], "completed");
        assert_eq!(json_stream_event(&completed)["usage"]["input_tokens"], 3);
    }

    #[test]
    fn ask_session_is_optional_and_thinking_defaults_off() {
        let stateless = Cli::try_parse_from(["hippocampus", "ask", "hello"]).unwrap();
        let Some(Command::Ask(args)) = stateless.command else {
            panic!("expected ask command");
        };
        assert!(args.session.is_none());
        assert!(!args.thinking_enabled());

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
    fn command_line_system_prompts_override_config() {
        let config = AppConfig {
            system_prompt: "from config".into(),
            ..AppConfig::default()
        };
        let mut new_args = NewArgs::default();
        assert_eq!(new_args.read_prompt(&config).unwrap(), "from config");
        new_args.system_prompt = Some("from cli".into());
        assert_eq!(new_args.read_prompt(&config).unwrap(), "from cli");

        let root = tempfile::tempdir().unwrap();
        let prompt_file = root.path().join("system.txt");
        std::fs::write(&prompt_file, "from file").unwrap();
        new_args.system_prompt = None;
        new_args.system_prompt_file = Some(prompt_file);
        assert_eq!(new_args.read_prompt(&config).unwrap(), "from file");
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

    #[test]
    fn knowledge_commands_parse_with_json_flags() {
        let cli = Cli::try_parse_from([
            "hippocampus",
            "--config",
            "custom.toml",
            "knowledge",
            "search",
            "海棠计划",
            "--json",
        ])
        .unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("custom.toml")));
        let Some(Command::Knowledge(KnowledgeArgs {
            command: KnowledgeCommand::Search { query, json },
        })) = cli.command
        else {
            panic!("expected knowledge search command");
        };
        assert_eq!(query, "海棠计划");
        assert!(json);
    }

    #[test]
    fn clear_command_parses() {
        let cli = Cli::try_parse_from(["hippocampus", "clear"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Clear)));
    }

    #[test]
    fn memory_consolidate_target_rules_parse() {
        let single =
            Cli::try_parse_from(["hippocampus", "memory", "consolidate", "session-a"]).unwrap();
        assert!(matches!(
            single.command,
            Some(Command::Memory(MemoryArgs {
                command: MemoryCommand::Consolidate {
                    session: Some(session),
                    all: false,
                    json: false,
                },
            })) if session == "session-a"
        ));

        let all = Cli::try_parse_from(["hippocampus", "memory", "consolidate", "--all", "--json"])
            .unwrap();
        assert!(matches!(
            all.command,
            Some(Command::Memory(MemoryArgs {
                command: MemoryCommand::Consolidate {
                    session: None,
                    all: true,
                    json: true,
                },
            }))
        ));

        assert!(Cli::try_parse_from(["hippocampus", "memory", "consolidate"]).is_err());
        assert!(
            Cli::try_parse_from(["hippocampus", "memory", "consolidate", "session-a", "--all",])
                .is_err()
        );
    }

    #[test]
    fn memory_consolidate_all_order_is_deterministic() {
        fn session(id: &str, updated_at: &str) -> Session {
            let mut session = Session::new(
                id.into(),
                "model".into(),
                "http://localhost:11434".into(),
                "system".into(),
                BudgetConfig::default(),
                true,
            )
            .unwrap();
            session.updated_at = updated_at.into();
            session
        }

        let mut sessions = vec![
            session("tie-b", "2026-08-12T12:00:00Z"),
            session("newest", "2026-08-13T12:00:00Z"),
            session("tie-a", "2026-08-12T12:00:00Z"),
        ];

        order_sessions_for_bulk_consolidation(&mut sessions);

        assert_eq!(
            sessions
                .iter()
                .map(|session| session.id.as_str())
                .collect::<Vec<_>>(),
            vec!["newest", "tie-a", "tie-b"]
        );
        assert_eq!(sessions[1].updated_at, sessions[2].updated_at);
    }

    #[test]
    fn tui_exit_reasons_map_to_exact_consolidation_triggers() {
        assert_eq!(
            consolidation_trigger_for_tui_exit(hippocampus::tui::TuiExitReason::ExitCommand),
            ConsolidationTrigger::TuiExit
        );
        assert_eq!(
            consolidation_trigger_for_tui_exit(hippocampus::tui::TuiExitReason::IdleCtrlC),
            ConsolidationTrigger::TuiIdleCtrlC
        );
    }

    #[test]
    fn memory_consolidation_output_is_machine_stable() {
        let statuses = [
            ConsolidationRunStatus::Disabled,
            ConsolidationRunStatus::UpToDate,
            ConsolidationRunStatus::Completed,
            ConsolidationRunStatus::Partial,
            ConsolidationRunStatus::Failed,
            ConsolidationRunStatus::Cancelled,
        ];
        let reports = statuses
            .into_iter()
            .enumerate()
            .map(|(index, status)| ConsolidationRunReport {
                session_id: format!("session-{index}"),
                trigger: ConsolidationTrigger::Manual,
                model: format!("model-{index}"),
                status,
                batches_attempted: 2,
                batches_applied: 1,
                events_attempted: 4,
                events_applied: 3,
                entities_attempted: 5,
                entities_applied: 4,
                claims_attempted: 6,
                claims_applied: 5,
                boundaries_attempted: 7,
                boundaries_applied: 6,
                watermark_before: 8,
                watermark_after: 9,
                warnings: Vec::new(),
            })
            .collect::<Vec<_>>();
        let encoded = serde_json::to_value(ConsolidationReports { reports: &reports }).unwrap();
        let object = encoded.as_object().unwrap();
        assert_eq!(object.len(), 1);
        assert!(object.contains_key("reports"));
        let encoded_reports = object["reports"].as_array().unwrap();
        for (report, expected_status) in encoded_reports.iter().zip([
            "disabled",
            "up_to_date",
            "completed",
            "partial",
            "failed",
            "cancelled",
        ]) {
            assert_eq!(report["trigger"], "manual");
            assert_eq!(report["status"], expected_status);
        }
        for report in &reports {
            let human = format_consolidation_report(report);
            assert!(human.contains(&report.session_id));
            assert!(human.contains(&report.model));
            assert!(human.contains("批次 1/2"));
            assert!(human.contains("事件 3/4"));
            assert!(human.contains("水位 8→9"));
            assert!(human.contains(consolidation_status_label(report.status)));
        }
    }
}
