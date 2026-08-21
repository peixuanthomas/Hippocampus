use std::io::{self, IsTerminal, Stdout};
use std::time::Duration;

use anyhow::{Result, bail};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Wrap,
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use unicode_width::UnicodeWidthChar;

use crate::engine::{ChatEngine, LimitAction, PreparationStatus};
use crate::model::{ChatEvent, ChatEventKind, Session, Turn};
use crate::ollama::{ChatBackend, ModelInfo, OllamaClient};
use crate::store::IndexSyncAfterSourceCommit;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
    Thinking,
    System,
    Debug,
    Error,
}

#[derive(Debug, Clone)]
struct Message {
    role: Role,
    content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activity {
    Idle,
    Preparing,
    Generating,
    AwaitingLimit,
    Cancelling,
    Switching,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiExitReason {
    ExitCommand,
    IdleCtrlC,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TuiRunOutcome {
    pub session: Session,
    pub exit_reason: TuiExitReason,
}

enum SwitchOutcome {
    Noop,
    Ready(Box<SwitchReady>),
}

struct SwitchReady {
    engine: ChatEngine<OllamaClient>,
    session: Session,
    model_info: ModelInfo,
}

type SwitchResult = std::result::Result<SwitchOutcome, String>;

const MAX_BACKGROUND_EVENTS_PER_FRAME: usize = 64;
const MAX_VISIBLE_STREAM_EVENTS_PER_FRAME: usize = 8;

enum BackgroundEvent {
    Status(String),
    Debug(String),
    LimitWarning(String),
    Stream(ChatEvent),
    Finished {
        session: Box<Session>,
        result: std::result::Result<(), String>,
    },
    SessionSwitched {
        generation: u64,
        result: Box<SwitchResult>,
    },
}

struct App {
    engine: ChatEngine<OllamaClient>,
    session: Session,
    model_info: ModelInfo,
    messages: Vec<Message>,
    editor: InputEditor,
    status: String,
    activity: Activity,
    live_thinking: String,
    live_answer: String,
    live_answer_started: bool,
    live_tokens: u64,
    scroll: usize,
    follow_output: bool,
    max_scroll: usize,
    debug: bool,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
    rx: mpsc::UnboundedReceiver<BackgroundEvent>,
    decision_tx: Option<mpsc::UnboundedSender<LimitAction>>,
    cancellation: Option<CancellationToken>,
    switch_task: Option<JoinHandle<()>>,
    switch_generation: u64,
}

impl App {
    fn new(engine: ChatEngine<OllamaClient>, session: Session, model_info: ModelInfo) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let messages = messages_from_session(&session);
        Self {
            engine,
            session,
            model_info,
            messages,
            editor: InputEditor::default(),
            status: "就绪".into(),
            activity: Activity::Idle,
            live_thinking: String::new(),
            live_answer: String::new(),
            live_answer_started: false,
            live_tokens: 0,
            scroll: 0,
            follow_output: true,
            max_scroll: 0,
            debug: false,
            tx,
            rx,
            decision_tx: None,
            cancellation: None,
            switch_task: None,
            switch_generation: 0,
        }
    }

    fn start_turn(&mut self, input: String) {
        self.messages.push(Message {
            role: Role::User,
            content: input.clone(),
        });
        self.editor.remember(input.clone());
        self.live_thinking.clear();
        self.live_answer.clear();
        self.live_answer_started = false;
        self.live_tokens = 0;
        self.activity = Activity::Preparing;
        self.status = "正在组装上下文…".into();
        self.follow_output = true;

        let mut session = self.session.clone();
        let engine = self.engine.clone();
        let debug = self.debug;
        let ui_tx = self.tx.clone();
        let (decision_tx, mut decision_rx) = mpsc::unbounded_channel();
        let cancellation = CancellationToken::new();
        self.decision_tx = Some(decision_tx);
        self.cancellation = Some(cancellation.clone());

        tokio::spawn(async move {
            let result: Result<()> = async {
                let mut prepared = engine.prepare_turn(&mut session, input).await?;
                if prepared.needs_limit_decision() {
                    let _ = ui_tx.send(BackgroundEvent::LimitWarning(prepared.message.clone()));
                    let action = decision_rx.recv().await.unwrap_or(LimitAction::EndSession);
                    prepared = engine.resolve_limit(&mut session, prepared, action).await?;
                    if !prepared.message.is_empty() {
                        let _ = ui_tx.send(BackgroundEvent::Status(prepared.message.clone()));
                    }
                }
                if prepared.status == PreparationStatus::Blocked
                    || prepared.status == PreparationStatus::Ended
                {
                    bail!(prepared.message);
                }
                if debug {
                    let _ = ui_tx.send(BackgroundEvent::Status("正在显示最终组装输入…".into()));
                    let _ = ui_tx.send(BackgroundEvent::Debug(format_prepared_turn(
                        &session, &prepared,
                    )));
                }
                let source = if prepared.plan.exact_input_tokens.is_some() {
                    "精确"
                } else {
                    "估计上界"
                };
                let input_tokens = prepared
                    .plan
                    .exact_input_tokens
                    .or(prepared.plan.estimated_upper_tokens)
                    .map_or_else(|| "未知".into(), |value| value.to_string());
                let _ = ui_tx.send(BackgroundEvent::Status(format!(
                    "生成中 · {source} input={input_tokens}/{} · included={} omitted={}",
                    prepared.plan.input_budget,
                    prepared.plan.included_turn_ids.len(),
                    prepared.plan.omitted_turn_ids.len()
                )));
                let stream_tx = ui_tx.clone();
                engine
                    .stream_turn(&mut session, &prepared, cancellation, move |event| {
                        let _ = stream_tx.send(BackgroundEvent::Stream(event));
                    })
                    .await
            }
            .await;
            let _ = ui_tx.send(BackgroundEvent::Finished {
                session: Box::new(session),
                result: result.map_err(|error| error.to_string()),
            });
        });
    }

    fn start_session_switch(&mut self, identifier: String) {
        self.activity = Activity::Switching;
        self.status = format!("正在切换到会话 {identifier}…");
        self.switch_generation = self.switch_generation.wrapping_add(1);
        let generation = self.switch_generation;
        let store = self.engine.store().clone();
        let config = self.engine.config().clone();
        let active_session_id = self.session.id.clone();
        let tx = self.tx.clone();
        self.switch_task = Some(tokio::spawn(async move {
            let result: Result<_> = async {
                let target = store.load(&identifier)?;
                if target.id == active_session_id {
                    return Ok(SwitchOutcome::Noop);
                }
                let client = OllamaClient::new(&target.ollama_host)?;
                let info = client
                    .check_model(&target.model, target.budget.context_window)
                    .await?;
                let engine = ChatEngine::with_config(store, client, config);
                Ok(SwitchOutcome::Ready(Box::new(SwitchReady {
                    engine,
                    session: target,
                    model_info: info,
                })))
            }
            .await;
            let _ = tx.send(BackgroundEvent::SessionSwitched {
                generation,
                result: Box::new(result.map_err(|error| error.to_string())),
            });
        }));
    }

    fn drain_background(&mut self) {
        let mut visible_stream_events = 0;
        for _ in 0..MAX_BACKGROUND_EVENTS_PER_FRAME {
            let Ok(event) = self.rx.try_recv() else {
                break;
            };
            let visibly_updates_stream = matches!(
                &event,
                BackgroundEvent::Stream(ChatEvent {
                    kind: ChatEventKind::Thinking | ChatEventKind::Content,
                    text,
                    ..
                }) if !text.is_empty()
            );
            match event {
                BackgroundEvent::Debug(content) => self.messages.push(Message {
                    role: Role::Debug,
                    content,
                }),
                BackgroundEvent::Status(status) => {
                    self.status = status;
                    if self.activity == Activity::Preparing {
                        self.activity = Activity::Generating;
                    }
                }
                BackgroundEvent::LimitWarning(message) => {
                    self.activity = Activity::AwaitingLimit;
                    self.status = message;
                }
                BackgroundEvent::Stream(event) => {
                    self.activity = Activity::Generating;
                    if let Some(tokens) = event.live_output_tokens {
                        self.live_tokens = tokens;
                    }
                    append_live_stream_event(
                        &mut self.live_thinking,
                        &mut self.live_answer,
                        &mut self.live_answer_started,
                        &event,
                    );
                }
                BackgroundEvent::Finished { session, result } => {
                    self.session = *session;
                    if !self.live_answer_started && !self.live_thinking.is_empty() {
                        self.messages.push(Message {
                            role: Role::Thinking,
                            content: std::mem::take(&mut self.live_thinking),
                        });
                    } else {
                        self.live_thinking.clear();
                    }
                    if !self.live_answer.is_empty() {
                        self.messages.push(Message {
                            role: Role::Assistant,
                            content: std::mem::take(&mut self.live_answer),
                        });
                    }
                    self.live_answer_started = false;
                    if let Some(summary) = self.session.turns.last().and_then(provenance_summary) {
                        self.messages.push(Message {
                            role: Role::Debug,
                            content: summary,
                        });
                    }
                    self.live_tokens = 0;
                    self.activity = Activity::Idle;
                    self.decision_tx = None;
                    self.cancellation = None;
                    match result {
                        Ok(()) => {
                            let turn = self.session.turns.last();
                            if let Some(error) = turn.and_then(|turn| turn.error.as_deref()) {
                                self.status = error.to_owned();
                            } else if let Some(turn) = turn {
                                self.status = format!(
                                    "完成 · input={} output={}",
                                    display_optional(turn.usage.input_tokens),
                                    display_optional(turn.usage.output_tokens)
                                );
                            } else {
                                self.status = "完成".into();
                            }
                        }
                        Err(error) => {
                            self.status = error.clone();
                            self.messages.push(Message {
                                role: Role::Error,
                                content: error,
                            });
                        }
                    }
                }
                BackgroundEvent::SessionSwitched { generation, result } => {
                    if generation != self.switch_generation || self.activity != Activity::Switching
                    {
                        continue;
                    }
                    self.switch_task = None;
                    self.activity = Activity::Idle;
                    match *result {
                        Ok(SwitchOutcome::Noop) => {
                            self.activity = Activity::Idle;
                            self.status = same_session_status().into();
                            self.push_system(same_session_status());
                        }
                        Ok(SwitchOutcome::Ready(ready)) => {
                            let SwitchReady {
                                engine,
                                mut session,
                                model_info,
                            } = *ready;
                            match engine.store().reopen(&mut session) {
                                Ok(()) => {
                                    self.accept_session_switch(engine, session, model_info, None)
                                }
                                Err(error) => {
                                    let committed = error
                                        .downcast_ref::<IndexSyncAfterSourceCommit>()
                                        .is_some();
                                    let warning = error.to_string();
                                    if committed
                                        && let Ok(persisted) = engine.store().load(&session.id)
                                        && persisted.status == crate::model::SessionStatus::Active
                                    {
                                        self.accept_session_switch(
                                            engine,
                                            persisted,
                                            model_info,
                                            Some(warning),
                                        );
                                    } else {
                                        self.status = format!("切换会话失败：{warning}");
                                        self.push_error(self.status.clone());
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            self.status = format!("切换会话失败：{error}");
                            self.push_error(self.status.clone());
                        }
                    }
                }
            }
            if visibly_updates_stream {
                visible_stream_events += 1;
                if visible_stream_events == MAX_VISIBLE_STREAM_EVENTS_PER_FRAME {
                    break;
                }
            }
        }
    }

    fn cancel_generation(&mut self) {
        if let Some(cancellation) = &self.cancellation {
            cancellation.cancel();
            self.activity = Activity::Cancelling;
            self.status = "正在中断并保存已收到的内容…".into();
        }
        if self.activity == Activity::Switching {
            if let Some(task) = self.switch_task.take() {
                task.abort();
            }
            self.switch_generation = self.switch_generation.wrapping_add(1);
            self.activity = Activity::Idle;
            self.status = "已取消会话切换。".into();
        }
    }

    fn accept_session_switch(
        &mut self,
        engine: ChatEngine<OllamaClient>,
        session: Session,
        model_info: ModelInfo,
        warning: Option<String>,
    ) {
        self.engine = engine;
        self.session = session;
        self.model_info = model_info;
        self.messages = messages_from_session(&self.session);
        self.scroll = 0;
        self.follow_output = true;
        self.status = "已切换会话。".into();
        if let Some(warning) = warning {
            self.push_error(format!("会话已切换，但派生索引同步失败：{warning}"));
        }
    }

    fn resolve_limit(&mut self, action: LimitAction) {
        if let Some(sender) = self.decision_tx.take() {
            let _ = sender.send(action);
            self.activity = Activity::Preparing;
            self.status = if action == LimitAction::ContinueWithTrim {
                "正在计算可保留的最大最近轮次…".into()
            } else {
                "正在暂停并保存会话…".into()
            };
        }
    }

    fn handle_idle_submit(&mut self) -> Result<Option<TuiExitReason>> {
        let input = self.editor.take().trim_end().to_owned();
        if input.trim().is_empty() {
            return Ok(None);
        }
        if input.starts_with('/') {
            return self.handle_command(&input);
        }
        self.start_turn(input);
        Ok(None)
    }

    fn handle_command(&mut self, command: &str) -> Result<Option<TuiExitReason>> {
        let parts = command.split_whitespace().collect::<Vec<_>>();
        match parts.as_slice() {
            ["/exit"] => return Ok(Some(TuiExitReason::ExitCommand)),
            ["/save"] => {
                let path = self.engine.store().save(&mut self.session)?;
                self.push_system(format!("已原子保存：{}", path.display()));
            }
            ["/help"] => self.push_system(
                "/list · /session <id> · /debug [on|off] · /budget · /think on|off · /save · /help · /exit\n鼠标滚轮或 PageUp/PageDown 滚动，Ctrl+C 中断生成或退出。",
            ),
            ["/list"] => match self.engine.store().list_sessions() {
                Ok(sessions) if sessions.is_empty() => self.push_system("没有保存的会话。"),
                Ok(sessions) => self.push_system(format_session_list(&sessions, &self.session.id)),
                Err(error) => self.push_error(format!("无法列出会话：{error}")),
            },
            ["/session", identifier] => self.start_session_switch((*identifier).to_owned()),
            ["/session"] | ["/session", ..] => self.push_error("用法：/session <id>"),
            ["/debug"] => self.push_system(format!(
                "debug：{}",
                if self.debug { "on" } else { "off" }
            )),
            ["/debug", value @ ("on" | "off")] => {
                self.debug = *value == "on";
                self.push_system(format!("debug 已设为：{value}"));
            }
            ["/debug", ..] => self.push_error("用法：/debug on|off"),
            ["/budget"] => {
                let budget = &self.session.budget;
                self.push_system(format!(
                    "context={} · input={} · output reserve={} · safety={}\n80%={} · 90%={} · active start={}\n回答累计 input={} output={} · probe 累计 input={} output={}",
                    budget.context_window,
                    budget.input_budget(),
                    budget.max_output_tokens,
                    budget.safety_margin_tokens,
                    budget.probe_threshold(),
                    budget.warning_threshold(),
                    self.session.active_context_start_index,
                    display_optional(self.session.cumulative_usage.input_tokens),
                    display_optional(self.session.cumulative_usage.output_tokens),
                    display_optional(self.session.cumulative_probe_usage.input_tokens),
                    display_optional(self.session.cumulative_probe_usage.output_tokens),
                ));
            }
            ["/think"] => self.push_system(format!(
                "thinking：{}",
                if self.session.think { "on" } else { "off" }
            )),
            ["/think", value @ ("on" | "off")] => {
                self.session.think = *value == "on";
                self.engine.store().save(&mut self.session)?;
                self.push_system(format!("thinking 已设为：{value}"));
            }
            ["/think", ..] => self.push_error("用法：/think on|off"),
            _ => self.push_error("未知命令；输入 /help 查看命令。"),
        }
        Ok(None)
    }

    fn push_system(&mut self, content: impl Into<String>) {
        self.messages.push(Message {
            role: Role::System,
            content: content.into(),
        });
        self.follow_output = true;
    }

    fn push_error(&mut self, content: impl Into<String>) {
        self.messages.push(Message {
            role: Role::Error,
            content: content.into(),
        });
        self.follow_output = true;
    }

    fn scroll_up(&mut self, lines: usize) {
        self.follow_output = false;
        self.scroll = self.scroll.saturating_sub(lines);
    }

    fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_add(lines).min(self.max_scroll);
        if self.scroll == self.max_scroll {
            self.follow_output = true;
        }
    }
}

const fn same_session_status() -> &'static str {
    "已是当前会话。"
}

fn append_live_stream_event(
    live_thinking: &mut String,
    live_answer: &mut String,
    live_answer_started: &mut bool,
    event: &ChatEvent,
) {
    match event.kind {
        ChatEventKind::Thinking if !*live_answer_started => live_thinking.push_str(&event.text),
        ChatEventKind::Content => {
            *live_answer_started = true;
            live_thinking.clear();
            live_answer.push_str(&event.text);
        }
        ChatEventKind::Thinking | ChatEventKind::Usage | ChatEventKind::Completed => {}
    }
}

fn messages_from_session(session: &Session) -> Vec<Message> {
    let mut messages = vec![Message {
        role: Role::System,
        content: "输入 /help 查看命令；Enter 发送，Ctrl+J 换行。".into(),
    }];
    for turn in &session.turns {
        messages.push(Message {
            role: Role::User,
            content: turn.user_content.clone(),
        });
        if !turn.thinking.is_empty() && turn.assistant_content.is_empty() {
            messages.push(Message {
                role: Role::Thinking,
                content: turn.thinking.clone(),
            });
        }
        if !turn.assistant_content.is_empty() {
            messages.push(Message {
                role: Role::Assistant,
                content: turn.assistant_content.clone(),
            });
        } else if let Some(error) = &turn.error {
            messages.push(Message {
                role: Role::Error,
                content: format!("[{}] {error}", turn.status.as_str()),
            });
        }
        if let Some(summary) = provenance_summary(turn) {
            messages.push(Message {
                role: Role::Debug,
                content: summary,
            });
        }
    }
    messages
}

fn format_session_list(sessions: &[Session], current_id: &str) -> String {
    sessions
        .iter()
        .map(|session| {
            format!(
                "{} {} · {} · {} · {} · {} 轮\n{}",
                if session.id == current_id { "*" } else { " " },
                session.id,
                session.status.as_str(),
                session.updated_at,
                session.model,
                session.turns.len(),
                session.title
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_prepared_turn(session: &Session, prepared: &crate::engine::PreparedTurn) -> String {
    use std::collections::BTreeMap;
    let mut per_role = BTreeMap::<&str, usize>::new();
    let mut chars = 0;
    let mut bytes = 0;
    for message in &prepared.plan.messages {
        *per_role.entry(&message.role).or_default() += 1;
        chars += message.content.chars().count();
        bytes += message.content.len();
    }
    let roles = per_role
        .into_iter()
        .map(|(role, count)| format!("{role}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let budget = &session.budget;
    let mut lines = vec![
        "DEBUG · 最终上下文组装输入（完整、未截断）".to_owned(),
        format!(
            "session={} model={} think={}",
            session.id, session.model, session.think
        ),
        format!(
            "num_ctx={} context_window={} num_predict={} max_output_tokens={}",
            budget.context_window,
            budget.context_window,
            budget.max_output_tokens,
            budget.max_output_tokens
        ),
        format!(
            "safety_margin={} input_budget={} trim/probe_80%={} warning_90%={}",
            budget.safety_margin_tokens,
            prepared.plan.input_budget,
            budget.probe_threshold(),
            budget.warning_threshold()
        ),
        format!(
            "messages={} per_role=[{roles}] chars={chars} bytes={bytes}",
            prepared.plan.messages.len()
        ),
        format!(
            "context_items={} included_turns={} ids=[{}] omitted_turns={} ids=[{}]",
            prepared.plan.context_items.len(),
            prepared.plan.included_turn_ids.len(),
            prepared.plan.included_turn_ids.join(", "),
            prepared.plan.omitted_turn_ids.len(),
            prepared.plan.omitted_turn_ids.join(", ")
        ),
        format!(
            "estimated_upper_input_tokens={} exact_input_tokens={} context_sha256={}",
            display_optional(prepared.plan.estimated_upper_tokens),
            display_optional(prepared.plan.exact_input_tokens),
            prepared.plan.context_sha256
        ),
        format!(
            "retrieval_selected={} knowledge_selected={}",
            prepared.plan.retrieval_trace.selected_evidence.len(),
            prepared.plan.knowledge_trace.selected_evidence.len()
        ),
        "--- assembled messages ---".to_owned(),
    ];
    for (index, message) in prepared.plan.messages.iter().enumerate() {
        lines.push(format!("[{index}] role={}", message.role));
        lines.push(message.content.clone());
        lines.push("---".to_owned());
    }
    lines.join("\n")
}

fn provenance_summary(turn: &Turn) -> Option<String> {
    let mut lines = Vec::new();
    if !turn.context_trace.knowledge.selected_evidence.is_empty() {
        lines.push("知识来源（由程序 trace 生成）".to_owned());
        for evidence in &turn.context_trace.knowledge.selected_evidence {
            lines.push(format!(
                "[K] {} · {} · revision={} · {}..{}",
                evidence.title,
                evidence.source_location,
                evidence.revision_id,
                evidence.start_char,
                evidence.end_char
            ));
        }
    }
    if !turn.context_trace.web.sources.is_empty() {
        lines.push("实时来源（由程序 trace 生成）".to_owned());
        for source in &turn.context_trace.web.sources {
            lines.push(format!(
                "[W] {} · {} · {}",
                source.kind, source.title, source.url
            ));
        }
    }
    let mut warnings = turn.context_trace.knowledge.warnings.clone();
    warnings.extend(turn.context_trace.web.warnings.clone());
    warnings.sort();
    warnings.dedup();
    lines.extend(
        warnings
            .into_iter()
            .map(|warning| format!("警告：{warning}")),
    );
    (!lines.is_empty()).then(|| lines.join("\n"))
}

pub async fn run(
    engine: ChatEngine<OllamaClient>,
    session: Session,
    model_info: ModelInfo,
) -> Result<TuiRunOutcome> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        bail!("TUI 需要交互终端；脚本调用请使用 `hippocampus ask \"问题\"`");
    }
    if let Err(error) = enable_raw_mode() {
        let _ = disable_raw_mode();
        return Err(error.into());
    }
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = match Terminal::new(backend) {
        Ok(terminal) => terminal,
        Err(error) => {
            let _ = disable_raw_mode();
            return Err(error.into());
        }
    };
    let mut alternate_screen = false;
    let mut bracketed_paste = false;
    let mut mouse_capture = false;
    let result: Result<TuiRunOutcome> = async {
        alternate_screen = true;
        execute!(terminal.backend_mut(), EnterAlternateScreen)?;
        bracketed_paste = true;
        execute!(terminal.backend_mut(), EnableBracketedPaste)?;
        mouse_capture = true;
        execute!(terminal.backend_mut(), EnableMouseCapture)?;
        terminal.clear()?;
        run_loop(&mut terminal, App::new(engine, session, model_info)).await
    }
    .await;
    let cleanup = restore_terminal(
        &mut terminal,
        alternate_screen,
        bracketed_paste,
        mouse_capture,
    );
    match (result, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(outcome), Ok(())) => Ok(outcome),
    }
}

fn restore_terminal(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    alternate_screen: bool,
    bracketed_paste: bool,
    mouse_capture: bool,
) -> Result<()> {
    let mut first_error = None;
    let mut record = |result: io::Result<()>| {
        if let Err(error) = result
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    };
    if mouse_capture {
        record(execute!(terminal.backend_mut(), DisableMouseCapture));
    }
    if bracketed_paste {
        record(execute!(terminal.backend_mut(), DisableBracketedPaste));
    }
    if alternate_screen {
        record(execute!(terminal.backend_mut(), LeaveAlternateScreen));
    }
    record(terminal.show_cursor());
    record(disable_raw_mode());
    first_error.map_or(Ok(()), |error| Err(error.into()))
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut app: App,
) -> Result<TuiRunOutcome> {
    loop {
        app.drain_background();
        terminal.draw(|frame| draw(frame, &mut app))?;
        if !event::poll(Duration::from_millis(40))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                if let Some(exit_reason) = handle_key(&mut app, key)? {
                    return Ok(TuiRunOutcome {
                        session: app.session,
                        exit_reason,
                    });
                }
            }
            Event::Paste(text) if app.activity == Activity::Idle => {
                app.editor.insert_str(&text);
                app.follow_output = true;
            }
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => app.scroll_up(3),
                MouseEventKind::ScrollDown => app.scroll_down(3),
                _ => {}
            },
            Event::Resize(_, _) => {}
            _ => {}
        }
    }
}

fn handle_key(app: &mut App, key: KeyEvent) -> Result<Option<TuiExitReason>> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if app.activity == Activity::AwaitingLimit {
            app.resolve_limit(limit_action_for_key(key).expect("Ctrl+C ends pending limit"));
            return Ok(None);
        }
        if app.activity == Activity::Idle {
            return Ok(Some(TuiExitReason::IdleCtrlC));
        }
        app.cancel_generation();
        return Ok(None);
    }
    if app.activity == Activity::AwaitingLimit {
        if let Some(action) = limit_action_for_key(key) {
            app.resolve_limit(action);
        }
        return Ok(None);
    }

    match key.code {
        KeyCode::PageUp => {
            app.scroll_up(8);
        }
        KeyCode::PageDown => {
            app.scroll_down(8);
        }
        _ if app.activity != Activity::Idle => return Ok(None),
        KeyCode::Enter
            if key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT) =>
        {
            app.editor.insert_char('\n');
            app.follow_output = true;
        }
        KeyCode::Enter => return app.handle_idle_submit(),
        KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.editor.insert_char('\n');
            app.follow_output = true;
        }
        KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.editor.move_line_start()
        }
        KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.editor.move_line_end()
        }
        KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.editor.delete_previous_word()
        }
        KeyCode::Backspace => app.editor.backspace(),
        KeyCode::Delete => app.editor.delete(),
        KeyCode::Left => app.editor.move_left(),
        KeyCode::Right => app.editor.move_right(),
        KeyCode::Home => app.editor.move_line_start(),
        KeyCode::End => app.editor.move_line_end(),
        KeyCode::Up => app.editor.move_up_or_history(),
        KeyCode::Down => app.editor.move_down_or_history(),
        KeyCode::Esc => app.editor.clear(),
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.editor.insert_char(ch);
            app.follow_output = true;
        }
        _ => {}
    }
    Ok(None)
}

fn limit_action_for_key(key: KeyEvent) -> Option<LimitAction> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(LimitAction::EndSession);
    }
    match key.code {
        KeyCode::Enter | KeyCode::Char('c' | 'C' | 'y' | 'Y') => {
            Some(LimitAction::ContinueWithTrim)
        }
        KeyCode::Esc | KeyCode::Char('e' | 'E' | 'n' | 'N') => Some(LimitAction::EndSession),
        _ => None,
    }
}

fn draw(frame: &mut ratatui::Frame<'_>, app: &mut App) {
    let area = frame.area();
    let input_height = input_height(&app.editor.text, area.width.saturating_sub(4));
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(1),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
    draw_header(frame, app, chunks[0]);
    draw_messages(frame, app, chunks[1]);
    draw_status(frame, app, chunks[2]);
    draw_input(frame, app, chunks[3]);
    draw_help(frame, app, chunks[4]);
    if app.activity == Activity::AwaitingLimit {
        draw_limit_dialog(frame, area);
    }
}

fn draw_header(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let state = app.session.status.as_str();
    let title = Line::from(vec![
        Span::styled(
            format!(" {} ", app.session.ai_name.to_uppercase()),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            " local memory · raw truth ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let right = format!(
        "{}  ·  think:{}  ·  ctx:{}  ·  {}  ",
        app.session.model,
        if app.session.think { "on" } else { "off" },
        app.session.budget.context_window,
        state
    );
    let header = Paragraph::new(title)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .alignment(Alignment::Left);
    frame.render_widget(header, area);
    let right_width = right
        .chars()
        .count()
        .min(area.width.saturating_sub(2) as usize) as u16;
    if right_width > 0 {
        let right_area = Rect::new(
            area.right().saturating_sub(right_width + 1),
            area.y + 1,
            right_width,
            1,
        );
        frame.render_widget(
            Paragraph::new(right).style(Style::default().fg(Color::Gray)),
            right_area,
        );
    }
}

fn draw_messages(frame: &mut ratatui::Frame<'_>, app: &mut App, area: Rect) {
    let block = Block::default()
        .borders(Borders::LEFT | Borders::RIGHT)
        .border_style(Style::default().fg(Color::Rgb(45, 50, 60)));
    let inner = block.inner(area);
    let width = inner.width.max(1) as usize;
    let mut lines = Vec::new();
    for message in visible_messages(&app.messages, app.debug) {
        append_message_lines(
            &mut lines,
            message.role,
            &message.content,
            width,
            &app.session.ai_name,
        );
    }
    if !app.live_thinking.is_empty() {
        append_message_lines(
            &mut lines,
            Role::Thinking,
            &app.live_thinking,
            width,
            &app.session.ai_name,
        );
    }
    if !app.live_answer.is_empty() || app.activity == Activity::Generating {
        append_message_lines(
            &mut lines,
            Role::Assistant,
            if app.live_answer.is_empty() {
                "…"
            } else {
                &app.live_answer
            },
            width,
            &app.session.ai_name,
        );
    }
    let viewport = inner.height as usize;
    let max_scroll = lines.len().saturating_sub(viewport);
    app.max_scroll = max_scroll;
    if app.follow_output {
        app.scroll = max_scroll;
    } else {
        app.scroll = app.scroll.min(max_scroll);
    }
    let content_length = lines.len();
    let visible_lines = message_window(&lines, app.scroll, viewport);
    let paragraph = Paragraph::new(Text::from(visible_lines)).block(block);
    frame.render_widget(paragraph, area);
    if max_scroll > 0 {
        let mut scrollbar = ScrollbarState::new(content_length)
            .viewport_content_length(viewport)
            .position(app.scroll);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area,
            &mut scrollbar,
        );
    }
}

fn message_window<'a>(lines: &[Line<'a>], scroll: usize, viewport: usize) -> Vec<Line<'a>> {
    lines.iter().skip(scroll).take(viewport).cloned().collect()
}

fn visible_messages(messages: &[Message], debug: bool) -> impl Iterator<Item = &Message> {
    messages
        .iter()
        .filter(move |message| debug || message.role != Role::Debug)
}

fn append_message_lines<'a>(
    lines: &mut Vec<Line<'a>>,
    role: Role,
    content: &str,
    width: usize,
    ai_name: &str,
) {
    let (label, color) = match role {
        Role::User => ("› You".to_owned(), Color::Cyan),
        Role::Assistant => (format!("◆ {ai_name}"), Color::Green),
        Role::Thinking => ("◌ Thinking".to_owned(), Color::Magenta),
        Role::System => ("· System".to_owned(), Color::Yellow),
        Role::Debug => ("▧ Debug".to_owned(), Color::DarkGray),
        Role::Error => ("! Error".to_owned(), Color::Red),
    };
    lines.push(Line::from(Span::styled(
        label,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )));
    let content_style = match role {
        Role::System | Role::Debug => Style::default().fg(Color::DarkGray),
        Role::Thinking => Style::default().fg(Color::LightMagenta),
        Role::Error => Style::default().fg(Color::LightRed),
        _ => Style::default().fg(Color::White),
    };
    for line in wrap_plain(content, width.saturating_sub(2).max(1)) {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(line, content_style),
        ]));
    }
    lines.push(Line::raw(""));
}

fn draw_status(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let color = match app.activity {
        Activity::Idle => Color::DarkGray,
        Activity::AwaitingLimit => Color::Yellow,
        Activity::Cancelling => Color::Red,
        Activity::Preparing | Activity::Generating | Activity::Switching => Color::Cyan,
    };
    let spinner = match app.activity {
        Activity::Idle => "",
        Activity::AwaitingLimit => "⚠ ",
        Activity::Cancelling => "■ ",
        Activity::Preparing | Activity::Generating | Activity::Switching => "● ",
    };
    frame.render_widget(
        Paragraph::new(format!(" {spinner}{}", app.status)).style(Style::default().fg(color)),
        area,
    );
}

fn draw_input(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let border = match app.activity {
        Activity::Idle => Color::Cyan,
        Activity::AwaitingLimit => Color::Yellow,
        Activity::Cancelling => Color::Red,
        Activity::Preparing | Activity::Generating | Activity::Switching => Color::DarkGray,
    };
    let title = if app.activity == Activity::Idle {
        " Message · Enter send · Ctrl+J newline "
    } else {
        " Working · Ctrl+C interrupt "
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border));
    let inner = block.inner(area);
    let text = if app.editor.text.is_empty() && app.activity == Activity::Idle {
        Text::from(Span::styled(
            "Ask anything, or type /help…",
            Style::default().fg(Color::DarkGray),
        ))
    } else {
        Text::from(app.editor.text.clone())
    };
    let (cursor_row, cursor_col) = app.editor.cursor_position(inner.width.max(1));
    let scroll = cursor_row.saturating_sub(inner.height.saturating_sub(1));
    frame.render_widget(
        Paragraph::new(text)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );
    if app.activity == Activity::Idle {
        frame.set_cursor_position((
            inner.x + cursor_col.min(inner.width.saturating_sub(1)),
            inner.y
                + cursor_row
                    .saturating_sub(scroll)
                    .min(inner.height.saturating_sub(1)),
        ));
    }
}

fn draw_help(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let text = format!(
        "  session {}  ·  Ollama {}  ·  model max {}  ·  PgUp/PgDn scroll",
        app.session.id, app.model_info.version, app.model_info.context_length
    );
    frame.render_widget(
        Paragraph::new(text).style(Style::default().fg(Color::Rgb(75, 80, 90))),
        area,
    );
}

fn draw_limit_dialog(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let width = area.width.min(62);
    let height = 7.min(area.height);
    let dialog = Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, dialog);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "上下文接近上限",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            Line::raw("继续会丢弃最旧的完整轮次；原始会话记录仍会保留。"),
            Line::raw("Enter / C：裁剪后继续    Esc / E：暂停会话"),
        ])
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .title(" Context budget ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Yellow)),
        ),
        dialog,
    );
}

fn input_height(text: &str, width: u16) -> u16 {
    let width = width.max(1) as usize;
    let rows = text
        .split('\n')
        .map(|line| display_width(line).div_ceil(width).max(1))
        .sum::<usize>();
    (rows as u16 + 2).clamp(4, 10)
}

fn wrap_plain(text: &str, width: usize) -> Vec<String> {
    let mut output = Vec::new();
    for source_line in text.split('\n') {
        if source_line.is_empty() {
            output.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut current_width = 0;
        for ch in source_line.chars() {
            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if current_width > 0 && current_width + ch_width > width {
                output.push(std::mem::take(&mut current));
                current_width = 0;
            }
            current.push(ch);
            current_width += ch_width;
        }
        output.push(current);
    }
    output
}

fn display_width(text: &str) -> usize {
    text.chars()
        .map(|ch| UnicodeWidthChar::width(ch).unwrap_or(0))
        .sum()
}

fn display_optional(value: Option<u64>) -> String {
    value.map_or_else(|| "未知".into(), |value| value.to_string())
}

#[derive(Default)]
struct InputEditor {
    text: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    draft: String,
}

impl InputEditor {
    fn insert_char(&mut self, ch: char) {
        self.text.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.history_index = None;
    }

    fn insert_str(&mut self, text: &str) {
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.history_index = None;
    }

    fn take(&mut self) -> String {
        self.cursor = 0;
        self.history_index = None;
        self.draft.clear();
        std::mem::take(&mut self.text)
    }

    fn remember(&mut self, text: String) {
        if self.history.last() != Some(&text) {
            self.history.push(text);
        }
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.history_index = None;
    }

    fn backspace(&mut self) {
        if let Some(previous) = previous_boundary(&self.text, self.cursor) {
            self.text.drain(previous..self.cursor);
            self.cursor = previous;
        }
    }

    fn delete(&mut self) {
        if let Some(next) = next_boundary(&self.text, self.cursor) {
            self.text.drain(self.cursor..next);
        }
    }

    fn move_left(&mut self) {
        if let Some(previous) = previous_boundary(&self.text, self.cursor) {
            self.cursor = previous;
        }
    }

    fn move_right(&mut self) {
        if let Some(next) = next_boundary(&self.text, self.cursor) {
            self.cursor = next;
        }
    }

    fn move_line_start(&mut self) {
        self.cursor = self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |index| index + 1);
    }

    fn move_line_end(&mut self) {
        self.cursor = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |offset| self.cursor + offset);
    }

    fn delete_previous_word(&mut self) {
        let prefix = &self.text[..self.cursor];
        let trimmed = prefix.trim_end_matches(char::is_whitespace);
        let start = trimmed.rfind(char::is_whitespace).map_or(0, |index| {
            index + trimmed[index..].chars().next().unwrap().len_utf8()
        });
        self.text.drain(start..self.cursor);
        self.cursor = start;
    }

    fn history_previous(&mut self) {
        if self.history.is_empty() || self.text.contains('\n') {
            return;
        }
        let index = match self.history_index {
            None => {
                self.draft = self.text.clone();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(index) => index - 1,
        };
        self.history_index = Some(index);
        self.text.clone_from(&self.history[index]);
        self.cursor = self.text.len();
    }

    fn move_up_or_history(&mut self) {
        let current_start = self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        if current_start == 0 {
            self.history_previous();
            return;
        }
        let column = self.text[current_start..self.cursor].chars().count();
        let previous_end = current_start - 1;
        let previous_start = self.text[..previous_end]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        self.cursor = byte_at_character_column(&self.text, previous_start, previous_end, column);
    }

    fn history_next(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.history.len() {
            self.history_index = Some(index + 1);
            self.text.clone_from(&self.history[index + 1]);
        } else {
            self.history_index = None;
            self.text.clone_from(&self.draft);
        }
        self.cursor = self.text.len();
    }

    fn move_down_or_history(&mut self) {
        let current_start = self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        let current_end = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |offset| self.cursor + offset);
        if current_end == self.text.len() {
            self.history_next();
            return;
        }
        let column = self.text[current_start..self.cursor].chars().count();
        let next_start = current_end + 1;
        let next_end = self.text[next_start..]
            .find('\n')
            .map_or(self.text.len(), |offset| next_start + offset);
        self.cursor = byte_at_character_column(&self.text, next_start, next_end, column);
    }

    fn cursor_position(&self, width: u16) -> (u16, u16) {
        let width = width.max(1) as usize;
        let mut row = 0_usize;
        let mut column = 0_usize;
        for ch in self.text[..self.cursor].chars() {
            if ch == '\n' {
                row += 1;
                column = 0;
                continue;
            }
            let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if column > 0 && column + char_width > width {
                row += 1;
                column = 0;
            }
            column += char_width;
            if column >= width {
                row += 1;
                column = 0;
            }
        }
        (
            row.min(u16::MAX as usize) as u16,
            column.min(u16::MAX as usize) as u16,
        )
    }
}

fn previous_boundary(text: &str, cursor: usize) -> Option<usize> {
    (cursor > 0).then(|| text[..cursor].char_indices().next_back().unwrap().0)
}

fn next_boundary(text: &str, cursor: usize) -> Option<usize> {
    (cursor < text.len()).then(|| cursor + text[cursor..].chars().next().unwrap().len_utf8())
}

fn byte_at_character_column(text: &str, start: usize, end: usize, column: usize) -> usize {
    text[start..end]
        .char_indices()
        .nth(column)
        .map_or(end, |(offset, _)| start + offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChatMessage, ContextPlan, Session, Turn};

    fn test_app() -> App {
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::SessionStore::new(root.path()).unwrap();
        let session = Session::new(
            "session-id".into(),
            "model-name".into(),
            "http://localhost:11434".into(),
            "system".into(),
            Default::default(),
            true,
        )
        .unwrap();
        let client = OllamaClient::new(&session.ollama_host).unwrap();
        App::new(
            ChatEngine::new(store, client),
            session,
            ModelInfo {
                version: "test".into(),
                name: "model-name".into(),
                context_length: 32_768,
            },
        )
    }

    fn session_with_turn(turn: Turn) -> Session {
        let mut session = Session::new(
            "session-id".into(),
            "model-name".into(),
            "http://localhost:11434".into(),
            "system".into(),
            Default::default(),
            true,
        )
        .unwrap();
        session.turns.push(turn);
        session
    }

    #[test]
    fn editor_handles_utf8_boundaries() {
        let mut editor = InputEditor::default();
        editor.insert_str("你a");
        editor.move_left();
        editor.backspace();
        assert_eq!(editor.text, "a");
        assert_eq!(editor.cursor, 0);
    }

    #[test]
    fn wrapping_respects_wide_characters() {
        assert_eq!(wrap_plain("你好abc", 4), vec!["你好", "abc"]);
    }

    #[test]
    fn multiline_arrows_preserve_character_column() {
        let mut editor = InputEditor::default();
        editor.insert_str("abcd\n你好吗\nxy");
        editor.cursor = "abcd\n你".len();
        editor.move_up_or_history();
        assert_eq!(editor.cursor, 1);
        editor.move_down_or_history();
        assert_eq!(editor.cursor, "abcd\n你".len());
    }

    #[test]
    fn debug_messages_are_hidden_until_enabled() {
        let messages = vec![
            Message {
                role: Role::User,
                content: "visible".into(),
            },
            Message {
                role: Role::Debug,
                content: "hidden".into(),
            },
        ];
        assert_eq!(visible_messages(&messages, false).count(), 1);
        assert_eq!(visible_messages(&messages, true).count(), 2);
    }

    #[test]
    fn live_thinking_is_cleared_when_answer_starts_and_stays_hidden() {
        let mut thinking = String::new();
        let mut answer = String::new();
        let mut answer_started = false;

        append_live_stream_event(
            &mut thinking,
            &mut answer,
            &mut answer_started,
            &ChatEvent::text(ChatEventKind::Thinking, "working through it".into(), 1),
        );
        assert_eq!(thinking, "working through it");

        append_live_stream_event(
            &mut thinking,
            &mut answer,
            &mut answer_started,
            &ChatEvent::text(ChatEventKind::Content, String::new(), 2),
        );
        assert!(answer_started);
        assert!(thinking.is_empty());
        assert!(answer.is_empty());

        append_live_stream_event(
            &mut thinking,
            &mut answer,
            &mut answer_started,
            &ChatEvent::text(ChatEventKind::Thinking, "late thought".into(), 3),
        );
        append_live_stream_event(
            &mut thinking,
            &mut answer,
            &mut answer_started,
            &ChatEvent::text(ChatEventKind::Content, "final answer".into(), 4),
        );
        assert!(thinking.is_empty());
        assert_eq!(answer, "final answer");
    }

    #[test]
    fn background_stream_is_batched_across_render_frames() {
        let mut app = test_app();
        for index in 0..=MAX_VISIBLE_STREAM_EVENTS_PER_FRAME {
            app.tx
                .send(BackgroundEvent::Stream(ChatEvent::text(
                    ChatEventKind::Content,
                    index.to_string(),
                    index as u64,
                )))
                .unwrap();
        }
        app.tx
            .send(BackgroundEvent::Finished {
                session: Box::new(app.session.clone()),
                result: Ok(()),
            })
            .unwrap();

        app.drain_background();
        assert_eq!(app.live_answer, "01234567");
        assert_eq!(app.activity, Activity::Generating);

        app.drain_background();
        assert!(app.live_answer.is_empty());
        assert_eq!(app.activity, Activity::Idle);
        assert!(
            app.messages.iter().any(|message| {
                message.role == Role::Assistant && message.content == "012345678"
            })
        );
    }

    #[test]
    fn session_messages_hide_thinking_when_answer_exists() {
        let mut turn = Turn::pending("question".into());
        turn.thinking = "private reasoning".into();
        turn.assistant_content = "answer".into();
        let session = session_with_turn(turn);

        let messages = messages_from_session(&session);

        assert!(
            messages
                .iter()
                .any(|message| { message.role == Role::Assistant && message.content == "answer" })
        );
        assert!(
            !messages
                .iter()
                .any(|message| message.role == Role::Thinking)
        );
        assert_eq!(session.turns[0].thinking, "private reasoning");
    }

    #[test]
    fn session_messages_keep_thinking_without_answer() {
        let mut turn = Turn::pending("question".into());
        turn.thinking = "unfinished reasoning".into();
        let session = session_with_turn(turn);

        let messages = messages_from_session(&session);

        assert!(messages.iter().any(|message| {
            message.role == Role::Thinking && message.content == "unfinished reasoning"
        }));
    }

    #[test]
    fn prepared_turn_formatter_includes_complete_messages_and_counts() {
        let session = Session::new(
            "session-id".into(),
            "model-name".into(),
            "http://localhost:11434".into(),
            "system".into(),
            Default::default(),
            true,
        )
        .unwrap();
        let prepared = crate::engine::PreparedTurn {
            session_id: session.id.clone(),
            turn_id: "turn-id".into(),
            turn_index: 0,
            plan: ContextPlan {
                messages: vec![
                    ChatMessage {
                        role: "system".into(),
                        content: "全量系统".into(),
                    },
                    ChatMessage {
                        role: "user".into(),
                        content: "完整用户输入".into(),
                    },
                ],
                context_items: Vec::new(),
                context_sha256: "abc123".into(),
                included_turn_ids: vec!["turn-a".into()],
                omitted_turn_ids: vec!["turn-b".into()],
                selected_history_indices: Vec::new(),
                estimated_upper_tokens: Some(42),
                exact_input_tokens: Some(40),
                input_budget: session.budget.input_budget(),
                identity_instruction: String::new(),
                untrusted_history_wrapped: false,
                retrieval_trace: Default::default(),
                evidence: Vec::new(),
                knowledge_trace: Default::default(),
            },
            status: PreparationStatus::Ready,
            message: String::new(),
        };
        let output = format_prepared_turn(&session, &prepared);
        assert!(output.contains("messages=2 per_role=[system=1, user=1]"));
        assert!(output.contains("完整用户输入"));
        assert!(output.contains("context_sha256=abc123"));
        assert!(output.contains("included_turns=1 ids=[turn-a]"));
    }

    #[test]
    fn message_window_reaches_rows_beyond_u16_scroll_limit() {
        let lines = (0..65_540)
            .map(|index| Line::raw(index.to_string()))
            .collect::<Vec<_>>();
        let window = message_window(&lines, 65_536, 3);
        assert_eq!(window.len(), 3);
        assert_eq!(window[0].to_string(), "65536");
        assert_eq!(window[2].to_string(), "65538");
    }

    #[test]
    fn ctrl_c_ends_a_pending_limit_instead_of_continuing() {
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(limit_action_for_key(key), Some(LimitAction::EndSession));
    }

    #[test]
    fn same_session_switch_reports_idle_status() {
        assert_eq!(same_session_status(), "已是当前会话。");
    }

    #[tokio::test]
    async fn only_explicit_idle_exit_intents_request_consolidation() {
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);

        let mut app = test_app();
        app.editor.insert_str("/exit");
        assert_eq!(
            handle_key(&mut app, enter).unwrap(),
            Some(TuiExitReason::ExitCommand)
        );

        let mut app = test_app();
        app.editor.insert_str("/exit extra");
        assert_eq!(handle_key(&mut app, enter).unwrap(), None);
        assert!(app.messages.iter().any(|message| {
            message.role == Role::Error && message.content.contains("未知命令")
        }));

        let mut app = test_app();
        assert_eq!(
            handle_key(&mut app, ctrl_c).unwrap(),
            Some(TuiExitReason::IdleCtrlC)
        );

        let mut app = test_app();
        let cancellation = CancellationToken::new();
        app.activity = Activity::Preparing;
        app.cancellation = Some(cancellation.clone());
        assert_eq!(handle_key(&mut app, ctrl_c).unwrap(), None);
        assert!(cancellation.is_cancelled());
        assert_eq!(app.activity, Activity::Cancelling);

        let mut app = test_app();
        let cancellation = CancellationToken::new();
        app.activity = Activity::Generating;
        app.cancellation = Some(cancellation.clone());
        assert_eq!(handle_key(&mut app, ctrl_c).unwrap(), None);
        assert!(cancellation.is_cancelled());
        assert_eq!(app.activity, Activity::Cancelling);

        let mut app = test_app();
        let (decision_tx, mut decision_rx) = mpsc::unbounded_channel();
        app.activity = Activity::AwaitingLimit;
        app.decision_tx = Some(decision_tx);
        assert_eq!(handle_key(&mut app, ctrl_c).unwrap(), None);
        assert_eq!(decision_rx.try_recv().unwrap(), LimitAction::EndSession);
        assert_eq!(app.activity, Activity::Preparing);

        let mut app = test_app();
        app.activity = Activity::Switching;
        app.switch_task = Some(tokio::spawn(async {
            std::future::pending::<()>().await;
        }));
        assert_eq!(handle_key(&mut app, ctrl_c).unwrap(), None);
        assert_eq!(app.activity, Activity::Idle);
        assert_eq!(app.status, "已取消会话切换。");

        let mut app = test_app();
        app.editor.insert_str("/session unknown-session");
        assert_eq!(handle_key(&mut app, enter).unwrap(), None);
        assert_eq!(app.activity, Activity::Switching);
        app.cancel_generation();
    }
}
