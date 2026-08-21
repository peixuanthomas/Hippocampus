use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::stream;
use pulldown_cmark::{Options, Parser, html};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify, mpsc};
use tokio_util::sync::CancellationToken;

use crate::engine::{ChatEngine, LimitAction, PreparationProgress, PreparationStatus};
use crate::knowledge::KnowledgeEvidence;
use crate::model::{ChatEvent, ChatEventKind, Session, TokenUsage, Turn};
use crate::ollama::{ChatBackend, ModelInfo, OllamaClient};

const INDEX_HTML: &str = include_str!("web/index.html");
const APP_CSS: &str = include_str!("web/app.css");
const APP_JS: &str = include_str!("web/app.js");
const MAX_MESSAGE_BYTES: usize = 1024 * 1024;

#[derive(Clone)]
struct WebState<B: ChatBackend> {
    engine: ChatEngine<B>,
    session: Arc<Mutex<Session>>,
    model_info: ModelInfo,
    runtime: Arc<Mutex<RuntimeState>>,
    generation_finished: Arc<Notify>,
}

#[derive(Default)]
struct RuntimeState {
    generation: Option<GenerationControl>,
}

struct GenerationControl {
    cancellation: CancellationToken,
    decision_tx: mpsc::UnboundedSender<LimitAction>,
}

#[derive(Debug, Serialize)]
struct SessionView {
    id: String,
    ai_name: String,
    title: String,
    status: String,
    model: String,
    ollama_version: String,
    model_context_length: u64,
    think: bool,
    busy: bool,
    budget: BudgetView,
    cumulative_usage: TokenUsage,
    cumulative_probe_usage: TokenUsage,
    turns: Vec<TurnView>,
}

#[derive(Debug, Serialize)]
struct BudgetView {
    context_window: u64,
    input_budget: u64,
    max_output_tokens: u64,
    safety_margin_tokens: u64,
    probe_threshold: u64,
    warning_threshold: u64,
    active_context_start_index: usize,
}

#[derive(Debug, Serialize)]
struct TurnView {
    id: String,
    status: String,
    user: String,
    assistant_markdown: String,
    assistant_html: String,
    thinking: String,
    usage: TokenUsage,
    probe_usage: TokenUsage,
    error: Option<String>,
    knowledge_sources: Vec<KnowledgeEvidence>,
    warnings: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ChatInput {
    message: String,
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DecisionInput {
    action: String,
}

#[derive(Debug, Deserialize)]
struct ThinkInput {
    enabled: bool,
}

#[derive(Debug, Serialize)]
struct ApiMessage {
    ok: bool,
    message: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiMessage {
                ok: false,
                message: self.message,
            }),
        )
            .into_response()
    }
}

pub async fn serve(
    engine: ChatEngine<OllamaClient>,
    session: Session,
    model_info: ModelInfo,
    address: SocketAddr,
) -> Result<()> {
    let state = WebState {
        engine,
        session: Arc::new(Mutex::new(session)),
        model_info,
        runtime: Arc::new(Mutex::new(RuntimeState::default())),
        generation_finished: Arc::new(Notify::new()),
    };
    let app = router(state.clone());
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("无法监听 {address}"))?;
    let actual = listener.local_addr()?;
    let browser_host = if actual.ip().is_unspecified() {
        "127.0.0.1".to_owned()
    } else {
        actual.ip().to_string()
    };
    println!(
        "Hippocampus Web UI：http://{browser_host}:{}",
        actual.port()
    );
    if !actual.ip().is_loopback() {
        eprintln!("警告：服务正在非回环地址监听，当前版本没有用户认证，请勿暴露到不可信网络。");
    }
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await;
    wait_for_idle(&state).await;
    serve_result.context("Web 服务异常退出")?;
    Ok(())
}

fn router<B: ChatBackend>(state: WebState<B>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app.css", get(styles))
        .route("/app.js", get(script))
        .route("/api/session", get(get_session))
        .route("/api/chat", post(chat))
        .route("/api/decision", post(decide))
        .route("/api/cancel", post(cancel))
        .route("/api/think", post(set_think))
        .route("/api/save", post(save))
        .route("/api/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_MESSAGE_BYTES))
        .with_state(state)
}

async fn index() -> Response {
    secured_static(Html(INDEX_HTML), "text/html; charset=utf-8")
}

async fn styles() -> Response {
    secured_static(APP_CSS, "text/css; charset=utf-8")
}

async fn script() -> Response {
    secured_static(APP_JS, "text/javascript; charset=utf-8")
}

fn secured_static(body: impl IntoResponse, content_type: &'static str) -> Response {
    let mut response = body.into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; img-src 'self' data: https: http:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        ),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    response
}

async fn health<B: ChatBackend>(State(state): State<WebState<B>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "ollama_version": state.model_info.version,
    }))
}

async fn get_session<B: ChatBackend>(State(state): State<WebState<B>>) -> Json<SessionView> {
    let session = state.session.lock().await;
    let busy = state.runtime.lock().await.generation.is_some();
    Json(session_view(&session, &state.model_info, busy))
}

async fn chat<B: ChatBackend>(
    State(state): State<WebState<B>>,
    Json(input): Json<ChatInput>,
) -> std::result::Result<Response, ApiError> {
    if input.message.trim().is_empty() {
        return Err(ApiError::bad_request("消息不能为空"));
    }
    let session_id = state.session.lock().await.id.clone();
    validate_session_id(input.session_id.as_deref(), &session_id)?;
    let (event_tx, event_rx) = mpsc::unbounded_channel::<Event>();
    send_event(&event_tx, "session", json!({ "session_id": session_id }));
    let (decision_tx, decision_rx) = mpsc::unbounded_channel();
    let cancellation = CancellationToken::new();
    {
        let mut runtime = state.runtime.lock().await;
        if runtime.generation.is_some() {
            return Err(ApiError::conflict("已有一轮正在生成，请先等待或取消"));
        }
        runtime.generation = Some(GenerationControl {
            cancellation: cancellation.clone(),
            decision_tx,
        });
    }
    tokio::spawn(run_generation(
        state,
        input.message,
        event_tx,
        decision_rx,
        cancellation,
    ));

    Ok(sse_response(event_rx, &session_id))
}

fn validate_session_id(supplied: Option<&str>, active: &str) -> std::result::Result<(), ApiError> {
    if supplied.is_some_and(|session_id| session_id != active) {
        return Err(ApiError::conflict("session_id 与服务当前活动会话不匹配"));
    }
    Ok(())
}

fn sse_response(event_rx: mpsc::UnboundedReceiver<Event>, session_id: &str) -> Response {
    let stream = stream::unfold(event_rx, |mut receiver| async move {
        receiver
            .recv()
            .await
            .map(|event| (Ok::<Event, Infallible>(event), receiver))
    });
    let mut response = Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keepalive"),
        )
        .into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-store, no-transform"),
    );
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    headers.insert(
        "x-hippocampus-session-id",
        HeaderValue::from_str(session_id).expect("session id is a valid header value"),
    );
    response
}

async fn run_generation<B: ChatBackend>(
    state: WebState<B>,
    message: String,
    event_tx: mpsc::UnboundedSender<Event>,
    mut decision_rx: mpsc::UnboundedReceiver<LimitAction>,
    cancellation: CancellationToken,
) {
    let result = generate(&state, message, &event_tx, &mut decision_rx, cancellation).await;
    if let Err(error) = result {
        send_event(
            &event_tx,
            "error",
            json!({ "message": format!("{error:#}") }),
        );
    }
    state.runtime.lock().await.generation = None;
    state.generation_finished.notify_waiters();
}

async fn generate<B: ChatBackend>(
    state: &WebState<B>,
    message: String,
    event_tx: &mpsc::UnboundedSender<Event>,
    decision_rx: &mut mpsc::UnboundedReceiver<LimitAction>,
    cancellation: CancellationToken,
) -> Result<()> {
    let mut session = state.session.lock().await;
    send_event(event_tx, "status", json!({ "message": "正在组装上下文…" }));
    let progress_tx = event_tx.clone();
    let mut prepared = state
        .engine
        .prepare_turn_with_progress(&mut session, message, move |progress| match progress {
            PreparationProgress::ExactContextCheckStarted { .. } => send_event(
                &progress_tx,
                "status",
                json!({ "message": "上下文接近上限，正在进行一次精确检查…" }),
            ),
        })
        .await?;
    if prepared.needs_limit_decision() {
        send_event(event_tx, "limit", json!({ "message": prepared.message }));
        let action = tokio::select! {
            _ = cancellation.cancelled() => LimitAction::EndSession,
            action = decision_rx.recv() => action.unwrap_or(LimitAction::EndSession),
        };
        prepared = state
            .engine
            .resolve_limit(&mut session, prepared, action)
            .await?;
        if !prepared.message.is_empty() {
            send_event(event_tx, "status", json!({ "message": prepared.message }));
        }
    }
    if matches!(
        prepared.status,
        PreparationStatus::Blocked | PreparationStatus::Ended
    ) {
        bail!(prepared.message);
    }
    let input_tokens = prepared
        .plan
        .exact_input_tokens
        .or(prepared.plan.estimated_upper_tokens);
    send_event(
        event_tx,
        "prepared",
        json!({
            "input_tokens": input_tokens,
            "input_budget": prepared.plan.input_budget,
            "exact": prepared.plan.exact_input_tokens.is_some(),
            "included": prepared.plan.included_turn_ids.len(),
            "omitted": prepared.plan.omitted_turn_ids.len(),
            "search_elapsed_ms": prepared.plan.retrieval_trace.elapsed_ms,
            "search_deadline_ms": prepared.plan.retrieval_trace.deadline_ms,
            "search_timed_out": prepared.plan.retrieval_trace.deadline_exceeded,
            "fast_fallback_used": prepared.plan.retrieval_trace.fast_fallback_used,
            "fallback_reason": prepared.plan.retrieval_trace.fallback_reason,
            "search_channels": prepared.plan.retrieval_trace.channels,
            "search_debug": prepared.plan.retrieval_trace.warnings,
        }),
    );
    let stream_tx = event_tx.clone();
    state
        .engine
        .stream_turn(&mut session, &prepared, cancellation, move |event| {
            send_chat_event(&stream_tx, event);
        })
        .await?;
    let turn = session
        .turns
        .get(prepared.turn_index)
        .ok_or_else(|| anyhow!("生成完成但会话中没有对应轮次"))?;
    send_event(
        event_tx,
        "done",
        json!({
            "session_id": session.id,
            "turn_id": turn.id,
            "status": turn.status.as_str(),
            "markdown": turn.assistant_content,
            "html": render_markdown(&turn.assistant_content),
            "usage": turn.usage,
            "error": turn.error,
            "knowledge_sources": turn.context_trace.knowledge.selected_evidence,
            "warnings": turn_warnings(turn),
            "session_status": session.status.as_str(),
            "title": session.title,
        }),
    );
    Ok(())
}

fn send_chat_event(sender: &mpsc::UnboundedSender<Event>, event: ChatEvent) {
    match event.kind {
        ChatEventKind::Thinking => send_event(
            sender,
            "thinking",
            json!({ "text": event.text, "live_tokens": event.live_output_tokens }),
        ),
        ChatEventKind::Content => send_event(
            sender,
            "content",
            json!({ "text": event.text, "live_tokens": event.live_output_tokens }),
        ),
        ChatEventKind::Usage => send_event(
            sender,
            "usage",
            json!({ "live_tokens": event.live_output_tokens }),
        ),
        ChatEventKind::Completed => send_event(
            sender,
            "completed",
            json!({
                "live_tokens": event.live_output_tokens,
                "usage": event.usage,
                "done_reason": event.done_reason,
            }),
        ),
    }
}

fn send_event(sender: &mpsc::UnboundedSender<Event>, name: &'static str, data: Value) {
    if let Ok(event) = Event::default().event(name).json_data(data) {
        let _ = sender.send(event);
    }
}

async fn decide<B: ChatBackend>(
    State(state): State<WebState<B>>,
    Json(input): Json<DecisionInput>,
) -> std::result::Result<Json<ApiMessage>, ApiError> {
    let action = match input.action.as_str() {
        "continue" => LimitAction::ContinueWithTrim,
        "end" => LimitAction::EndSession,
        _ => return Err(ApiError::bad_request("action 必须是 continue 或 end")),
    };
    let runtime = state.runtime.lock().await;
    let control = runtime
        .generation
        .as_ref()
        .ok_or_else(|| ApiError::conflict("当前没有等待处理的生成任务"))?;
    control
        .decision_tx
        .send(action)
        .map_err(|_| ApiError::conflict("上下文决策通道已经关闭"))?;
    Ok(Json(ApiMessage {
        ok: true,
        message: "决策已提交".into(),
    }))
}

async fn cancel<B: ChatBackend>(
    State(state): State<WebState<B>>,
) -> std::result::Result<Json<ApiMessage>, ApiError> {
    let runtime = state.runtime.lock().await;
    let control = runtime
        .generation
        .as_ref()
        .ok_or_else(|| ApiError::conflict("当前没有正在生成的任务"))?;
    control.cancellation.cancel();
    Ok(Json(ApiMessage {
        ok: true,
        message: "正在中断并保存已收到的内容".into(),
    }))
}

async fn set_think<B: ChatBackend>(
    State(state): State<WebState<B>>,
    Json(input): Json<ThinkInput>,
) -> std::result::Result<Json<ApiMessage>, ApiError> {
    ensure_idle(&state).await?;
    let mut session = state.session.lock().await;
    session.think = input.enabled;
    state
        .engine
        .store()
        .save(&mut session)
        .map_err(internal_api_error)?;
    Ok(Json(ApiMessage {
        ok: true,
        message: format!(
            "thinking 已设为：{}",
            if input.enabled { "on" } else { "off" }
        ),
    }))
}

async fn save<B: ChatBackend>(
    State(state): State<WebState<B>>,
) -> std::result::Result<Json<ApiMessage>, ApiError> {
    ensure_idle(&state).await?;
    let mut session = state.session.lock().await;
    let path = state
        .engine
        .store()
        .save(&mut session)
        .map_err(internal_api_error)?;
    Ok(Json(ApiMessage {
        ok: true,
        message: format!("已原子保存：{}", path.display()),
    }))
}

async fn ensure_idle<B: ChatBackend>(state: &WebState<B>) -> std::result::Result<(), ApiError> {
    if state.runtime.lock().await.generation.is_some() {
        Err(ApiError::conflict("生成期间不能修改会话设置"))
    } else {
        Ok(())
    }
}

async fn wait_for_idle<B: ChatBackend>(state: &WebState<B>) {
    loop {
        let finished = state.generation_finished.notified();
        if state.runtime.lock().await.generation.is_none() {
            return;
        }
        finished.await;
    }
}

fn internal_api_error(error: anyhow::Error) -> ApiError {
    ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("{error:#}"),
    }
}

fn session_view(session: &Session, model_info: &ModelInfo, busy: bool) -> SessionView {
    let budget = &session.budget;
    SessionView {
        id: session.id.clone(),
        ai_name: session.ai_name.clone(),
        title: session.title.clone(),
        status: session.status.as_str().into(),
        model: session.model.clone(),
        ollama_version: model_info.version.clone(),
        model_context_length: model_info.context_length,
        think: session.think,
        busy,
        budget: BudgetView {
            context_window: budget.context_window,
            input_budget: budget.input_budget(),
            max_output_tokens: budget.max_output_tokens,
            safety_margin_tokens: budget.safety_margin_tokens,
            probe_threshold: budget.probe_threshold(),
            warning_threshold: budget.warning_threshold(),
            active_context_start_index: session.active_context_start_index,
        },
        cumulative_usage: session.cumulative_usage,
        cumulative_probe_usage: session.cumulative_probe_usage,
        turns: session
            .turns
            .iter()
            .map(|turn| TurnView {
                id: turn.id.clone(),
                status: turn.status.as_str().into(),
                user: turn.user_content.clone(),
                assistant_markdown: turn.assistant_content.clone(),
                assistant_html: render_markdown(&turn.assistant_content),
                thinking: turn.thinking.clone(),
                usage: turn.usage,
                probe_usage: turn.probe_usage,
                error: turn.error.clone(),
                knowledge_sources: turn.context_trace.knowledge.selected_evidence.clone(),
                warnings: turn_warnings(turn),
            })
            .collect(),
    }
}

fn turn_warnings(turn: &Turn) -> Vec<String> {
    let mut warnings = turn.context_trace.knowledge.warnings.clone();
    warnings.sort();
    warnings.dedup();
    warnings
}

pub fn render_markdown(markdown: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_TASKLISTS);
    options.insert(Options::ENABLE_FOOTNOTES);
    let parser = Parser::new_ext(markdown, options);
    let mut rendered = String::new();
    html::push_html(&mut rendered, parser);
    ammonia::Builder::default().clean(&rendered).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::model::ChatEvent;
    use crate::ollama::{ChatRequest, OllamaError};

    fn test_state() -> WebState<OllamaClient> {
        let root = tempfile::tempdir().unwrap().keep();
        let store = crate::store::SessionStore::new(&root).unwrap();
        let session = store
            .create(
                "model",
                "http://127.0.0.1:11434",
                None,
                Default::default(),
                false,
            )
            .unwrap();
        let client = OllamaClient::new("http://127.0.0.1:11434").unwrap();
        WebState {
            engine: ChatEngine::new(store, client),
            session: Arc::new(Mutex::new(session)),
            model_info: ModelInfo {
                version: "test".into(),
                name: "model".into(),
                context_length: 32_768,
            },
            runtime: Arc::new(Mutex::new(RuntimeState::default())),
            generation_finished: Arc::new(Notify::new()),
        }
    }

    #[derive(Clone)]
    struct ThreeRoundClient;

    #[async_trait]
    impl ChatBackend for ThreeRoundClient {
        async fn check_model(&self, model: &str, _: u64) -> Result<ModelInfo, OllamaError> {
            Ok(ModelInfo {
                version: "test".into(),
                name: model.into(),
                context_length: 32_768,
            })
        }

        async fn render_prompt(
            &self,
            _: &str,
            _: &[crate::model::ChatMessage],
            _: bool,
            _: u64,
        ) -> Result<Option<String>, OllamaError> {
            Ok(None)
        }

        async fn probe(
            &self,
            _: &str,
            _: &[crate::model::ChatMessage],
            _: bool,
            _: u64,
        ) -> Result<TokenUsage, OllamaError> {
            Ok(TokenUsage::new(Some(100), Some(1)))
        }

        async fn stream_chat(
            &self,
            _: ChatRequest,
            _: CancellationToken,
            emit: &mut (dyn FnMut(ChatEvent) + Send),
        ) -> Result<(), OllamaError> {
            emit(ChatEvent::text(ChatEventKind::Content, "answer".into(), 1));
            emit(ChatEvent {
                kind: ChatEventKind::Completed,
                text: String::new(),
                live_output_tokens: Some(1),
                usage: Some(TokenUsage::new(Some(100), Some(1))),
                done_reason: Some("stop".into()),
            });
            Ok(())
        }
    }

    fn three_round_state() -> WebState<ThreeRoundClient> {
        let root = tempfile::tempdir().unwrap().keep();
        let store = crate::store::SessionStore::new(&root).unwrap();
        let session = store
            .create(
                "model",
                "http://127.0.0.1:11434",
                None,
                Default::default(),
                false,
            )
            .unwrap();
        WebState {
            engine: ChatEngine::new(store, ThreeRoundClient),
            session: Arc::new(Mutex::new(session)),
            model_info: ModelInfo {
                version: "test".into(),
                name: "model".into(),
                context_length: 32_768,
            },
            runtime: Arc::new(Mutex::new(RuntimeState::default())),
            generation_finished: Arc::new(Notify::new()),
        }
    }

    #[test]
    fn markdown_is_rich_and_sanitized() {
        let rendered = render_markdown(
            "# 标题\n\n- **粗体**\n\n```rust\nfn main() {}\n```\n\n<script>alert(1)</script>",
        );
        assert!(rendered.contains("<h1>标题</h1>"));
        assert!(rendered.contains("<strong>粗体</strong>"));
        assert!(rendered.contains("<pre><code>"));
        assert!(!rendered.contains("<script>"));
    }

    #[tokio::test]
    async fn static_page_has_security_headers_and_local_assets() {
        let response = index().await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response
                .headers()
                .contains_key(header::CONTENT_SECURITY_POLICY)
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let html = String::from_utf8(body.to_vec()).unwrap();
        assert!(html.contains("/app.css"));
        assert!(html.contains("/app.js"));
        assert!(!html.contains("https://cdn"));
    }

    #[tokio::test]
    async fn health_route_is_json() {
        let response = router(test_state())
            .oneshot(
                http::Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
    }

    #[tokio::test]
    async fn chat_rejects_mismatched_session_without_starting_generation() {
        let state = test_state();
        let response = router(state.clone())
            .oneshot(
                http::Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"message":"hello","session_id":"wrong"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert!(state.runtime.lock().await.generation.is_none());
    }

    #[tokio::test]
    async fn chat_stream_has_session_event_and_no_buffering_headers() {
        let (sender, receiver) = mpsc::unbounded_channel();
        send_event(&sender, "session", json!({"session_id": "session-1"}));
        drop(sender);
        let response = sse_response(receiver, "session-1");
        assert_eq!(
            response.headers()[header::CACHE_CONTROL],
            "no-cache, no-store, no-transform"
        );
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        assert_eq!(response.headers()["x-hippocampus-session-id"], "session-1");
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.starts_with("event: session\n"));
        assert!(text.contains(r#"data: {"session_id":"session-1"}"#));
    }

    #[tokio::test]
    async fn graceful_shutdown_waits_for_active_generation() {
        let state = test_state();
        let (decision_tx, _decision_rx) = mpsc::unbounded_channel();
        state.runtime.lock().await.generation = Some(GenerationControl {
            cancellation: CancellationToken::new(),
            decision_tx,
        });
        let wait_state = state.clone();
        let mut waiter = tokio::spawn(async move { wait_for_idle(&wait_state).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut waiter)
                .await
                .is_err()
        );

        state.runtime.lock().await.generation = None;
        state.generation_finished.notify_waiters();
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn three_consecutive_api_rounds_keep_terminal_context_traces_frozen() {
        let state = three_round_state();
        let session_id = state.session.lock().await.id.clone();
        for round in 1..=3 {
            let response = router(state.clone())
                .oneshot(
                    http::Request::builder()
                        .method("POST")
                        .uri("/api/chat")
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(format!(
                            r#"{{"message":"round {round}","session_id":"{session_id}"}}"#
                        )))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            let stream = String::from_utf8(body.to_vec()).unwrap();
            assert!(!stream.contains("event: error\n"), "{stream}");
            assert!(stream.contains("event: prepared\n"), "{stream}");
            assert!(stream.contains("\"search_deadline_ms\":15000"), "{stream}");
            assert!(stream.contains("\"search_timed_out\":false"), "{stream}");
            assert!(stream.contains("event: done\n"), "{stream}");
        }

        wait_for_idle(&state).await;
        let in_memory = state.session.lock().await.clone();
        let persisted = state.engine.store().load(&session_id).unwrap();
        assert_eq!(in_memory.turns.len(), 3);
        assert!(
            in_memory
                .turns
                .iter()
                .all(|turn| turn.status == crate::model::TurnStatus::Complete)
        );
        assert_eq!(
            in_memory
                .turns
                .iter()
                .map(|turn| &turn.context_trace)
                .collect::<Vec<_>>(),
            persisted
                .turns
                .iter()
                .map(|turn| &turn.context_trace)
                .collect::<Vec<_>>()
        );
    }
}
