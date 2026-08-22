use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::model::{ChatEvent, ChatEventKind, ChatMessage, TokenUsage};

const MODEL_UNLOAD_TIMEOUT: Duration = Duration::from_secs(10);

type LoadedModels = BTreeMap<String, BTreeSet<String>>;

static LOADED_MODELS: OnceLock<Mutex<LoadedModels>> = OnceLock::new();

fn loaded_models() -> &'static Mutex<LoadedModels> {
    LOADED_MODELS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum OllamaError {
    #[error("无法连接 Ollama ({host}): {message}")]
    Connection { host: String, message: String },
    #[error("{0}")]
    ModelNotFound(String),
    #[error("{0}")]
    Protocol(String),
    #[error("{message}")]
    Stream {
        message: String,
        live_output_tokens: u64,
    },
    #[error("{message}")]
    ContextLength {
        message: String,
        prompt_tokens: Option<u64>,
        context_tokens: Option<u64>,
    },
    #[error("用户中断生成，未收到最终权威 token 计数")]
    Cancelled { live_output_tokens: u64 },
    #[error("{0}")]
    Other(String),
}

impl OllamaError {
    pub fn live_output_tokens(&self) -> Option<u64> {
        match self {
            Self::Stream {
                live_output_tokens, ..
            }
            | Self::Cancelled { live_output_tokens } => Some(*live_output_tokens),
            _ => None,
        }
    }

    pub fn is_transient(&self) -> bool {
        match self {
            Self::Connection { .. } | Self::Stream { .. } => true,
            Self::Other(message) => ["408", "429", "500", "502", "503", "504"]
                .iter()
                .any(|status| message.starts_with(&format!("Ollama HTTP {status}"))),
            Self::ModelNotFound(_)
            | Self::Protocol(_)
            | Self::ContextLength { .. }
            | Self::Cancelled { .. } => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub version: String,
    pub name: String,
    pub context_length: u64,
    pub supports_thinking: bool,
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub think: bool,
    pub num_ctx: u64,
    pub num_predict: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    pub dimensions: Option<usize>,
    pub truncate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_eval_count: Option<u64>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuredChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub schema: Value,
    pub think: bool,
    pub num_ctx: u64,
    pub num_predict: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuredChatResponse {
    pub content: String,
    pub usage: TokenUsage,
    pub done_reason: Option<String>,
}

impl StructuredChatResponse {
    pub fn output_limit_reached(&self, configured_limit: u64) -> bool {
        self.done_reason.as_deref() == Some("length")
            || self
                .usage
                .output_tokens
                .is_some_and(|tokens| tokens >= configured_limit)
    }
}

#[async_trait]
pub trait ChatBackend: Clone + Send + Sync + 'static {
    async fn check_model(
        &self,
        model: &str,
        requested_context: u64,
    ) -> Result<ModelInfo, OllamaError>;

    async fn render_prompt(
        &self,
        model: &str,
        messages: &[ChatMessage],
        think: bool,
        num_ctx: u64,
    ) -> Result<Option<String>, OllamaError>;

    async fn probe(
        &self,
        model: &str,
        messages: &[ChatMessage],
        think: bool,
        num_ctx: u64,
    ) -> Result<TokenUsage, OllamaError>;

    async fn stream_chat(
        &self,
        request: ChatRequest,
        cancellation: CancellationToken,
        emit: &mut (dyn FnMut(ChatEvent) + Send),
    ) -> Result<(), OllamaError>;

    async fn embed(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, OllamaError> {
        Err(OllamaError::Other("当前聊天后端不支持嵌入".into()))
    }

    async fn structured_chat(
        &self,
        _request: StructuredChatRequest,
    ) -> Result<StructuredChatResponse, OllamaError> {
        Err(OllamaError::Other("当前聊天后端不支持结构化输出".into()))
    }
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    host: String,
    client: Client,
}

impl OllamaClient {
    pub fn new(host: &str) -> Result<Self, OllamaError> {
        Self::with_timeout(host, Duration::from_secs(600))
    }

    pub fn with_timeout(host: &str, timeout: Duration) -> Result<Self, OllamaError> {
        let normalized = host.trim_end_matches('/');
        let parsed = Url::parse(normalized)
            .map_err(|_| OllamaError::Other(format!("无效 Ollama 地址: {host:?}")))?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(OllamaError::Other(format!("无效 Ollama 地址: {host:?}")));
        }
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|error| OllamaError::Other(error.to_string()))?;
        Ok(Self {
            host: normalized.to_owned(),
            client,
        })
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, OllamaError> {
        validate_embedding_request(&request)?;
        let payload = embedding_payload(&request);
        let response = self
            .request_json(reqwest::Method::POST, "/api/embed", Some(payload))
            .await?;
        self.record_model_use(&request.model);
        parse_embedding_response(response, &request)
    }

    pub async fn structured_chat(
        &self,
        request: StructuredChatRequest,
    ) -> Result<StructuredChatResponse, OllamaError> {
        validate_structured_chat_request(&request)?;
        let payload = structured_chat_payload(&request);
        self.request_structured_chat_response(&request.model, payload)
            .await
    }

    /// Unloads every model successfully used by this process from its Ollama host.
    pub async fn unload_tracked_models() -> Result<(), OllamaError> {
        let tracked = {
            let mut guard = loaded_models()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::take(&mut *guard)
        };
        let mut failures = Vec::new();
        for (host, models) in tracked {
            let client = match Self::with_timeout(&host, MODEL_UNLOAD_TIMEOUT) {
                Ok(client) => client,
                Err(error) => {
                    failures.push(format!("{host}: {error}"));
                    continue;
                }
            };
            for model in models {
                if let Err(error) = client.unload_model(&model).await {
                    failures.push(format!("{host} / {model}: {error}"));
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(OllamaError::Other(format!(
                "退出时未能卸载部分 Ollama 模型：{}",
                failures.join("；")
            )))
        }
    }

    async fn request_json(
        &self,
        method: reqwest::Method,
        path: &str,
        payload: Option<Value>,
    ) -> Result<Value, OllamaError> {
        let mut request = self
            .client
            .request(method, format!("{}{}", self.host, path))
            .header("Accept", "application/json");
        if let Some(payload) = payload {
            request = request.json(&payload);
        }
        let response = request
            .send()
            .await
            .map_err(|error| self.connection(error))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| self.connection(error))?;
        let payload: Value = serde_json::from_slice(&bytes)
            .map_err(|error| OllamaError::Protocol(format!("Ollama 返回了无效 JSON: {error}")))?;
        if !status.is_success() || payload.get("error").is_some() {
            return Err(api_error(&payload, Some(status)));
        }
        if !payload.is_object() {
            return Err(OllamaError::Protocol("Ollama 响应不是 JSON 对象".into()));
        }
        Ok(payload)
    }

    async fn request_chat_events(
        &self,
        model: &str,
        payload: Value,
    ) -> Result<Vec<Value>, OllamaError> {
        let response = self
            .client
            .post(format!("{}/api/chat", self.host))
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|error| self.connection(error))?;
        let status = response.status();
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|error| self.connection(error))?;
            let payload = serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| json!({"error": String::from_utf8_lossy(&bytes)}));
            return Err(api_error(&payload, Some(status)));
        }
        self.record_model_use(model);
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut events = Vec::new();
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(&chunk.map_err(|error| self.connection(error))?);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                if let Some(event) = parse_chat_event_line(&line[..line.len() - 1])? {
                    events.push(event);
                }
            }
        }
        if let Some(event) = parse_chat_event_line(&buffer)? {
            events.push(event);
        }
        Ok(events)
    }

    /// Streams a structured response while deliberately discarding every thinking fragment.
    /// Only final content and authoritative completion metadata survive this boundary.
    async fn request_structured_chat_response(
        &self,
        model: &str,
        payload: Value,
    ) -> Result<StructuredChatResponse, OllamaError> {
        let response = self
            .client
            .post(format!("{}/api/chat", self.host))
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|error| self.connection(error))?;
        let status = response.status();
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|error| self.connection(error))?;
            let payload = serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| json!({"error": String::from_utf8_lossy(&bytes)}));
            return Err(api_error(&payload, Some(status)));
        }
        self.record_model_use(model);
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut accumulator = StructuredResponseAccumulator::default();
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(&chunk.map_err(|error| self.connection(error))?);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                accumulator.push_line(&line[..line.len().saturating_sub(1)])?;
            }
        }
        accumulator.push_line(&buffer)?;
        accumulator.finish()
    }

    async fn unload_model(&self, model: &str) -> Result<(), OllamaError> {
        self.request_json(
            reqwest::Method::POST,
            "/api/generate",
            Some(model_unload_payload(model)),
        )
        .await?;
        Ok(())
    }

    fn record_model_use(&self, model: &str) {
        let mut guard = loaded_models()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .entry(self.host.clone())
            .or_default()
            .insert(model.to_owned());
    }

    fn connection(&self, error: reqwest::Error) -> OllamaError {
        OllamaError::Connection {
            host: self.host.clone(),
            message: error.to_string(),
        }
    }

    fn chat_payload(
        model: &str,
        messages: &[ChatMessage],
        think: bool,
        num_ctx: u64,
        num_predict: u64,
    ) -> Value {
        json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "think": think,
            "truncate": false,
            "shift": false,
            "keep_alive": "5m",
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            }
        })
    }
}

fn validate_embedding_request(request: &EmbeddingRequest) -> Result<(), OllamaError> {
    if request.model.trim().is_empty() {
        return Err(OllamaError::Other("嵌入模型不能为空".into()));
    }
    if request.input.is_empty() {
        return Err(OllamaError::Other("嵌入输入不能为空".into()));
    }
    if request.input.iter().any(|text| text.trim().is_empty()) {
        return Err(OllamaError::Other("嵌入输入不能包含空文本".into()));
    }
    if request.dimensions == Some(0) {
        return Err(OllamaError::Other("嵌入维度必须大于零".into()));
    }
    Ok(())
}

fn embedding_payload(request: &EmbeddingRequest) -> Value {
    let mut payload = json!({
        "model": request.model,
        "input": request.input,
        "truncate": request.truncate,
        "keep_alive": "5m",
    });
    if let Some(dimensions) = request.dimensions {
        payload
            .as_object_mut()
            .expect("embedding payload is an object")
            .insert("dimensions".into(), Value::from(dimensions));
    }
    payload
}

fn model_unload_payload(model: &str) -> Value {
    json!({"model": model, "keep_alive": 0})
}

fn parse_embedding_response(
    payload: Value,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, OllamaError> {
    let response: EmbeddingResponse = serde_json::from_value(payload)
        .map_err(|error| OllamaError::Protocol(format!("Ollama 嵌入响应结构无效: {error}")))?;
    validate_embedding_response(response, request)
}

fn validate_embedding_response(
    response: EmbeddingResponse,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, OllamaError> {
    if response.model.trim().is_empty() {
        return Err(OllamaError::Protocol("Ollama 嵌入响应缺少模型名".into()));
    }
    if response.model != request.model {
        return Err(OllamaError::Protocol(format!(
            "Ollama 嵌入响应模型 {:?} 与请求模型 {:?} 不一致",
            response.model, request.model
        )));
    }
    if response.embeddings.len() != request.input.len() {
        return Err(OllamaError::Protocol(format!(
            "Ollama 返回 {} 个向量，但请求包含 {} 个输入",
            response.embeddings.len(),
            request.input.len()
        )));
    }
    let Some(dimension) = response.embeddings.first().map(Vec::len) else {
        return Err(OllamaError::Protocol("Ollama 嵌入响应没有向量".into()));
    };
    if dimension == 0 {
        return Err(OllamaError::Protocol("Ollama 返回了空向量".into()));
    }
    if response
        .embeddings
        .iter()
        .any(|vector| vector.len() != dimension)
    {
        return Err(OllamaError::Protocol(
            "Ollama 返回的嵌入向量维度不一致".into(),
        ));
    }
    if request
        .dimensions
        .is_some_and(|expected| expected != dimension)
    {
        return Err(OllamaError::Protocol(format!(
            "Ollama 返回的嵌入维度 {dimension} 与请求不一致"
        )));
    }
    if response
        .embeddings
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(OllamaError::Protocol(
            "Ollama 返回的嵌入包含非有限值".into(),
        ));
    }
    Ok(response)
}

fn validate_structured_chat_request(request: &StructuredChatRequest) -> Result<(), OllamaError> {
    if request.model.trim().is_empty() {
        return Err(OllamaError::Other("结构化输出模型不能为空".into()));
    }
    if request.messages.is_empty() {
        return Err(OllamaError::Other("结构化输出消息不能为空".into()));
    }
    if !request.schema.is_object() {
        return Err(OllamaError::Other(
            "结构化输出 schema 必须是 JSON 对象".into(),
        ));
    }
    if request.num_ctx == 0 || request.num_predict == 0 {
        return Err(OllamaError::Other(
            "结构化输出 num_ctx 与 num_predict 必须大于零".into(),
        ));
    }
    Ok(())
}

fn structured_chat_payload(request: &StructuredChatRequest) -> Value {
    let messages = structured_chat_messages(request);
    json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
        "think": request.think,
        "format": request.schema,
        "truncate": false,
        "shift": false,
        "keep_alive": "5m",
        "options": {
            "num_ctx": request.num_ctx,
            "num_predict": request.num_predict,
            "temperature": 0,
            "seed": 0,
        }
    })
}

pub(crate) fn structured_chat_messages(request: &StructuredChatRequest) -> Vec<ChatMessage> {
    let schema = serde_json::to_string(&request.schema)
        .expect("serde_json::Value serialization is infallible");
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    messages.push(ChatMessage {
        role: "system".into(),
        content: format!(
            "Respond with exactly one JSON value matching the following JSON Schema. Do not include Markdown, code fences, or any extra text.\nJSON Schema:\n{schema}"
        ),
    });
    messages.extend(request.messages.iter().cloned());
    messages
}

fn parse_structured_chat_response(payload: &Value) -> Result<StructuredChatResponse, OllamaError> {
    let content = payload
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if payload.get("done").and_then(Value::as_bool) != Some(true) {
        return Err(OllamaError::Protocol(
            "Ollama 结构化响应缺少最终完成事件".into(),
        ));
    }
    let input = payload.get("prompt_eval_count").and_then(Value::as_u64);
    let output = payload.get("eval_count").and_then(Value::as_u64);
    Ok(StructuredChatResponse {
        content: content.to_owned(),
        usage: TokenUsage::new(input, output),
        done_reason: payload
            .get("done_reason")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

#[derive(Default)]
struct StructuredResponseAccumulator {
    content: String,
    final_event: Option<Value>,
}

impl StructuredResponseAccumulator {
    fn push_line(&mut self, line: &[u8]) -> Result<(), OllamaError> {
        let Some(event) = parse_chat_event_line(line)? else {
            return Ok(());
        };
        // Thinking is intentionally not cloned, concatenated, logged, or returned.
        if let Some(fragment) = event
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
        {
            self.content.push_str(fragment);
        }
        if event.get("done").and_then(Value::as_bool) == Some(true) {
            self.final_event = Some(event);
        }
        Ok(())
    }

    fn finish(self) -> Result<StructuredChatResponse, OllamaError> {
        let mut final_event = self
            .final_event
            .ok_or_else(|| OllamaError::Protocol("Ollama 结构化流在最终事件前结束".into()))?;
        final_event
            .as_object_mut()
            .expect("validated event object")
            .insert("message".into(), json!({"content": self.content}));
        parse_structured_chat_response(&final_event)
    }
}

#[cfg(test)]
fn parse_chat_event_bytes(bytes: &[u8]) -> Result<Vec<Value>, OllamaError> {
    let mut events = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        if let Some(event) = parse_chat_event_line(line)? {
            events.push(event);
        }
    }
    Ok(events)
}

fn parse_chat_event_line(line: &[u8]) -> Result<Option<Value>, OllamaError> {
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    let event: Value = serde_json::from_slice(line)
        .map_err(|error| OllamaError::Protocol(format!("Ollama 返回了无效流式 JSON: {error}")))?;
    if !event.is_object() {
        return Err(OllamaError::Protocol(
            "Ollama 流式响应事件不是 JSON 对象".into(),
        ));
    }
    if event.get("error").is_some() {
        return Err(api_error(&event, None));
    }
    Ok(Some(event))
}

#[cfg(test)]
fn parse_structured_chat_events(events: &[Value]) -> Result<StructuredChatResponse, OllamaError> {
    let final_event = events
        .last()
        .ok_or_else(|| OllamaError::Protocol("Ollama 结构化响应为空".into()))?;
    let mut content = String::new();
    for event in events {
        if let Some(fragment) = event
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
        {
            content.push_str(fragment);
        }
    }
    let mut aggregate = final_event.clone();
    aggregate
        .as_object_mut()
        .expect("validated event object")
        .insert("message".into(), json!({"content": content}));
    parse_structured_chat_response(&aggregate)
}

#[async_trait]
impl ChatBackend for OllamaClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, OllamaError> {
        OllamaClient::embed(self, request).await
    }

    async fn structured_chat(
        &self,
        request: StructuredChatRequest,
    ) -> Result<StructuredChatResponse, OllamaError> {
        OllamaClient::structured_chat(self, request).await
    }

    async fn check_model(
        &self,
        model: &str,
        requested_context: u64,
    ) -> Result<ModelInfo, OllamaError> {
        let version_payload = self
            .request_json(reqwest::Method::GET, "/api/version", None)
            .await?;
        let version = version_payload
            .get("version")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let tags = self
            .request_json(reqwest::Method::GET, "/api/tags", None)
            .await?;
        let available = tags
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| {
                item.get("name")
                    .or_else(|| item.get("model"))
                    .and_then(Value::as_str)
            })
            .any(|name| name == model);
        if !available {
            return Err(OllamaError::ModelNotFound(format!(
                "本地未安装模型 {model:?}；请先运行: ollama pull {model}"
            )));
        }
        let details = self
            .request_json(
                reqwest::Method::POST,
                "/api/show",
                Some(json!({"model": model})),
            )
            .await?;
        let context_length = extract_context_length(&details);
        if context_length == 0 {
            return Err(OllamaError::Protocol(format!(
                "Ollama 未返回模型 {model:?} 的上下文长度"
            )));
        }
        if requested_context > context_length {
            return Err(OllamaError::Other(format!(
                "配置的上下文 {requested_context} 超过模型上限 {context_length}"
            )));
        }
        Ok(ModelInfo {
            version,
            name: model.to_owned(),
            context_length,
            supports_thinking: details
                .get("capabilities")
                .and_then(Value::as_array)
                .is_some_and(|values| {
                    values
                        .iter()
                        .any(|value| value.as_str() == Some("thinking"))
                }),
        })
    }

    async fn render_prompt(
        &self,
        model: &str,
        messages: &[ChatMessage],
        think: bool,
        num_ctx: u64,
    ) -> Result<Option<String>, OllamaError> {
        let mut payload = Self::chat_payload(model, messages, think, num_ctx, 1);
        payload
            .as_object_mut()
            .expect("chat payload is an object")
            .insert("_debug_render_only".into(), Value::Bool(true));
        let events = self.request_chat_events(model, payload).await?;
        Ok(events.iter().rev().find_map(|response| {
            response
                .get("_debug_info")
                .and_then(|value| value.get("rendered_template"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        }))
    }

    async fn probe(
        &self,
        model: &str,
        messages: &[ChatMessage],
        think: bool,
        num_ctx: u64,
    ) -> Result<TokenUsage, OllamaError> {
        let mut payload = Self::chat_payload(model, messages, think, num_ctx, 1);
        let options = payload
            .get_mut("options")
            .and_then(Value::as_object_mut)
            .expect("chat options are an object");
        options.insert("temperature".into(), Value::from(0));
        options.insert("seed".into(), Value::from(0));
        let events = self.request_chat_events(model, payload).await?;
        let response = events
            .last()
            .ok_or_else(|| OllamaError::Protocol("精确探测响应为空".into()))?;
        let input = response.get("prompt_eval_count").and_then(Value::as_u64);
        let output = response.get("eval_count").and_then(Value::as_u64);
        match (input, output) {
            (Some(input), Some(output)) => Ok(TokenUsage::new(Some(input), Some(output))),
            _ => Err(OllamaError::Protocol(
                "精确探测响应缺少 prompt_eval_count 或 eval_count".into(),
            )),
        }
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
        cancellation: CancellationToken,
        emit: &mut (dyn FnMut(ChatEvent) + Send),
    ) -> Result<(), OllamaError> {
        let mut payload = Self::chat_payload(
            &request.model,
            &request.messages,
            request.think,
            request.num_ctx,
            request.num_predict,
        );
        let object = payload.as_object_mut().expect("chat payload is an object");
        object.insert("logprobs".into(), Value::Bool(true));
        object.insert("top_logprobs".into(), Value::from(0));

        let response = self
            .client
            .post(format!("{}/api/chat", self.host))
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|error| self.connection(error))?;
        let status = response.status();
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|error| self.connection(error))?;
            let payload = serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| json!({"error": String::from_utf8_lossy(&bytes)}));
            return Err(api_error(&payload, Some(status)));
        }
        self.record_model_use(&request.model);

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::<u8>::new();
        let mut live_output_tokens = 0_u64;
        let mut saw_done = false;

        loop {
            let next = tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(OllamaError::Cancelled { live_output_tokens });
                }
                value = stream.next() => value,
            };
            let Some(chunk) = next else { break };
            let chunk = chunk.map_err(|error| OllamaError::Stream {
                message: format!("Ollama 流读取失败: {error}"),
                live_output_tokens,
            })?;
            buffer.extend_from_slice(&chunk);

            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=newline).collect::<Vec<_>>();
                let line = &line[..line.len().saturating_sub(1)];
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                let events = parse_stream_line(line, &mut live_output_tokens)?;
                for event in events {
                    if event.kind == ChatEventKind::Completed {
                        saw_done = true;
                    }
                    emit(event);
                }
                if saw_done {
                    return Ok(());
                }
            }
        }

        if !buffer.iter().all(u8::is_ascii_whitespace) {
            let events = parse_stream_line(&buffer, &mut live_output_tokens)?;
            for event in events {
                if event.kind == ChatEventKind::Completed {
                    saw_done = true;
                }
                emit(event);
            }
        }
        if saw_done {
            Ok(())
        } else {
            Err(OllamaError::Stream {
                message: "Ollama 流在最终计数事件之前结束".into(),
                live_output_tokens,
            })
        }
    }
}

fn parse_stream_line(
    line: &[u8],
    live_output_tokens: &mut u64,
) -> Result<Vec<ChatEvent>, OllamaError> {
    let chunk: Value = serde_json::from_slice(line).map_err(|error| OllamaError::Stream {
        message: format!("Ollama 流返回了无效 JSON: {error}"),
        live_output_tokens: *live_output_tokens,
    })?;
    let object = chunk.as_object().ok_or_else(|| OllamaError::Stream {
        message: "Ollama 流事件不是 JSON 对象".into(),
        live_output_tokens: *live_output_tokens,
    })?;
    if object.contains_key("error") {
        return Err(api_error(&chunk, None));
    }
    let increment = object
        .get("logprobs")
        .and_then(Value::as_array)
        .map_or(0, |items| items.len() as u64);
    *live_output_tokens += increment;
    let mut events = Vec::new();
    if let Some(message) = object.get("message").and_then(Value::as_object) {
        if let Some(thinking) = nonempty_string(message, "thinking") {
            events.push(ChatEvent::text(
                ChatEventKind::Thinking,
                thinking.to_owned(),
                *live_output_tokens,
            ));
        }
        if let Some(content) = nonempty_string(message, "content") {
            events.push(ChatEvent::text(
                ChatEventKind::Content,
                content.to_owned(),
                *live_output_tokens,
            ));
        }
    }
    if increment > 0 {
        events.push(ChatEvent {
            kind: ChatEventKind::Usage,
            text: String::new(),
            live_output_tokens: Some(*live_output_tokens),
            usage: None,
            done_reason: None,
        });
    }
    if object.get("done").and_then(Value::as_bool) == Some(true) {
        let input = object.get("prompt_eval_count").and_then(Value::as_u64);
        let output = object.get("eval_count").and_then(Value::as_u64);
        let (Some(input), Some(output)) = (input, output) else {
            return Err(OllamaError::Stream {
                message: "Ollama 最终事件缺少精确 token 计数".into(),
                live_output_tokens: *live_output_tokens,
            });
        };
        events.push(ChatEvent {
            kind: ChatEventKind::Completed,
            text: String::new(),
            live_output_tokens: Some(*live_output_tokens),
            usage: Some(TokenUsage::new(Some(input), Some(output))),
            done_reason: object
                .get("done_reason")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        });
    }
    Ok(events)
}

fn nonempty_string<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn extract_context_length(payload: &Value) -> u64 {
    let direct = payload
        .get("details")
        .and_then(|value| value.get("context_length"))
        .and_then(Value::as_u64);
    if let Some(value) = direct {
        return value;
    }
    payload
        .get("model_info")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|object| object.iter())
        .filter(|(key, _)| key.ends_with(".context_length"))
        .filter_map(|(_, value)| value.as_u64())
        .max()
        .unwrap_or(0)
}

fn api_error(payload: &Value, status: Option<StatusCode>) -> OllamaError {
    let raw_error = payload.get("error").unwrap_or(payload);
    let mut message = match raw_error {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    let mut details = raw_error.as_object();
    let parsed_nested;
    if details.is_none() {
        parsed_nested = raw_error
            .as_str()
            .and_then(|text| serde_json::from_str::<Value>(text).ok());
        details = parsed_nested
            .as_ref()
            .and_then(|value| value.get("error").unwrap_or(value).as_object());
    }
    if let Some(details) = details
        && let Some(value) = details.get("message").and_then(Value::as_str)
    {
        message = value.to_owned();
    }
    let prompt_tokens = details
        .and_then(|value| value.get("n_prompt_tokens"))
        .and_then(Value::as_u64);
    let context_tokens = details
        .and_then(|value| value.get("n_ctx"))
        .and_then(Value::as_u64);
    let lowered = message.to_lowercase();
    if prompt_tokens.is_some()
        || lowered.contains("exceeds the available context")
        || lowered.contains("context length")
        || lowered.contains("prompt is too long")
    {
        return OllamaError::ContextLength {
            message,
            prompt_tokens,
            context_tokens,
        };
    }
    if status == Some(StatusCode::NOT_FOUND) || lowered.contains("not found") {
        return OllamaError::ModelNotFound(message);
    }
    let prefix = status.map_or_else(
        || "Ollama: ".to_owned(),
        |status| format!("Ollama HTTP {status}: "),
    );
    OllamaError::Other(prefix + &message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding_request() -> EmbeddingRequest {
        EmbeddingRequest {
            model: "qwen3-embedding:8b".into(),
            input: vec!["one".into(), "two".into()],
            dimensions: Some(3),
            truncate: true,
        }
    }

    fn structured_request() -> StructuredChatRequest {
        StructuredChatRequest {
            model: "qwen".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "extract".into(),
            }],
            schema: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            think: true,
            num_ctx: 4_096,
            num_predict: 256,
        }
    }

    #[test]
    fn stream_channels_and_authoritative_usage_are_separate() {
        let mut live = 0;
        let first = br#"{"message":{"thinking":"think"},"done":false,"logprobs":[{}]}"#;
        let events = parse_stream_line(first, &mut live).unwrap();
        assert_eq!(events[0].kind, ChatEventKind::Thinking);
        assert_eq!(live, 1);
        let final_line = br#"{"message":{"content":"answer"},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":4}"#;
        let events = parse_stream_line(final_line, &mut live).unwrap();
        assert_eq!(events[0].kind, ChatEventKind::Content);
        assert_eq!(events[1].usage.unwrap().total_tokens, Some(16));
    }

    #[test]
    fn nested_context_error_exposes_counts() {
        let nested = json!({
            "error": serde_json::to_string(&json!({"error": {
                "message": "request exceeds the available context size",
                "n_prompt_tokens": 6012,
                "n_ctx": 2048
            }})).unwrap()
        });
        assert!(matches!(
            api_error(&nested, Some(StatusCode::BAD_REQUEST)),
            OllamaError::ContextLength {
                prompt_tokens: Some(6012),
                context_tokens: Some(2048),
                ..
            }
        ));
    }

    #[test]
    fn transient_error_classification_is_allowlisted() {
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(OllamaError::Other(format!("Ollama HTTP {status}: retry")).is_transient());
        }
        assert!(
            OllamaError::Connection {
                host: "http://localhost".into(),
                message: "connection reset".into(),
            }
            .is_transient()
        );
        assert!(!OllamaError::Other("Ollama HTTP 400: invalid".into()).is_transient());
        assert!(!OllamaError::ModelNotFound("missing".into()).is_transient());
        assert!(
            !OllamaError::ContextLength {
                message: "too long".into(),
                prompt_tokens: Some(9),
                context_tokens: Some(8),
            }
            .is_transient()
        );
    }

    #[test]
    fn embedding_payload_and_parser_follow_the_batch_contract() {
        let request = embedding_request();
        let payload = embedding_payload(&request);
        assert_eq!(payload["model"], request.model);
        assert_eq!(payload["input"], json!(["one", "two"]));
        assert_eq!(payload["truncate"], true);
        assert_eq!(payload["keep_alive"], "5m");
        assert_eq!(payload["dimensions"], 3);

        let response = parse_embedding_response(
            json!({
                "model": "qwen3-embedding:8b",
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "prompt_eval_count": 4,
                "total_duration": 20,
                "load_duration": 2
            }),
            &request,
        )
        .unwrap();
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.prompt_eval_count, Some(4));

        let mut without_dimensions = request;
        without_dimensions.dimensions = None;
        assert!(
            embedding_payload(&without_dimensions)
                .get("dimensions")
                .is_none()
        );
    }

    #[test]
    fn embedding_contract_rejects_invalid_requests_and_responses() {
        let request = embedding_request();
        assert!(validate_embedding_request(&request).is_ok());
        for payload in [
            json!({"model":"qwen3-embedding:8b","embeddings":[[1.0, 2.0, 3.0]]}),
            json!({"model":"qwen3-embedding:8b","embeddings":[[1.0, 2.0], [3.0, 4.0]]}),
            json!({"model":"qwen3-embedding:8b","embeddings":[[1.0, 2.0, 3.0], [4.0, 5.0]]}),
            json!({"model":"qwen3-embedding:8b","embeddings":[[], []]}),
            json!({"model":"","embeddings":[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}),
            json!({"model":"   ","embeddings":[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}),
            json!({"model":"other","embeddings":[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}),
            json!({"model":"qwen3-embedding:8b ","embeddings":[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}),
        ] {
            assert!(parse_embedding_response(payload, &request).is_err());
        }

        let nonfinite = EmbeddingResponse {
            model: request.model.clone(),
            embeddings: vec![vec![1.0, f32::NAN, 3.0], vec![4.0, 5.0, 6.0]],
            prompt_eval_count: None,
            total_duration: None,
            load_duration: None,
        };
        assert!(validate_embedding_response(nonfinite, &request).is_err());

        let mut invalid = request.clone();
        invalid.model = " ".into();
        assert!(validate_embedding_request(&invalid).is_err());
        invalid = request.clone();
        invalid.input.clear();
        assert!(validate_embedding_request(&invalid).is_err());
        invalid = request.clone();
        invalid.input[1] = "\t".into();
        assert!(validate_embedding_request(&invalid).is_err());
        invalid = request;
        invalid.dimensions = Some(0);
        assert!(validate_embedding_request(&invalid).is_err());
    }

    #[test]
    fn structured_chat_payload_enables_requested_thinking_without_tools() {
        let request = structured_request();
        let payload = structured_chat_payload(&request);
        assert_eq!(payload["format"], request.schema);
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["think"], true);
        assert_eq!(payload["truncate"], false);
        assert_eq!(payload["shift"], false);
        assert_eq!(payload["keep_alive"], "5m");
        assert_eq!(payload["options"]["num_ctx"], 4_096);
        assert_eq!(payload["options"]["num_predict"], 256);
        assert_eq!(payload["options"]["temperature"], 0);
        assert_eq!(payload["options"]["seed"], 0);
        assert!(payload.get("tools").is_none());

        let response = parse_structured_chat_response(&json!({
            "message": {"content": "{\"name\":\"Ada\"}"},
            "prompt_eval_count": 12,
            "eval_count": 5,
            "done_reason": "stop",
            "done": true
        }))
        .unwrap();
        assert_eq!(response.content, "{\"name\":\"Ada\"}");
        assert_eq!(response.usage, TokenUsage::new(Some(12), Some(5)));
        assert_eq!(response.done_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn all_chat_payloads_force_streaming() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let payload = OllamaClient::chat_payload("model", &messages, false, 4096, 32);
        assert_eq!(payload["stream"], true);

        assert_eq!(
            structured_chat_payload(&structured_request())["stream"],
            true
        );
    }

    #[tokio::test]
    async fn unload_tracked_models_covers_successful_request_paths_and_hosts() {
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::response::{IntoResponse, Response};
        use axum::routing::post;
        use axum::{Json, Router};
        use std::sync::{Arc, Mutex};

        #[derive(Clone, Default)]
        struct MockState {
            chats: Arc<Mutex<Vec<Value>>>,
            embeds: Arc<Mutex<Vec<Value>>>,
            unloads: Arc<Mutex<Vec<Value>>>,
        }

        async fn chat(State(state): State<MockState>, Json(payload): Json<Value>) -> Response {
            state.chats.lock().unwrap().push(payload.clone());
            if payload["model"] == "failed-model" {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "model request failed"})),
                )
                    .into_response();
            }
            let response = json!({
                "message": {"content": "{\"name\":\"Ada\"}"},
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 5
            });
            (StatusCode::OK, format!("{response}\n")).into_response()
        }

        async fn embed(State(state): State<MockState>, Json(payload): Json<Value>) -> Json<Value> {
            state.embeds.lock().unwrap().push(payload.clone());
            Json(json!({
                "model": payload["model"],
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            }))
        }

        async fn unload(State(state): State<MockState>, Json(payload): Json<Value>) -> Json<Value> {
            state.unloads.lock().unwrap().push(payload);
            Json(json!({"done": true, "done_reason": "unload"}))
        }

        async fn spawn_mock() -> (String, MockState, tokio::task::JoinHandle<()>) {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let host = format!("http://{}", listener.local_addr().unwrap());
            let state = MockState::default();
            let app = Router::new()
                .route("/api/chat", post(chat))
                .route("/api/embed", post(embed))
                .route("/api/generate", post(unload))
                .with_state(state.clone());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            (host, state, server)
        }

        let (first_host, first_state, first_server) = spawn_mock().await;
        let (second_host, second_state, second_server) = spawn_mock().await;
        let first = OllamaClient::new(&first_host).unwrap();
        let second = OllamaClient::new(&second_host).unwrap();

        let mut shared = structured_request();
        shared.model = "shared-model".into();
        first.structured_chat(shared.clone()).await.unwrap();

        let chat_request = ChatRequest {
            model: "shared-model".into(),
            messages: shared.messages.clone(),
            think: false,
            num_ctx: 4_096,
            num_predict: 32,
        };
        ChatBackend::stream_chat(&first, chat_request, CancellationToken::new(), &mut |_| {})
            .await
            .unwrap();

        let mut failed = structured_request();
        failed.model = "failed-model".into();
        assert!(first.structured_chat(failed).await.is_err());

        second.embed(embedding_request()).await.unwrap();
        OllamaClient::unload_tracked_models().await.unwrap();

        let first_chats = first_state.chats.lock().unwrap().clone();
        assert_eq!(first_chats.len(), 3);
        assert_eq!(
            first_chats
                .iter()
                .filter(|payload| payload["model"] == "shared-model")
                .count(),
            2
        );
        assert!(
            first_chats
                .iter()
                .any(|payload| payload["model"] == "failed-model")
        );
        assert_eq!(second_state.embeds.lock().unwrap().len(), 1);

        assert_eq!(
            *first_state.unloads.lock().unwrap(),
            vec![json!({"model": "shared-model", "keep_alive": 0})]
        );
        assert_eq!(
            *second_state.unloads.lock().unwrap(),
            vec![json!({"model": "qwen3-embedding:8b", "keep_alive": 0})]
        );
        assert!(
            first_state
                .unloads
                .lock()
                .unwrap()
                .iter()
                .all(|payload| payload["model"] != "failed-model")
        );

        first_server.abort();
        second_server.abort();
    }

    #[test]
    fn streamed_structured_chat_concatenates_chunks_and_accepts_final_without_newline() {
        let bytes = br#"{"message":{"thinking":"SECRET_THINKING_ONE"},"done":false}
{"message":{"thinking":"SECRET_THINKING_TWO","content":""},"done":false}
{"message":{"content":"{\"name\":"},"done":false}
{"message":{"content":"\"Ada\"}"},"done":true,"prompt_eval_count":12,"eval_count":5,"done_reason":"stop"}"#;
        let events = parse_chat_event_bytes(bytes).unwrap();
        let response = parse_structured_chat_events(&events).unwrap();
        assert_eq!(response.content, "{\"name\":\"Ada\"}");
        assert_eq!(response.usage, TokenUsage::new(Some(12), Some(5)));
        assert_eq!(response.done_reason.as_deref(), Some("stop"));
        let audited = serde_json::to_string(&response).unwrap();
        assert!(!audited.contains("SECRET_THINKING_ONE"));
        assert!(!audited.contains("SECRET_THINKING_TWO"));

        let mut streamed = StructuredResponseAccumulator::default();
        for line in bytes.split(|byte| *byte == b'\n') {
            streamed.push_line(line).unwrap();
        }
        assert_eq!(streamed.finish().unwrap(), response);
    }

    #[test]
    fn structured_chat_rejects_invalid_requests_and_responses() {
        let request = structured_request();
        let empty = parse_structured_chat_response(&json!({
            "message":{"content":""},"prompt_eval_count":1,"eval_count":256,
            "done_reason":"length","done":true
        }))
        .unwrap();
        assert!(empty.content.is_empty());
        assert!(empty.output_limit_reached(256));
        let raw = parse_structured_chat_response(&json!({
            "message": {"content": "not json"},
            "prompt_eval_count": 1,
            "done": true
        }))
        .unwrap();
        assert_eq!(raw.content, "not json");
        assert_eq!(raw.usage, TokenUsage::new(Some(1), None));
        assert_eq!(
            parse_structured_chat_response(&json!({"message":{"content":"{}"},"done":true}))
                .unwrap()
                .usage,
            TokenUsage::new(None, None)
        );

        let mut invalid = request.clone();
        invalid.model.clear();
        assert!(validate_structured_chat_request(&invalid).is_err());
        invalid = request.clone();
        invalid.messages.clear();
        assert!(validate_structured_chat_request(&invalid).is_err());
        invalid = request.clone();
        invalid.schema = json!([]);
        assert!(validate_structured_chat_request(&invalid).is_err());
        invalid = request.clone();
        invalid.num_ctx = 0;
        assert!(validate_structured_chat_request(&invalid).is_err());
        invalid = request;
        invalid.num_predict = 0;
        assert!(validate_structured_chat_request(&invalid).is_err());
    }
}
