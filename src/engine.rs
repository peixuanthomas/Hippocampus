use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use tokio_util::sync::CancellationToken;

use crate::context::ContextAssembler;
use crate::model::{
    ChatEvent, ChatEventKind, ContextPlan, ContextTrace, ModelRequestTrace, ProvenanceQuality,
    Session, SessionStatus, TokenUsage, Turn, TurnStatus, utc_now,
};
use crate::ollama::{ChatBackend, ChatRequest, OllamaError};
use crate::retrieval::{RecallResult, RecalledEvidence};
use crate::store::SessionStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitAction {
    ContinueWithTrim,
    EndSession,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparationStatus {
    Ready,
    LimitWarning,
    Blocked,
    Ended,
}

#[derive(Debug, Clone)]
pub struct PreparedTurn {
    pub session_id: String,
    pub turn_id: String,
    pub turn_index: usize,
    pub plan: ContextPlan,
    pub status: PreparationStatus,
    pub message: String,
}

impl PreparedTurn {
    pub fn ready(&self) -> bool {
        self.status == PreparationStatus::Ready
    }

    pub fn needs_limit_decision(&self) -> bool {
        self.status == PreparationStatus::LimitWarning
    }
}

#[derive(Debug, Clone)]
pub struct ChatEngine<B: ChatBackend> {
    store: SessionStore,
    client: B,
    assembler: ContextAssembler,
}

struct StreamSnapshot<'a> {
    thinking: &'a str,
    content: &'a str,
    live_output_tokens: u64,
    final_usage: Option<TokenUsage>,
}

impl<B: ChatBackend> ChatEngine<B> {
    pub fn new(store: SessionStore, client: B) -> Self {
        Self {
            store,
            client,
            assembler: ContextAssembler,
        }
    }

    pub fn store(&self) -> &SessionStore {
        &self.store
    }

    pub fn client(&self) -> &B {
        &self.client
    }

    pub async fn prepare_turn(
        &self,
        session: &mut Session,
        user_content: String,
    ) -> Result<PreparedTurn> {
        if user_content.trim().is_empty() {
            bail!("用户输入不能为空");
        }
        self.recover_stale_pending(session)?;
        session.status = SessionStatus::Active;
        if session.turns.is_empty() {
            let compact = user_content
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let mut title = compact.chars().take(40).collect::<String>();
            if compact.chars().count() > 40 {
                title.push('…');
            }
            session.title = title;
        }

        let start_before = session.active_context_start_index;
        session.turns.push(Turn::pending(user_content.clone()));
        let turn_index = session.turns.len() - 1;
        self.store.save(session)?;

        let prepared = self
            .prepare_persisted_turn(session, turn_index, user_content, start_before)
            .await;
        if let Err(error) = &prepared
            && session.turns[turn_index].status == TurnStatus::Pending
        {
            let turn = &mut session.turns[turn_index];
            turn.status = TurnStatus::Failed;
            turn.error = Some(error.to_string());
            turn.touch();
            self.store.save(session)?;
        }
        prepared
    }

    async fn prepare_persisted_turn(
        &self,
        session: &mut Session,
        turn_index: usize,
        user_content: String,
        start_before: usize,
    ) -> Result<PreparedTurn> {
        let history = session
            .eligible_turns(Some(turn_index), true)
            .into_iter()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let current_event_id = crate::model::event_id(
            &session.id,
            Some(&session.turns[turn_index].id),
            crate::model::EventRole::User,
        );
        let recent_event_ids = history
            .iter()
            .flat_map(|index| {
                let turn = &session.turns[*index];
                [
                    crate::model::event_id(
                        &session.id,
                        Some(&turn.id),
                        crate::model::EventRole::User,
                    ),
                    crate::model::event_id(
                        &session.id,
                        Some(&turn.id),
                        crate::model::EventRole::Assistant,
                    ),
                ]
            })
            .collect::<Vec<_>>();
        let recall = self
            .store
            .retrieval()
            .keyword_recall(
                &user_content,
                &current_event_id,
                &recent_event_ids,
                session.retrieval.clone(),
            )
            .inspect_err(|error| {
                session.turns[turn_index].context_trace.retrieval = crate::model::RetrievalTrace {
                    status: "failed".into(),
                    current_query_event_id: current_event_id.clone(),
                    error: Some(error.to_string()),
                    config: session.retrieval.clone(),
                    ..Default::default()
                };
            })?;
        let (mut plan, render_supported) = self
            .build_plan(session, turn_index, &history, &user_content, &recall)
            .await?;
        if !render_supported
            || plan
                .estimated_upper_tokens
                .is_some_and(|tokens| tokens >= session.budget.probe_threshold())
        {
            self.probe_plan(session, turn_index, &mut plan).await?;
        }

        if plan_metric(&plan)? >= session.budget.warning_threshold() {
            let (mut mandatory, _) = self
                .build_plan(session, turn_index, &[], &user_content, &recall)
                .await?;
            self.probe_plan(session, turn_index, &mut mandatory).await?;
            if plan_metric(&mandatory)? > session.budget.input_budget() {
                return self.block_mandatory(session, turn_index, mandatory, start_before);
            }
            apply_trace(
                session,
                turn_index,
                &plan,
                "limit_warning",
                start_before,
                start_before,
            );
            session.turns[turn_index].touch();
            self.store.save(session)?;
            return Ok(self.prepared(
                session,
                turn_index,
                plan,
                PreparationStatus::LimitWarning,
                "上下文已达到临界阈值；请选择丢弃最旧完整轮次后继续，或暂停当前会话。",
            ));
        }

        apply_trace(
            session,
            turn_index,
            &plan,
            "ready",
            start_before,
            start_before,
        );
        session.turns[turn_index].touch();
        self.store.save(session)?;
        Ok(self.prepared(session, turn_index, plan, PreparationStatus::Ready, ""))
    }

    pub async fn resolve_limit(
        &self,
        session: &mut Session,
        prepared: PreparedTurn,
        action: LimitAction,
    ) -> Result<PreparedTurn> {
        if !prepared.needs_limit_decision() {
            bail!("当前轮次不需要上下文临界决策");
        }
        self.pending_turn(session, &prepared)?;
        let start_before = session.active_context_start_index;

        if action == LimitAction::EndSession {
            let turn = &mut session.turns[prepared.turn_index];
            turn.status = TurnStatus::Blocked;
            turn.error = Some("用户选择在上下文临界点暂停会话；消息未发送给模型".into());
            turn.context_trace.decision = "paused_by_user".into();
            turn.touch();
            session.status = SessionStatus::Paused;
            self.store.save(session)?;
            let mut ended = prepared;
            ended.status = PreparationStatus::Ended;
            ended.message = session.turns[ended.turn_index]
                .error
                .clone()
                .unwrap_or_default();
            return Ok(ended);
        }

        let history = session
            .eligible_turns(Some(prepared.turn_index), true)
            .into_iter()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let mut cache = HashMap::from([(history.len(), prepared.plan.clone())]);
        let user_content = session.turns[prepared.turn_index].user_content.clone();
        let mut mandatory = self.assembler.assemble_with_recall(
            session,
            &user_content,
            Some(&[]),
            Some(prepared.turn_index),
            Some(&self.recall_from_plan(&prepared.plan)?),
        );
        self.probe_plan(session, prepared.turn_index, &mut mandatory)
            .await?;
        cache.insert(0, mandatory.clone());
        let mandatory_tokens = plan_metric(&mandatory)?;
        if mandatory_tokens > session.budget.input_budget() {
            return self.block_mandatory(session, prepared.turn_index, mandatory, start_before);
        }
        if mandatory_tokens > session.budget.trim_target() {
            apply_trace(
                session,
                prepared.turn_index,
                &mandatory,
                "mandatory_above_trim_target",
                start_before,
                start_before,
            );
            let turn = &mut session.turns[prepared.turn_index];
            turn.status = TurnStatus::Blocked;
            turn.error =
                Some("系统提示与当前输入超过 80% 安全裁剪目标，请缩短系统提示或当前输入".into());
            turn.touch();
            session.status = SessionStatus::Paused;
            self.store.save(session)?;
            return Ok(self.prepared(
                session,
                prepared.turn_index,
                mandatory,
                PreparationStatus::Blocked,
                "系统提示与当前输入超过 80% 安全裁剪目标，请缩短系统提示或当前输入",
            ));
        }

        let target = session.budget.trim_target();
        let mut low = 0_usize;
        let mut high = history.len();
        let mut best_count = 0_usize;
        while low <= high {
            let middle = (low + high) / 2;
            let candidate_metric = if let Some(candidate) = cache.get(&middle) {
                plan_metric(candidate)?
            } else {
                let start = history.len().saturating_sub(middle);
                let mut candidate = self.assembler.assemble_with_recall(
                    session,
                    &user_content,
                    Some(&history[start..]),
                    Some(prepared.turn_index),
                    Some(&self.recall_from_plan(&prepared.plan)?),
                );
                self.probe_plan(session, prepared.turn_index, &mut candidate)
                    .await?;
                let metric = plan_metric(&candidate)?;
                cache.insert(middle, candidate);
                metric
            };
            if candidate_metric <= target {
                best_count = middle;
                low = middle + 1;
            } else if middle == 0 {
                break;
            } else {
                high = middle - 1;
            }
        }
        let selected_plan = cache
            .remove(&best_count)
            .ok_or_else(|| anyhow!("内部错误：裁剪结果缺失"))?;
        let new_start = if best_count > 0 {
            *selected_plan
                .selected_history_indices
                .first()
                .ok_or_else(|| anyhow!("内部错误：保留上下文没有索引"))?
        } else {
            prepared.turn_index
        };
        session.active_context_start_index = new_start;
        session.status = SessionStatus::Active;
        apply_trace(
            session,
            prepared.turn_index,
            &selected_plan,
            "trimmed_and_continued",
            start_before,
            new_start,
        );
        session.turns[prepared.turn_index].touch();
        self.store.save(session)?;
        Ok(self.prepared(
            session,
            prepared.turn_index,
            selected_plan,
            PreparationStatus::Ready,
            &format!("已保留最近 {best_count} 个完整轮次并继续。"),
        ))
    }

    pub async fn stream_turn<F>(
        &self,
        session: &mut Session,
        prepared: &PreparedTurn,
        cancellation: CancellationToken,
        mut emit: F,
    ) -> Result<()>
    where
        F: FnMut(ChatEvent) + Send,
    {
        if !prepared.ready() {
            bail!("轮次尚未准备好，不能生成");
        }
        self.pending_turn(session, prepared)?;
        let mut thinking = String::new();
        let mut content = String::new();
        let mut live_output_tokens = 0_u64;
        let mut final_usage = None;
        let mut done_reason = None;
        let request = ChatRequest {
            model: session.model.clone(),
            messages: prepared.plan.messages.clone(),
            think: session.think,
            num_ctx: session.budget.context_window,
            num_predict: session.budget.max_output_tokens,
        };
        let turn = &mut session.turns[prepared.turn_index];
        turn.request_started_at = Some(utc_now());
        turn.touch();
        self.store.save(session)?;
        let result = self
            .client
            .stream_chat(request, cancellation, &mut |event| {
                if let Some(value) = event.live_output_tokens {
                    live_output_tokens = value;
                }
                match event.kind {
                    ChatEventKind::Thinking => thinking.push_str(&event.text),
                    ChatEventKind::Content => content.push_str(&event.text),
                    ChatEventKind::Completed => {
                        final_usage = event.usage;
                        done_reason.clone_from(&event.done_reason);
                    }
                    ChatEventKind::Usage => {}
                }
                emit(event);
            })
            .await;

        if let Err(error) = result {
            self.persist_stream_error(
                session,
                prepared.turn_index,
                &error,
                StreamSnapshot {
                    thinking: &thinking,
                    content: &content,
                    live_output_tokens: error.live_output_tokens().unwrap_or(live_output_tokens),
                    final_usage,
                },
            )?;
            return Err(error.into());
        }

        let Some(usage) = final_usage else {
            let error = OllamaError::Protocol("模型流在完成事件之前结束".into());
            self.persist_stream_error(
                session,
                prepared.turn_index,
                &error,
                StreamSnapshot {
                    thinking: &thinking,
                    content: &content,
                    live_output_tokens,
                    final_usage: None,
                },
            )?;
            return Err(error.into());
        };
        if prepared.plan.exact_input_tokens.is_some()
            && prepared.plan.exact_input_tokens != usage.input_tokens
        {
            let error = OllamaError::Protocol(
                "精确探测与正式请求的输入 token 不一致；拒绝将该轮加入上下文".into(),
            );
            self.persist_stream_error(
                session,
                prepared.turn_index,
                &error,
                StreamSnapshot {
                    thinking: &thinking,
                    content: &content,
                    live_output_tokens,
                    final_usage: Some(usage),
                },
            )?;
            return Err(error.into());
        }

        let turn = &mut session.turns[prepared.turn_index];
        turn.thinking = thinking;
        turn.assistant_content = content;
        turn.usage = usage;
        turn.done_reason = done_reason;
        turn.context_trace.exact_input_tokens = usage.input_tokens;
        if turn.assistant_content.is_empty() {
            turn.status = TurnStatus::NoAnswer;
            turn.error = Some("模型未返回可作为后续上下文的正文".into());
        } else if turn.done_reason.as_deref() == Some("length") {
            turn.status = TurnStatus::Truncated;
            turn.error = Some("回答达到输出 token 上限，正文可能不完整".into());
        } else {
            turn.status = TurnStatus::Complete;
            turn.error = None;
        }
        turn.touch();
        session.status = SessionStatus::Active;
        self.store.save(session)?;
        Ok(())
    }

    async fn build_plan(
        &self,
        session: &Session,
        turn_index: usize,
        history: &[usize],
        user_content: &str,
        recall: &RecallResult,
    ) -> Result<(ContextPlan, bool)> {
        let mut plan = self.assembler.assemble_with_recall(
            session,
            user_content,
            Some(history),
            Some(turn_index),
            Some(recall),
        );
        match self
            .client
            .render_prompt(
                &session.model,
                &plan.messages,
                session.think,
                session.budget.context_window,
            )
            .await
        {
            Ok(Some(rendered)) => {
                ContextAssembler::apply_rendered_upper_bound(&mut plan, &rendered);
                Ok((plan, true))
            }
            Ok(None) => Ok((plan, false)),
            Err(OllamaError::ContextLength { prompt_tokens, .. }) => {
                plan.exact_input_tokens =
                    Some(prompt_tokens.unwrap_or(session.budget.context_window + 1));
                Ok((plan, true))
            }
            Err(error) => Err(error.into()),
        }
    }

    async fn probe_plan(
        &self,
        session: &mut Session,
        turn_index: usize,
        plan: &mut ContextPlan,
    ) -> Result<()> {
        if plan.exact_input_tokens.is_some() {
            return Ok(());
        }
        match self
            .client
            .probe(
                &session.model,
                &plan.messages,
                session.think,
                session.budget.context_window,
            )
            .await
        {
            Ok(usage) => {
                plan.exact_input_tokens = usage.input_tokens;
                session.turns[turn_index].probe_usage.add(usage);
            }
            Err(OllamaError::ContextLength { prompt_tokens, .. }) => {
                plan.exact_input_tokens =
                    Some(prompt_tokens.unwrap_or(session.budget.context_window + 1));
            }
            Err(error) => return Err(error.into()),
        }
        Ok(())
    }

    fn block_mandatory(
        &self,
        session: &mut Session,
        turn_index: usize,
        plan: ContextPlan,
        start_before: usize,
    ) -> Result<PreparedTurn> {
        let message = "系统提示与当前输入本身已超过输入预算；请缩短输入或提高上下文配置";
        apply_trace(
            session,
            turn_index,
            &plan,
            "mandatory_input_exceeded",
            start_before,
            start_before,
        );
        let turn = &mut session.turns[turn_index];
        turn.status = TurnStatus::Blocked;
        turn.error = Some(message.into());
        turn.touch();
        session.status = SessionStatus::Paused;
        self.store.save(session)?;
        Ok(self.prepared(
            session,
            turn_index,
            plan,
            PreparationStatus::Blocked,
            message,
        ))
    }

    fn persist_stream_error(
        &self,
        session: &mut Session,
        turn_index: usize,
        error: &OllamaError,
        snapshot: StreamSnapshot<'_>,
    ) -> Result<()> {
        let turn = &mut session.turns[turn_index];
        turn.thinking = snapshot.thinking.to_owned();
        turn.assistant_content = snapshot.content.to_owned();
        turn.usage = snapshot.final_usage.unwrap_or_else(|| {
            TokenUsage::new(
                None,
                (snapshot.live_output_tokens > 0).then_some(snapshot.live_output_tokens),
            )
        });
        match error {
            OllamaError::ContextLength { .. } => {
                turn.status = TurnStatus::Blocked;
                session.status = SessionStatus::Paused;
            }
            OllamaError::Cancelled { .. } => {
                turn.status = TurnStatus::Interrupted;
                session.status = SessionStatus::Paused;
            }
            _ if !snapshot.thinking.is_empty()
                || !snapshot.content.is_empty()
                || snapshot.live_output_tokens > 0 =>
            {
                turn.status = TurnStatus::Interrupted;
            }
            _ => turn.status = TurnStatus::Failed,
        }
        turn.error = Some(error.to_string());
        turn.touch();
        self.store.save(session)?;
        Ok(())
    }

    fn pending_turn(&self, session: &Session, prepared: &PreparedTurn) -> Result<()> {
        if session.id != prepared.session_id {
            bail!("prepared turn belongs to a different session");
        }
        let turn = session
            .turns
            .get(prepared.turn_index)
            .ok_or_else(|| anyhow!("prepared turn index is no longer valid"))?;
        if turn.id != prepared.turn_id || turn.status != TurnStatus::Pending {
            bail!("prepared turn no longer references a pending turn");
        }
        Ok(())
    }

    fn recover_stale_pending(&self, session: &mut Session) -> Result<()> {
        let mut changed = false;
        for turn in &mut session.turns {
            if turn.status == TurnStatus::Pending {
                turn.status = TurnStatus::Interrupted;
                turn.error = Some("上次进程在该轮完成前退出".into());
                turn.touch();
                changed = true;
            }
        }
        if changed {
            self.store.save(session)?;
        }
        Ok(())
    }

    fn prepared(
        &self,
        session: &Session,
        turn_index: usize,
        plan: ContextPlan,
        status: PreparationStatus,
        message: &str,
    ) -> PreparedTurn {
        PreparedTurn {
            session_id: session.id.clone(),
            turn_id: session.turns[turn_index].id.clone(),
            turn_index,
            plan,
            status,
            message: message.to_owned(),
        }
    }

    fn recall_from_plan(&self, plan: &ContextPlan) -> Result<RecallResult> {
        let evidence = plan
            .evidence
            .iter()
            .map(|selected| {
                let content = self.store.retrieval().resolve_span(&selected.span)?.content;
                Ok::<RecalledEvidence, anyhow::Error>(RecalledEvidence {
                    selected: selected.clone(),
                    content,
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(RecallResult {
            trace: plan.retrieval_trace.clone(),
            evidence,
        })
    }
}

fn plan_metric(plan: &ContextPlan) -> Result<u64> {
    plan.exact_input_tokens
        .or(plan.estimated_upper_tokens)
        .ok_or_else(|| anyhow!("上下文计划缺少精确或估计 token 数"))
}

fn apply_trace(
    session: &mut Session,
    turn_index: usize,
    plan: &ContextPlan,
    decision: &str,
    start_before: usize,
    start_after: usize,
) {
    let request = ModelRequestTrace {
        model: session.model.clone(),
        think: session.think,
        context_window: session.budget.context_window,
        max_output_tokens: session.budget.max_output_tokens,
    };
    let turn = &mut session.turns[turn_index];
    turn.context_trace = ContextTrace {
        included_turn_ids: plan.included_turn_ids.clone(),
        omitted_turn_ids: plan.omitted_turn_ids.clone(),
        estimated_upper_tokens: plan.estimated_upper_tokens,
        exact_input_tokens: plan.exact_input_tokens,
        input_budget: plan.input_budget,
        decision: decision.to_owned(),
        active_context_start_before: start_before,
        active_context_start_after: start_after,
        context_items: plan.context_items.clone(),
        context_sha256: Some(plan.context_sha256.clone()),
        request: Some(request),
        provenance_quality: ProvenanceQuality::Exact,
        retrieval: plan.retrieval_trace.clone(),
    };
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use super::*;
    use crate::model::{BudgetConfig, ChatMessage};
    use crate::ollama::ModelInfo;

    #[derive(Clone)]
    struct FakeClient {
        count: u64,
        history_cost: Option<(u64, u64)>,
        render_supported: bool,
        probes: Arc<Mutex<usize>>,
        events: Vec<ChatEvent>,
        stream_error: Option<OllamaError>,
        observe_source: Option<(PathBuf, Arc<Mutex<bool>>)>,
    }

    impl FakeClient {
        fn new(count: u64) -> Self {
            Self {
                count,
                history_cost: None,
                render_supported: true,
                probes: Arc::new(Mutex::new(0)),
                events: vec![
                    ChatEvent::text(ChatEventKind::Thinking, "reason".into(), 1),
                    ChatEvent::text(ChatEventKind::Content, "answer".into(), 2),
                    ChatEvent {
                        kind: ChatEventKind::Completed,
                        text: String::new(),
                        live_output_tokens: Some(2),
                        usage: Some(TokenUsage::new(Some(count), Some(3))),
                        done_reason: Some("stop".into()),
                    },
                ],
                stream_error: None,
                observe_source: None,
            }
        }

        fn count_for(&self, messages: &[ChatMessage]) -> u64 {
            if let Some((base, per_history)) = self.history_cost {
                base + per_history
                    * messages
                        .iter()
                        .filter(|message| message.role == "assistant")
                        .count() as u64
            } else {
                self.count
            }
        }
    }

    #[async_trait]
    impl ChatBackend for FakeClient {
        async fn check_model(&self, model: &str, _: u64) -> Result<ModelInfo, OllamaError> {
            Ok(ModelInfo {
                version: "test".into(),
                name: model.into(),
                context_length: 65_536,
            })
        }

        async fn render_prompt(
            &self,
            _: &str,
            messages: &[ChatMessage],
            _: bool,
            _: u64,
        ) -> Result<Option<String>, OllamaError> {
            Ok(self
                .render_supported
                .then(|| "x".repeat(self.count_for(messages) as usize)))
        }

        async fn probe(
            &self,
            _: &str,
            messages: &[ChatMessage],
            _: bool,
            _: u64,
        ) -> Result<TokenUsage, OllamaError> {
            *self.probes.lock().unwrap() += 1;
            Ok(TokenUsage::new(Some(self.count_for(messages)), Some(1)))
        }

        async fn stream_chat(
            &self,
            _: ChatRequest,
            _: CancellationToken,
            emit: &mut (dyn FnMut(ChatEvent) + Send),
        ) -> Result<(), OllamaError> {
            if let Some((path, observed)) = &self.observe_source {
                let raw = std::fs::read(path).unwrap();
                let persisted: Session = serde_json::from_slice(&raw).unwrap();
                *observed.lock().unwrap() = persisted
                    .turns
                    .last()
                    .is_some_and(|turn| turn.request_started_at.is_some());
            }
            for event in &self.events {
                emit(event.clone());
            }
            self.stream_error.clone().map_or(Ok(()), Err)
        }
    }

    fn budget() -> BudgetConfig {
        BudgetConfig {
            context_window: 1_000,
            max_output_tokens: 100,
            safety_margin_tokens: 0,
            probe_ratio: 0.8,
            warning_ratio: 0.9,
            trim_target_ratio: 0.8,
        }
    }

    #[tokio::test]
    async fn below_threshold_streams_and_persists_exact_usage() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let client = FakeClient::new(100);
        let engine = ChatEngine::new(store.clone(), client.clone());
        let prepared = engine
            .prepare_turn(&mut session, "hello".into())
            .await
            .unwrap();
        assert!(prepared.ready());
        assert_eq!(*client.probes.lock().unwrap(), 0);
        engine
            .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        assert_eq!(session.turns[0].status, TurnStatus::Complete);
        assert_eq!(session.turns[0].thinking, "reason");
        assert_eq!(session.turns[0].usage.total_tokens, Some(103));
        let answer_id = crate::model::event_id(
            &session.id,
            Some(&session.turns[0].id),
            crate::model::EventRole::Assistant,
        );
        let trace = engine
            .store()
            .retrieval()
            .answer_context(&answer_id)
            .unwrap();
        assert_eq!(trace.provenance_quality, ProvenanceQuality::Exact);
        assert_eq!(
            trace
                .items
                .iter()
                .map(|item| item.resolved.content.as_str())
                .collect::<Vec<_>>(),
            vec![session.system_prompt.as_str(), "hello"]
        );
        let next = ContextAssembler.assemble(&session, "next", None, None);
        assert!(!format!("{:?}", next.messages).contains("reason"));
    }

    #[tokio::test]
    async fn request_start_is_durable_before_backend_stream_begins() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let observed = Arc::new(Mutex::new(false));
        let mut client = FakeClient::new(100);
        client.observe_source = Some((
            root.path().join(format!("{}.json", session.id)),
            observed.clone(),
        ));
        let engine = ChatEngine::new(store, client);
        let prepared = engine
            .prepare_turn(&mut session, "hello".into())
            .await
            .unwrap();
        engine
            .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        assert!(*observed.lock().unwrap());
    }

    #[tokio::test]
    async fn warning_can_end_without_streaming() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let engine = ChatEngine::new(store, FakeClient::new(850));
        let prepared = engine
            .prepare_turn(&mut session, "hello".into())
            .await
            .unwrap();
        assert!(prepared.needs_limit_decision());
        let ended = engine
            .resolve_limit(&mut session, prepared, LimitAction::EndSession)
            .await
            .unwrap();
        assert_eq!(ended.status, PreparationStatus::Ended);
        assert_eq!(session.status, SessionStatus::Paused);
        assert_eq!(session.turns[0].status, TurnStatus::Blocked);
        let answer_id = crate::model::event_id(
            &session.id,
            Some(&session.turns[0].id),
            crate::model::EventRole::Assistant,
        );
        assert!(matches!(
            engine.store().retrieval().get_event(&answer_id),
            Err(crate::retrieval::RetrievalError::EventNotFound(_))
        ));
    }

    #[tokio::test]
    async fn render_fallback_forces_exact_probe() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let mut client = FakeClient::new(100);
        client.render_supported = false;
        let probes = client.probes.clone();
        let prepared = ChatEngine::new(store, client)
            .prepare_turn(&mut session, "hello".into())
            .await
            .unwrap();
        assert_eq!(prepared.plan.exact_input_tokens, Some(100));
        assert_eq!(*probes.lock().unwrap(), 1);
        assert_eq!(session.turns[0].probe_usage.total_tokens, Some(101));
    }

    #[tokio::test]
    async fn thresholds_are_inclusive() {
        for (count, expected) in [
            (720, PreparationStatus::Ready),
            (810, PreparationStatus::LimitWarning),
        ] {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path()).unwrap();
            let mut session = store
                .create("model", "http://localhost", None, budget(), true)
                .unwrap();
            let client = FakeClient::new(count);
            let probes = client.probes.clone();
            let prepared = ChatEngine::new(store, client)
                .prepare_turn(&mut session, "hello".into())
                .await
                .unwrap();
            assert_eq!(prepared.status, expected);
            assert!(*probes.lock().unwrap() >= 1);
        }
    }

    fn completed_turn(index: usize) -> Turn {
        let mut turn = Turn::pending(format!("user-{index}"));
        turn.status = TurnStatus::Complete;
        turn.assistant_content = format!("assistant-{index}");
        turn.usage = TokenUsage::new(Some(10), Some(5));
        turn
    }

    #[tokio::test]
    async fn continue_keeps_maximum_recent_suffix_at_trim_target() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        session.turns = (0..3).map(completed_turn).collect();
        store.save(&mut session).unwrap();
        let mut client = FakeClient::new(100);
        client.history_cost = Some((300, 250));
        let engine = ChatEngine::new(store, client);
        let prepared = engine
            .prepare_turn(&mut session, "current".into())
            .await
            .unwrap();
        let resumed = engine
            .resolve_limit(&mut session, prepared, LimitAction::ContinueWithTrim)
            .await
            .unwrap();
        assert_eq!(resumed.plan.exact_input_tokens, Some(550));
        assert_eq!(
            resumed.plan.included_turn_ids,
            vec![session.turns[2].id.clone()]
        );
        assert_eq!(session.active_context_start_index, 2);
        assert_eq!(
            session.turns.last().unwrap().context_trace.decision,
            "trimmed_and_continued"
        );
    }

    #[tokio::test]
    async fn interrupted_stream_never_promotes_probe_usage() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let mut client = FakeClient::new(750);
        client.events = vec![ChatEvent::text(ChatEventKind::Content, "partial".into(), 2)];
        client.stream_error = Some(OllamaError::Stream {
            message: "lost".into(),
            live_output_tokens: 2,
        });
        let engine = ChatEngine::new(store, client);
        let prepared = engine
            .prepare_turn(&mut session, "question".into())
            .await
            .unwrap();
        assert!(
            engine
                .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
                .await
                .is_err()
        );
        let turn = session.turns.last().unwrap();
        assert_eq!(turn.status, TurnStatus::Interrupted);
        assert_eq!(turn.usage, TokenUsage::new(None, Some(2)));
        assert_eq!(turn.probe_usage, TokenUsage::new(Some(750), Some(1)));
        assert!(!turn.context_eligible());
    }
}
