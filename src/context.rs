use crate::knowledge::KnowledgeRecall;
use crate::model::{
    ChatMessage, ContextItemTrace, ContextPlan, EventRole, Session, SourceSpan, content_sha256,
    context_sha256, event_id, identity_instruction,
};
use crate::retrieval::RecallResult;

#[derive(Debug, Default, Clone, Copy)]
pub struct ContextAssembler;

impl ContextAssembler {
    const FALLBACK_BASE_OVERHEAD: u64 = 256;
    const FALLBACK_MESSAGE_OVERHEAD: u64 = 64;

    pub fn assemble(
        &self,
        session: &Session,
        current_user: &str,
        history_indices: Option<&[usize]>,
        current_turn_index: Option<usize>,
    ) -> ContextPlan {
        self.assemble_with_recall(
            session,
            current_user,
            history_indices,
            current_turn_index,
            None,
        )
    }

    pub fn assemble_with_recall(
        &self,
        session: &Session,
        current_user: &str,
        history_indices: Option<&[usize]>,
        current_turn_index: Option<usize>,
        recall: Option<&RecallResult>,
    ) -> ContextPlan {
        self.assemble_with_recall_and_knowledge(
            session,
            current_user,
            history_indices,
            current_turn_index,
            recall,
            None,
        )
    }

    pub fn assemble_with_recall_and_knowledge(
        &self,
        session: &Session,
        current_user: &str,
        history_indices: Option<&[usize]>,
        current_turn_index: Option<usize>,
        recall: Option<&RecallResult>,
        knowledge: Option<&KnowledgeRecall>,
    ) -> ContextPlan {
        let selected_indices = history_indices.map_or_else(
            || {
                session
                    .eligible_turns(current_turn_index, true)
                    .into_iter()
                    .map(|(index, _)| index)
                    .collect()
            },
            <[usize]>::to_vec,
        );

        let mut messages = Vec::new();
        let mut context_items = Vec::new();
        if !session.system_prompt.is_empty() {
            push_message(
                &mut messages,
                &mut context_items,
                &session.id,
                None,
                EventRole::System,
                &session.system_prompt,
            );
        }
        let identity_instruction = identity_instruction(&session.ai_name);
        messages.push(ChatMessage {
            role: EventRole::System.as_str().to_owned(),
            content: identity_instruction.clone(),
        });
        if let Some(message) = knowledge.and_then(|value| value.trace.injected_message.as_ref()) {
            messages.push(ChatMessage {
                role: EventRole::System.as_str().to_owned(),
                content: message.clone(),
            });
        }
        if let Some(recall) = recall {
            for item in &recall.evidence {
                // Evidence messages are deliberately original-role spans, not
                // a generated wrapper.  This keeps exact provenance intact.
                context_items.push(ContextItemTrace {
                    role: item.selected.role,
                    span: item.selected.span.clone(),
                    content_sha256: item.selected.content_sha256.clone(),
                });
                messages.push(ChatMessage {
                    role: item.selected.role.as_str().into(),
                    content: item.content.clone(),
                });
            }
        }
        for index in &selected_indices {
            let turn = &session.turns[*index];
            push_message(
                &mut messages,
                &mut context_items,
                &session.id,
                Some(&turn.id),
                EventRole::User,
                &turn.user_content,
            );
            push_message(
                &mut messages,
                &mut context_items,
                &session.id,
                Some(&turn.id),
                EventRole::Assistant,
                &turn.assistant_content,
            );
        }
        let current_turn_id = current_turn_index
            .and_then(|index| session.turns.get(index))
            .map_or("__transient_current__", |turn| turn.id.as_str());
        push_message(
            &mut messages,
            &mut context_items,
            &session.id,
            Some(current_turn_id),
            EventRole::User,
            current_user,
        );

        let included_turn_ids = selected_indices
            .iter()
            .map(|index| session.turns[*index].id.clone())
            .collect::<Vec<_>>();
        let omitted_turn_ids = session
            .eligible_turns(current_turn_index, false)
            .into_iter()
            .filter(|(index, _)| !selected_indices.contains(index))
            .map(|(_, turn)| turn.id.clone())
            .collect::<Vec<_>>();
        let estimated_upper_tokens = Some(Self::estimate_upper_bound(&messages));

        let context_sha256 = context_sha256(&messages);
        ContextPlan {
            messages,
            context_items,
            context_sha256,
            included_turn_ids,
            omitted_turn_ids,
            selected_history_indices: selected_indices,
            estimated_upper_tokens,
            exact_input_tokens: None,
            input_budget: session.budget.input_budget(),
            identity_instruction,
            retrieval_trace: recall.map(|value| value.trace.clone()).unwrap_or_default(),
            evidence: recall
                .map(|value| value.trace.selected_evidence.clone())
                .unwrap_or_default(),
            knowledge_trace: knowledge
                .map(|value| value.trace.clone())
                .unwrap_or_default(),
        }
    }

    pub fn estimate_upper_bound(messages: &[ChatMessage]) -> u64 {
        Self::FALLBACK_BASE_OVERHEAD
            + messages
                .iter()
                .map(|message| {
                    message.content.len() as u64
                        + message.role.len() as u64
                        + Self::FALLBACK_MESSAGE_OVERHEAD
                })
                .sum::<u64>()
    }

    pub fn apply_rendered_upper_bound(plan: &mut ContextPlan, rendered_prompt: &str) {
        plan.estimated_upper_tokens = Some(rendered_prompt.len() as u64);
    }
}

fn push_message(
    messages: &mut Vec<ChatMessage>,
    context_items: &mut Vec<ContextItemTrace>,
    session_id: &str,
    turn_id: Option<&str>,
    role: EventRole,
    content: &str,
) {
    let event_id = event_id(session_id, turn_id, role);
    context_items.push(ContextItemTrace {
        role,
        span: SourceSpan {
            event_id,
            start_char: 0,
            end_char: content.chars().count(),
        },
        content_sha256: content_sha256(content),
    });
    messages.push(ChatMessage {
        role: role.as_str().to_owned(),
        content: content.to_owned(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SessionStatus, TokenUsage, Turn, TurnStatus, utc_now};

    fn complete(user: &str, assistant: &str, thinking: &str) -> Turn {
        let now = utc_now();
        Turn {
            id: uuid::Uuid::new_v4().simple().to_string(),
            created_at: now.clone(),
            updated_at: now,
            status: TurnStatus::Complete,
            user_content: user.to_owned(),
            assistant_content: assistant.to_owned(),
            thinking: thinking.to_owned(),
            usage: TokenUsage::zero(),
            probe_usage: TokenUsage::zero(),
            context_trace: Default::default(),
            request_started_at: Some(utc_now()),
            done_reason: None,
            error: None,
        }
    }

    #[test]
    fn thinking_is_never_reinjected() {
        let mut session = Session::new(
            "one".into(),
            "model".into(),
            "http://localhost".into(),
            "system".into(),
            Default::default(),
            true,
        )
        .unwrap();
        session.status = SessionStatus::Active;
        session.turns = vec![
            complete("old", "old-a", "secret"),
            complete("new", "new-a", "new-secret"),
        ];
        session.active_context_start_index = 1;
        let plan = ContextAssembler.assemble(&session, "current", None, None);
        assert!(plan.identity_instruction.contains("LLM"));
        assert_eq!(plan.messages[1].content, plan.identity_instruction);
        assert_eq!(plan.included_turn_ids, vec![session.turns[1].id.clone()]);
        assert!(!format!("{:?}", plan.messages).contains("secret"));
        assert_eq!(plan.messages.last().unwrap().content, "current");
        assert_eq!(plan.context_items.len() + 1, plan.messages.len());
        assert!(
            plan.context_items
                .iter()
                .all(|item| item.role != EventRole::Assistant
                    || item.content_sha256 != content_sha256("new-secret"))
        );
        for (message, item) in plan
            .messages
            .iter()
            .filter(|message| message.content != plan.identity_instruction)
            .zip(&plan.context_items)
        {
            assert_eq!(item.role.as_str(), message.role);
            assert_eq!(item.span.start_char, 0);
            assert_eq!(item.span.end_char, message.content.chars().count());
            assert_eq!(item.content_sha256, content_sha256(&message.content));
        }
        assert_eq!(plan.context_sha256, context_sha256(&plan.messages));
    }
}
