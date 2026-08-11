use crate::model::{ChatMessage, ContextPlan, Session};

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
        if !session.system_prompt.is_empty() {
            messages.push(ChatMessage {
                role: "system".to_owned(),
                content: session.system_prompt.clone(),
            });
        }
        for index in &selected_indices {
            let turn = &session.turns[*index];
            messages.push(ChatMessage {
                role: "user".to_owned(),
                content: turn.user_content.clone(),
            });
            messages.push(ChatMessage {
                role: "assistant".to_owned(),
                content: turn.assistant_content.clone(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_owned(),
            content: current_user.to_owned(),
        });

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

        ContextPlan {
            messages,
            included_turn_ids,
            omitted_turn_ids,
            selected_history_indices: selected_indices,
            estimated_upper_tokens,
            exact_input_tokens: None,
            input_budget: session.budget.input_budget(),
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
        assert_eq!(plan.included_turn_ids, vec![session.turns[1].id.clone()]);
        assert!(!format!("{:?}", plan.messages).contains("secret"));
        assert_eq!(plan.messages.last().unwrap().content, "current");
    }
}
