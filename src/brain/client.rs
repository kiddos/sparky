use std::time::Duration;
use log::{debug, info, warn};
use serde_json::Value;

use super::decision::GemmaDecision;
use super::personality::PersonalityArchetype;
use super::state::EmotionalState;

#[derive(Clone)]
pub struct GemmaClient {
    pub api_key: Option<String>,
    pub api_base: String,
    pub model: String,
    pub personality: PersonalityArchetype,
    pub timeout: Duration,
    pub has_warned_missing_key: bool,
}

impl GemmaClient {
    pub fn new(
        api_key: Option<String>,
        api_base: Option<String>,
        model: Option<String>,
        personality: PersonalityArchetype,
    ) -> Self {
        let base = api_base
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string())
            .trim_end_matches('/')
            .to_string();

        let chosen_model = model.unwrap_or_else(|| "gemma-4".to_string());

        Self {
            api_key,
            api_base: base,
            model: chosen_model,
            personality,
            timeout: Duration::from_secs(6),
            has_warned_missing_key: false,
        }
    }

    /// Consult Gemma 4 to produce Sparky's next emotional transition and dialogue.
    pub fn decide(
        &mut self,
        state: &EmotionalState,
        recent_events: &[String],
    ) -> GemmaDecision {
        let key = match &self.api_key {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                if !self.has_warned_missing_key {
                    warn!(
                        "No GEMMA_API_KEY found in environment or .env file. Sparky is running using local heuristic intelligence."
                    );
                    info!(
                        "Tip: Copy .env.example to .env and set GEMMA_API_KEY to enable full Gemma 4 intelligence."
                    );
                    self.has_warned_missing_key = true;
                }
                return self.heuristic_fallback(state, recent_events);
            }
        };

        let system_prompt = self.personality.system_prompt();
        let user_prompt = build_user_prompt(state, recent_events);

        let result = if self.is_openai_compatible() {
            self.call_openai_compatible(&key, &system_prompt, &user_prompt)
        } else {
            self.call_google_gemini(&key, &system_prompt, &user_prompt)
        };

        match result {
            Ok(decision) => {
                debug!(
                    "Gemma 4 response received: emotion='{}', dialogue='{}'",
                    decision.target_emotion, decision.dialogue
                );
                decision
            }
            Err(e) => {
                warn!(
                    "Gemma 4 API request failed ({}). Falling back to local personality heuristics.",
                    e
                );
                self.heuristic_fallback(state, recent_events)
            }
        }
    }

    /// Consult Gemma 4 to directly respond to a user chat message.
    pub fn decide_chat(
        &mut self,
        state: &EmotionalState,
        recent_events: &[String],
        user_msg: &str,
    ) -> GemmaDecision {
        let key = match &self.api_key {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                return self.personality.chat_reply(user_msg, state);
            }
        };

        let system_prompt = self.personality.system_prompt();
        let user_prompt = format!(
            "{}\n\nIMPORTANT: The user just typed this message to you in the chat window:\n\"{}\"\nRespond directly to the user in your character voice (1-2 sentences), pick your next emotion animation, and output JSON:",
            build_user_prompt(state, recent_events),
            user_msg
        );

        let result = if self.is_openai_compatible() {
            self.call_openai_compatible(&key, &system_prompt, &user_prompt)
        } else {
            self.call_google_gemini(&key, &system_prompt, &user_prompt)
        };

        match result {
            Ok(decision) => {
                debug!(
                    "Gemma 4 chat response received: emotion='{}', dialogue='{}'",
                    decision.target_emotion, decision.dialogue
                );
                decision
            }
            Err(e) => {
                warn!(
                    "Gemma 4 API request failed ({}). Falling back to personality chat heuristics.",
                    e
                );
                self.personality.chat_reply(user_msg, state)
            }
        }
    }

    fn is_openai_compatible(&self) -> bool {
        self.api_base.contains("/v1")
            || self.api_base.contains(":11434")
            || self.api_base.contains("openai.com")
            || self.api_base.contains("openrouter.ai")
    }

    /// Google AI Studio / Gemini API call
    fn call_google_gemini(
        &self,
        api_key: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<GemmaDecision, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.api_base, self.model, api_key
        );

        let body = serde_json::json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [{ "text": user_prompt }]
                }
            ],
            "systemInstruction": {
                "parts": [{ "text": system_prompt }]
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.85
            }
        });

        let resp: Value = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(body)?
            .into_json()?;

        let raw_text = resp["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| {
                format!(
                    "Missing candidates text in response: {}",
                    serde_json::to_string(&resp).unwrap_or_default()
                )
            })?;

        parse_json_decision(raw_text)
    }

    /// OpenAI-compatible call (Ollama, vLLM, OpenRouter, etc.)
    fn call_openai_compatible(
        &self,
        api_key: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<GemmaDecision, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/chat/completions", self.api_base);

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "response_format": { "type": "json_object" },
            "temperature": 0.85
        });

        let mut req = ureq::post(&url).timeout(self.timeout);
        if !api_key.is_empty() && api_key != "none" {
            req = req.set("Authorization", &format!("Bearer {}", api_key));
        }

        let resp: Value = req.send_json(body)?.into_json()?;

        let raw_text = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                format!(
                    "Missing choices content in response: {}",
                    serde_json::to_string(&resp).unwrap_or_default()
                )
            })?;

        parse_json_decision(raw_text)
    }

    /// Offline/fallback generator based on personality and current emotional state.
    pub fn heuristic_fallback(
        &self,
        state: &EmotionalState,
        recent_events: &[String],
    ) -> GemmaDecision {
        self.personality.heuristic_decision(state, recent_events)
    }
}

fn build_user_prompt(state: &EmotionalState, recent_events: &[String]) -> String {
    let events_str = if recent_events.is_empty() {
        "No major user interaction recently (Sparky is idling).".to_string()
    } else {
        recent_events
            .iter()
            .rev()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n- ")
    };

    format!(
        "Current Internal State: {}\nDominant Vibe: {}\n\nRecent User Events:\n- {}\n\nDecide Sparky's next target_emotion, dialogue, and vector deltas as JSON:",
        state.prompt_summary(),
        state.dominant_vibe(),
        events_str
    )
}

/// Robust JSON extractor that handles markdown code blocks or conversational prefixes.
pub fn parse_json_decision(raw: &str) -> Result<GemmaDecision, Box<dyn std::error::Error + Send + Sync>> {
    let cleaned = clean_json_str(raw);
    match serde_json::from_str::<GemmaDecision>(cleaned) {
        Ok(decision) => Ok(decision),
        Err(orig_err) => {
            // Attempt to parse into Value and extract fields leniently
            if let Ok(val) = serde_json::from_str::<Value>(cleaned) {
                let target_emotion = val["target_emotion"]
                    .as_str()
                    .unwrap_or("happy1")
                    .to_string();
                let dialogue = val["dialogue"]
                    .as_str()
                    .unwrap_or("Sparky chirps happily!")
                    .to_string();
                let energy_delta = val["energy_delta"].as_f64().map(|v| v as f32);
                let mood_delta = val["mood_delta"].as_f64().map(|v| v as f32);
                let affection_delta = val["affection_delta"].as_f64().map(|v| v as f32);
                let boredom_delta = val["boredom_delta"].as_f64().map(|v| v as f32);
                let duration_sec = val["duration_sec"].as_f64().map(|v| v as f32);

                return Ok(GemmaDecision {
                    target_emotion,
                    dialogue,
                    energy_delta,
                    mood_delta,
                    affection_delta,
                    boredom_delta,
                    duration_sec,
                });
            }
            Err(Box::new(orig_err))
        }
    }
}

fn clean_json_str(s: &str) -> &str {
    let trimmed = s.trim();
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if end >= start {
                return &trimmed[start..=end];
            }
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_clean_json() {
        let raw = r#"{"target_emotion": "naughty", "dialogue": "Hehe!", "energy_delta": -5.0, "mood_delta": 10.0}"#;
        let d = parse_json_decision(raw).unwrap();
        assert_eq!(d.target_emotion, "naughty");
        assert_eq!(d.dialogue, "Hehe!");
        assert_eq!(d.energy_delta, Some(-5.0));
    }

    #[test]
    fn test_parse_markdown_fenced_json() {
        let raw = "Here is Sparky's reaction:\n```json\n{\n  \"target_emotion\": \"happy2\",\n  \"dialogue\": \"Yay!\"\n}\n```";
        let d = parse_json_decision(raw).unwrap();
        assert_eq!(d.target_emotion, "happy2");
        assert_eq!(d.dialogue, "Yay!");
    }

    #[test]
    fn test_heuristic_fallback_drag() {
        let mut client = GemmaClient::new(None, None, None, PersonalityArchetype::ChaosGremlin);
        let state = EmotionalState::default();
        let events = vec!["User dragged Sparky".to_string()];
        let decision = client.decide(&state, &events);
        assert_eq!(decision.target_emotion, "naughty");
        assert!(!decision.dialogue.is_empty());
    }

    #[test]
    fn test_heuristic_fallback_bored() {
        let mut client = GemmaClient::new(None, None, None, PersonalityArchetype::ZenMonk);
        let mut state = EmotionalState::default();
        state.boredom = 90.0;
        let events = vec![];
        let decision = client.decide(&state, &events);
        assert_eq!(decision.target_emotion, "sleepy");
        assert!(decision.dialogue.contains("tranquil") || !decision.dialogue.is_empty());
    }
}
