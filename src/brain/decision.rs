use rand::Rng;
use serde::{Deserialize, Serialize};

/// Structured output from Gemma 4 representing Sparky's decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmaDecision {
    pub target_emotion: String,
    pub dialogue: String,
    #[serde(default)]
    pub energy_delta: Option<f32>,
    #[serde(default)]
    pub mood_delta: Option<f32>,
    #[serde(default)]
    pub affection_delta: Option<f32>,
    #[serde(default)]
    pub boredom_delta: Option<f32>,
    #[serde(default)]
    pub duration_sec: Option<f32>,
}

impl GemmaDecision {
    pub fn new(target_emotion: impl Into<String>, dialogue: impl Into<String>) -> Self {
        Self {
            target_emotion: target_emotion.into(),
            dialogue: dialogue.into(),
            energy_delta: None,
            mood_delta: None,
            affection_delta: None,
            boredom_delta: None,
            duration_sec: Some(5.0),
        }
    }

    /// Map the requested emotion to one of the available video names.
    ///
    /// Available animation files are:
    /// `angry`, `happy1`, `happy2`, `naughty`, `sleepy`, `tired`.
    pub fn resolve_animation_name(&self, available_names: &[String]) -> Option<String> {
        if available_names.is_empty() {
            return None;
        }

        let emotion = self.target_emotion.trim().to_ascii_lowercase();

        // 1. Direct exact match
        if let Some(found) = available_names.iter().find(|n| n.eq_ignore_ascii_case(&emotion)) {
            return Some(found.clone());
        }

        // 2. If 'happy', randomly pick 'happy1' or 'happy2' if available
        if emotion == "happy" || emotion.contains("joy") || emotion.contains("glad") {
            let happy_candidates: Vec<&String> = available_names
                .iter()
                .filter(|n| n.starts_with("happy"))
                .collect();
            if !happy_candidates.is_empty() {
                let mut rng = rand::rng();
                let idx = rng.random_range(0..happy_candidates.len());
                return Some(happy_candidates[idx].clone());
            }
        }

        // 3. Substring matching
        for anim in available_names {
            let anim_lower = anim.to_ascii_lowercase();
            if emotion.contains(&anim_lower) || anim_lower.contains(&emotion) {
                return Some(anim.clone());
            }
        }

        // 4. Semantic synonyms
        if emotion.contains("mad") || emotion.contains("furious") || emotion.contains("annoy") {
            if let Some(a) = available_names.iter().find(|n| n.contains("angry")) {
                return Some(a.clone());
            }
        }
        if emotion.contains("cheeky") || emotion.contains("smug") || emotion.contains("mischief") {
            if let Some(a) = available_names.iter().find(|n| n.contains("naughty")) {
                return Some(a.clone());
            }
        }
        if emotion.contains("drowsy") || emotion.contains("nap") || emotion.contains("rest") {
            if let Some(a) = available_names.iter().find(|n| n.contains("sleepy")) {
                return Some(a.clone());
            }
        }
        if emotion.contains("exhaust") || emotion.contains("worn") || emotion.contains("burnout") {
            if let Some(a) = available_names.iter().find(|n| n.contains("tired")) {
                return Some(a.clone());
            }
        }

        // Fallback to first available
        available_names.first().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_animation_name() {
        let available = vec![
            "angry".to_string(),
            "happy1".to_string(),
            "happy2".to_string(),
            "naughty".to_string(),
            "sleepy".to_string(),
            "tired".to_string(),
        ];

        let d1 = GemmaDecision::new("naughty", "smirk");
        assert_eq!(d1.resolve_animation_name(&available).unwrap(), "naughty");

        let d2 = GemmaDecision::new("happy", "yay");
        let h = d2.resolve_animation_name(&available).unwrap();
        assert!(h == "happy1" || h == "happy2");

        let d3 = GemmaDecision::new("furious", "grr");
        assert_eq!(d3.resolve_animation_name(&available).unwrap(), "angry");

        let d4 = GemmaDecision::new("drowsy", "zzz");
        assert_eq!(d4.resolve_animation_name(&available).unwrap(), "sleepy");
    }
}
