use crate::state::emotion::{EmotionRegistry, LoopMode};
use std::time::{Duration, Instant};

pub struct StateMachine {
    pub current_emotion: String,
    pub active_clip_name: String,
    pub active_tint: [f32; 3],
    state_start: Instant,
    last_interaction: Instant,
}

impl StateMachine {
    pub fn new(registry: &EmotionRegistry) -> Self {
        let (clip, tint) = registry.resolve_visuals("sit");
        log::info!(
            "State machine initialized (initial emotion: 'sit', clip: '{clip}', tint: {tint:?})"
        );
        Self {
            current_emotion: "sit".to_string(),
            active_clip_name: clip,
            active_tint: tint,
            state_start: Instant::now(),
            last_interaction: Instant::now(),
        }
    }

    pub fn transition_to(&mut self, emotion_id: &str, registry: &EmotionRegistry) {
        if registry.get(emotion_id).is_some() {
            let (clip, tint) = registry.resolve_visuals(emotion_id);
            log::info!(
                "Emotion transition: '{}' -> '{}' (clip: '{}', tint: {:?})",
                self.current_emotion,
                emotion_id,
                clip,
                tint
            );
            self.current_emotion = emotion_id.to_string();
            self.active_clip_name = clip;
            self.active_tint = tint;
            self.state_start = Instant::now();
        }
    }

    #[allow(dead_code)]
    pub fn on_pet(&mut self, registry: &EmotionRegistry) {
        self.last_interaction = Instant::now();
        self.transition_to("happy", registry);
    }

    #[allow(dead_code)]
    pub fn on_poke(&mut self, registry: &EmotionRegistry) {
        self.last_interaction = Instant::now();
        self.transition_to("angry", registry);
    }

    pub fn touch_interaction(&mut self) {
        self.last_interaction = Instant::now();
    }

    pub fn tick(&mut self, registry: &EmotionRegistry) {
        let elapsed = self.state_start.elapsed();

        if let Some(def) = registry.get(&self.current_emotion) {
            match def.loop_mode {
                LoopMode::OnceThenReturn => {
                    if elapsed > Duration::from_secs_f32(def.duration_secs) {
                        self.transition_to(&def.return_to, registry);
                    }
                }
                LoopMode::Loop => {
                    // Check boredom timer when sitting idle (3 mins)
                    if self.current_emotion == "sit"
                        && self.last_interaction.elapsed() > Duration::from_secs(180)
                    {
                        self.transition_to("naughty", registry);
                        self.last_interaction = Instant::now();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_machine_transitions() {
        let registry = EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
            .expect("Failed to load emotions");
        let mut sm = StateMachine::new(&registry);

        assert_eq!(sm.current_emotion, "sit");
        assert_eq!(sm.active_clip_name, "sit");

        sm.on_pet(&registry);
        assert_eq!(sm.current_emotion, "happy");
        assert_eq!(sm.active_clip_name, "happy");

        sm.on_poke(&registry);
        assert_eq!(sm.current_emotion, "angry");
        assert_eq!(sm.active_clip_name, "angry");

        // Extended emotion with dedicated asset or fallback
        sm.transition_to("sad", &registry);
        assert_eq!(sm.current_emotion, "sad");
        let expected_sad_clip = if registry.has_asset("sad.webp") {
            "sad"
        } else {
            "sit"
        };
        assert_eq!(sm.active_clip_name, expected_sad_clip);
        assert_eq!(sm.active_tint, [0.80, 0.85, 1.05]);

        // Extended emotion tired
        sm.transition_to("tired", &registry);
        assert_eq!(sm.current_emotion, "tired");
        let expected_tired_clip = if registry.has_asset("tired.webp") {
            "tired"
        } else {
            "sit"
        };
        assert_eq!(sm.active_clip_name, expected_tired_clip);

        // Extended emotion sleep with fallback
        sm.transition_to("sleep", &registry);
        assert_eq!(sm.current_emotion, "sleep");
        let expected_sleep_clip = if registry.has_asset("sleep.webp") {
            "sleep"
        } else if registry.has_asset("tired.webp") {
            "tired"
        } else {
            "sit"
        };
        assert_eq!(sm.active_clip_name, expected_sleep_clip);
    }
}
