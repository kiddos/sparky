use crate::state::emotion::EmotionRegistry;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub struct StateMachine {
    pub current_emotion: String,
    pub active_clip_name: String,
    pub active_tint: [f32; 3],
    pub emotion_queue: VecDeque<String>,
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
            emotion_queue: VecDeque::new(),
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

    /// Push the next emotion into the queue.
    /// When there are more than 3 emotions in the queue during the next push (queue length >= 3),
    /// pop the last one and push the current one into the queue.
    pub fn push_emotion(&mut self, emotion_id: &str, registry: &EmotionRegistry) {
        if registry.get(emotion_id).is_some() {
            if self.emotion_queue.len() >= 3 {
                let popped = self.emotion_queue.pop_back();
                log::info!(
                    "Emotion queue capacity reached (>= 3). Popped last emotion {:?} to push '{}'",
                    popped,
                    emotion_id
                );
            }
            self.emotion_queue.push_back(emotion_id.to_string());
            log::info!(
                "Pushed emotion '{}' to queue. Current queue: {:?}",
                emotion_id,
                self.emotion_queue
            );
        } else {
            log::warn!(
                "Attempted to push invalid emotion '{}' into queue; ignoring",
                emotion_id
            );
        }
    }

    /// Called after current animation has completed.
    /// Consumes the next emotion from the queue and changes the emotion.
    /// If no next emotion is presented in the queue, keeps the current emotion.
    /// Returns true if an emotion was consumed and changed, false if kept current emotion.
    pub fn on_animation_completed(&mut self, registry: &EmotionRegistry) -> bool {
        if let Some(next_emotion) = self.emotion_queue.pop_front() {
            log::info!(
                "Animation completed. Consuming queue: transitioning '{}' -> '{}'",
                self.current_emotion,
                next_emotion
            );
            self.transition_to(&next_emotion, registry);
            true
        } else {
            log::debug!(
                "Animation completed. Queue empty; keeping current emotion '{}'",
                self.current_emotion
            );
            self.state_start = Instant::now();
            false
        }
    }

    #[allow(dead_code)]
    pub fn clear_queue(&mut self) {
        self.emotion_queue.clear();
    }

    #[allow(dead_code)]
    pub fn queue(&self) -> &VecDeque<String> {
        &self.emotion_queue
    }

    #[allow(dead_code)]
    pub fn on_pet(&mut self, registry: &EmotionRegistry) {
        self.last_interaction = Instant::now();
        self.push_emotion("happy", registry);
    }

    #[allow(dead_code)]
    pub fn on_poke(&mut self, registry: &EmotionRegistry) {
        self.last_interaction = Instant::now();
        self.push_emotion("angry", registry);
    }

    pub fn touch_interaction(&mut self) {
        self.last_interaction = Instant::now();
    }

    pub fn tick(&mut self, registry: &EmotionRegistry) {
        let elapsed = self.state_start.elapsed();

        if let Some(def) = registry.get(&self.current_emotion) {
            let duration = Duration::from_secs_f32(def.duration_secs);
            if elapsed >= duration {
                // Current animation duration completed
                self.on_animation_completed(registry);
            }
        }

        // Check boredom timer when sitting idle with empty queue (3 mins)
        if self.current_emotion == "sit"
            && self.emotion_queue.is_empty()
            && self.last_interaction.elapsed() > Duration::from_secs(180)
        {
            self.push_emotion("naughty", registry);
            self.last_interaction = Instant::now();
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

        // on_pet pushes "happy" to the queue; current emotion remains "sit" until animation completes
        sm.on_pet(&registry);
        assert_eq!(sm.current_emotion, "sit");
        assert_eq!(sm.emotion_queue.front().map(|s| s.as_str()), Some("happy"));
        let transitioned = sm.on_animation_completed(&registry);
        assert!(transitioned);
        assert_eq!(sm.current_emotion, "happy");
        assert_eq!(sm.active_clip_name, "happy");
        assert!(sm.emotion_queue.is_empty());

        // on_poke pushes "angry" to the queue; current emotion remains "happy" until animation completes
        sm.on_poke(&registry);
        assert_eq!(sm.current_emotion, "happy");
        assert_eq!(sm.emotion_queue.front().map(|s| s.as_str()), Some("angry"));
        let transitioned = sm.on_animation_completed(&registry);
        assert!(transitioned);
        assert_eq!(sm.current_emotion, "angry");
        assert_eq!(sm.active_clip_name, "angry");
        assert!(sm.emotion_queue.is_empty());

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

    #[test]
    fn test_emotion_queue_push_and_fifo_consumption() {
        let registry = EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
            .expect("Failed to load emotions");
        let mut sm = StateMachine::new(&registry);

        assert_eq!(sm.current_emotion, "sit");
        assert!(sm.emotion_queue.is_empty());

        // Push happy, then naughty into queue
        sm.push_emotion("happy", &registry);
        sm.push_emotion("naughty", &registry);

        // Emotion does not change directly
        assert_eq!(sm.current_emotion, "sit");
        assert_eq!(sm.emotion_queue.len(), 2);
        assert_eq!(sm.emotion_queue[0], "happy");
        assert_eq!(sm.emotion_queue[1], "naughty");

        // First animation completes: consumes "happy"
        let changed = sm.on_animation_completed(&registry);
        assert!(changed);
        assert_eq!(sm.current_emotion, "happy");
        assert_eq!(sm.emotion_queue.len(), 1);
        assert_eq!(sm.emotion_queue[0], "naughty");

        // Second animation completes: consumes "naughty"
        let changed = sm.on_animation_completed(&registry);
        assert!(changed);
        assert_eq!(sm.current_emotion, "naughty");
        assert!(sm.emotion_queue.is_empty());

        // Third animation completes: queue empty, keeps "naughty"
        let changed = sm.on_animation_completed(&registry);
        assert!(!changed);
        assert_eq!(sm.current_emotion, "naughty");
        assert!(sm.emotion_queue.is_empty());
    }

    #[test]
    fn test_emotion_queue_limit_pop_last_on_overflow() {
        let registry = EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
            .expect("Failed to load emotions");
        let mut sm = StateMachine::new(&registry);

        // Push 3 emotions (capacity)
        sm.push_emotion("happy", &registry);
        sm.push_emotion("naughty", &registry);
        sm.push_emotion("angry", &registry);
        assert_eq!(sm.emotion_queue.len(), 3);
        assert_eq!(sm.emotion_queue[0], "happy");
        assert_eq!(sm.emotion_queue[1], "naughty");
        assert_eq!(sm.emotion_queue[2], "angry");

        // During next push: queue has 3 (more than 3 would occur).
        // Pop the last one ("angry") and push current one ("smirk").
        sm.push_emotion("smirk", &registry);
        assert_eq!(sm.emotion_queue.len(), 3);
        assert_eq!(sm.emotion_queue[0], "happy");
        assert_eq!(sm.emotion_queue[1], "naughty");
        assert_eq!(sm.emotion_queue[2], "smirk");

        // Push another ("sad"): pops "smirk" and pushes "sad"
        sm.push_emotion("sad", &registry);
        assert_eq!(sm.emotion_queue.len(), 3);
        assert_eq!(sm.emotion_queue[0], "happy");
        assert_eq!(sm.emotion_queue[1], "naughty");
        assert_eq!(sm.emotion_queue[2], "sad");

        // Consume queue step by step
        assert!(sm.on_animation_completed(&registry));
        assert_eq!(sm.current_emotion, "happy");

        assert!(sm.on_animation_completed(&registry));
        assert_eq!(sm.current_emotion, "naughty");

        assert!(sm.on_animation_completed(&registry));
        assert_eq!(sm.current_emotion, "sad");

        // Queue is now empty; keep current emotion "sad"
        assert!(!sm.on_animation_completed(&registry));
        assert_eq!(sm.current_emotion, "sad");
    }

    #[test]
    fn test_emotion_queue_empty_keeps_current_emotion() {
        let registry = EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
            .expect("Failed to load emotions");
        let mut sm = StateMachine::new(&registry);

        assert_eq!(sm.current_emotion, "sit");
        assert!(sm.emotion_queue.is_empty());

        // Repeated animation completions with empty queue keep "sit"
        assert!(!sm.on_animation_completed(&registry));
        assert_eq!(sm.current_emotion, "sit");
        assert!(!sm.on_animation_completed(&registry));
        assert_eq!(sm.current_emotion, "sit");
    }
}
