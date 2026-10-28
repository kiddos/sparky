use serde::{Deserialize, Serialize};

/// Psychological model of Sparky representing dynamic emotion vectors.
///
/// Vectors:
/// - energy: 0.0 to 100.0 (physical energy level)
/// - mood: -100.0 (sad/frustrated) to +100.0 (delighted/joyful)
/// - affection: 0.0 (distant) to 100.0 (adoring/attached)
/// - boredom: 0.0 (fully engaged) to 100.0 (unstimulated/sleepy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub energy: f32,
    pub mood: f32,
    pub affection: f32,
    pub boredom: f32,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            energy: 85.0,
            mood: 40.0,
            affection: 50.0,
            boredom: 10.0,
        }
    }
}

impl EmotionalState {
    #[allow(dead_code)]
    pub fn new(energy: f32, mood: f32, affection: f32, boredom: f32) -> Self {
        let mut state = Self {
            energy,
            mood,
            affection,
            boredom,
        };
        state.clamp();
        state
    }

    /// Clamp all vectors to their defined valid ranges.
    pub fn clamp(&mut self) {
        self.energy = self.energy.clamp(0.0, 100.0);
        self.mood = self.mood.clamp(-100.0, 100.0);
        self.affection = self.affection.clamp(0.0, 100.0);
        self.boredom = self.boredom.clamp(0.0, 100.0);
    }

    /// Apply vector deltas returned by Gemma 4 or local heuristics.
    pub fn apply_delta(
        &mut self,
        energy_delta: f32,
        mood_delta: f32,
        affection_delta: f32,
        boredom_delta: f32,
    ) {
        self.energy += energy_delta;
        self.mood += mood_delta;
        self.affection += affection_delta;
        self.boredom += boredom_delta;
        self.clamp();
    }

    /// Natural passive decay over time.
    ///
    /// - Boredom slowly increases if no interactions occur.
    /// - Mood drifts gently toward neutral baseline (0.0).
    /// - Energy slowly depletes.
    pub fn decay_over_time(&mut self, elapsed_secs: f32) {
        let factor = (elapsed_secs / 60.0).min(5.0); // scale per minute
        self.boredom += factor * 3.5;
        self.energy -= factor * 1.5;

        if self.mood > 10.0 {
            self.mood -= factor * 2.0;
        } else if self.mood < -10.0 {
            self.mood += factor * 2.0;
        }

        self.clamp();
    }

    /// Generate a concise summary suitable for LLM context prompts.
    pub fn prompt_summary(&self) -> String {
        format!(
            "Energy: {:.0}/100, Mood: {:+.0}/100, Affection: {:.0}/100, Boredom: {:.0}/100",
            self.energy, self.mood, self.affection, self.boredom
        )
    }

    /// Determine primary qualitative state label.
    pub fn dominant_vibe(&self) -> &'static str {
        if self.energy < 25.0 {
            "exhausted"
        } else if self.boredom > 75.0 {
            "bored / drowsy"
        } else if self.mood > 60.0 {
            "ecstatic"
        } else if self.mood < -40.0 {
            "grumpy"
        } else if self.affection > 75.0 {
            "loving"
        } else {
            "content"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamping() {
        let state = EmotionalState::new(150.0, -200.0, 120.0, -10.0);
        assert_eq!(state.energy, 100.0);
        assert_eq!(state.mood, -100.0);
        assert_eq!(state.affection, 100.0);
        assert_eq!(state.boredom, 0.0);
    }

    #[test]
    fn test_apply_delta() {
        let mut state = EmotionalState::default();
        state.apply_delta(-10.0, 25.0, 5.0, -5.0);
        assert_eq!(state.energy, 75.0);
        assert_eq!(state.mood, 65.0);
        assert_eq!(state.affection, 55.0);
        assert_eq!(state.boredom, 5.0);
    }

    #[test]
    fn test_decay() {
        let mut state = EmotionalState::new(80.0, 50.0, 50.0, 0.0);
        state.decay_over_time(60.0); // 1 minute
        assert!(state.boredom > 0.0);
        assert!(state.energy < 80.0);
        assert!(state.mood < 50.0);
    }
}
