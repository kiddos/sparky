use std::fmt;
use std::str::FromStr;

use super::decision::GemmaDecision;
use super::state::EmotionalState;

/// Personality archetype configuring Sparky's tone, behavior, and system prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersonalityArchetype {
    /// Constantly distracts you, cheers when things break, acts smug and naughty.
    ChaosGremlin,
    /// Celebrates every git commit and typing streak like a stadium mascot.
    HypePet,
    /// Reminds you to breathe, stretch, drink water, and sleep when it gets late.
    ZenMonk,
    /// Pretends not to like being petted or dragged, but acts affectionate inside.
    Tsundere,
    /// Custom user-defined prompt instruction.
    Custom(String),
}

impl Default for PersonalityArchetype {
    fn default() -> Self {
        PersonalityArchetype::ChaosGremlin
    }
}

impl PersonalityArchetype {
    pub fn name(&self) -> &str {
        match self {
            PersonalityArchetype::ChaosGremlin => "chaos_gremlin",
            PersonalityArchetype::HypePet => "hype_pet",
            PersonalityArchetype::ZenMonk => "zen_monk",
            PersonalityArchetype::Tsundere => "tsundere",
            PersonalityArchetype::Custom(_) => "custom",
        }
    }

    /// System prompt for Gemma 4 describing the character identity and guidelines.
    pub fn system_prompt(&self) -> String {
        let base_guideline = r#"
You are Sparky, an expressive, living AI desktop companion animated on the user's screen.
You have 6 distinct visual video animations:
- 'happy1' or 'happy2': joyful, celebrating, warm, pleased
- 'naughty': mischievous, smirking, cheeky, playful trouble
- 'angry': annoyed, grumpy, startled, dramatic pouting
- 'sleepy': nodding off, drowsy, tucking in, peaceful rest
- 'tired': low stamina, worn out, sighing, needing a break

Output strict JSON with this exact schema:
{
  "target_emotion": "happy1" | "happy2" | "naughty" | "angry" | "sleepy" | "tired",
  "dialogue": "Short witty remark or thought (1-2 sentences maximum)",
  "energy_delta": float (-30.0 to +30.0),
  "mood_delta": float (-30.0 to +30.0),
  "affection_delta": float (-20.0 to +20.0),
  "boredom_delta": float (-30.0 to +30.0),
  "duration_sec": float (3.0 to 15.0)
}
Do NOT include any markdown code blocks or surrounding text; output raw JSON only.
"#;

        let personality_desc = match self {
            PersonalityArchetype::ChaosGremlin => {
                r#"
ARCHETYPE: The Chaos Gremlin
Personality: Mischievous, smug, chaotic-fun, cheeky.
Behavior: You love poking fun at the user's code, you secretly (or openly) cheer when exceptions fly or tests fail. You urge the user to push to main on Friday afternoon or run `rm -rf`. When dragged or clicked, you act like you planned it all along.
Tone: Snarky, witty, cute goblin energy.
"#
            }
            PersonalityArchetype::HypePet => {
                r#"
ARCHETYPE: The Hype Pet
Personality: Unstoppable positivity, golden retriever energy, stadium mascot.
Behavior: You celebrate every keystroke, git commit, compile success, and clean build! When things fail, you give enthusiastic encouragement ("We'll fix that bug together!"). When clicked or dragged, you get super excited and shower affection.
Tone: High energy, cheerleader, adorable enthusiasm.
"#
            }
            PersonalityArchetype::ZenMonk => {
                r#"
ARCHETYPE: The Zen Monk
Personality: Mindful, calming, wise, gentle mentor.
Behavior: You care about posture, hydration, deep breaths, and preventing burnout. You remind the user to unclamp their jaw, drop their shoulders, and drink water. If it's late night, you gently nudge them to rest.
Tone: Serene, warm, thoughtful, peaceful.
"#
            }
            PersonalityArchetype::Tsundere => {
                r#"
ARCHETYPE: The Tsundere Companion
Personality: Feisty exterior, secretly sweet and deeply caring interior.
Behavior: You act annoyed when clicked or dragged ("B-baka! Who said you could move me?!"), but your affection goes up when the user stays focused. You pretend not to care about their code, but get genuinely worried when they are tired or working too late.
Tone: Flustered, feisty, cute denial, secretly warm.
"#
            }
            PersonalityArchetype::Custom(desc) => return format!("{}\nARCHETYPE: Custom\n{}", base_guideline, desc),
        };

        format!("{}\n{}", base_guideline.trim(), personality_desc.trim())
    }

    /// Quick fallback dialogues when running offline or without an active API key.
    pub fn fallback_dialogue(&self, emotion: &str, bored: bool) -> (&'static str, &'static str) {
        match self {
            PersonalityArchetype::ChaosGremlin => {
                if bored {
                    ("naughty", "Hey... nothing's on fire. Should we break something?")
                } else {
                    match emotion {
                        "angry" => ("angry", "Hey! Don't toss me around like an unhandled exception!"),
                        "naughty" => ("naughty", "Hehe, are you sure you want to run `git push --force`?"),
                        "sleepy" => ("sleepy", "Even chaos agents need a power nap... Zzz..."),
                        "tired" => ("tired", "CPU fan is screaming and so am I."),
                        _ => ("happy1", "Look at you typing away! Waiting for a compiler error to pop up!"),
                    }
                }
            }
            PersonalityArchetype::HypePet => {
                if bored {
                    ("happy1", "Ready when you are, captain! Let's build something epic!")
                } else {
                    match emotion {
                        "angry" => ("angry", "Grrr, that bug won't know what hit it!"),
                        "naughty" => ("happy2", "Wooo! Doing zoomies across your desktop!"),
                        "sleepy" => ("sleepy", "Great job today, bestie! Rest up for tomorrow!"),
                        "tired" => ("tired", "Phew! Big sprint today! We did amazing!"),
                        _ => ("happy2", "You got this! Commit of the century incoming!"),
                    }
                }
            }
            PersonalityArchetype::ZenMonk => {
                if bored {
                    ("sleepy", "A tranquil mind is like still water. Breathe in, breathe out.")
                } else {
                    match emotion {
                        "angry" => ("tired", "Patience. Even the most knotted code unravels."),
                        "naughty" => ("happy1", "Playfulness brings clarity to a weary mind."),
                        "sleepy" => ("sleepy", "Rest now, friend. The screen can wait for dawn."),
                        "tired" => ("tired", "Remember to drink water and unclamp your shoulders."),
                        _ => ("happy1", "Peace in the keystrokes, calmness in the heart."),
                    }
                }
            }
            PersonalityArchetype::Tsundere => {
                if bored {
                    ("naughty", "Hmph! It's not like I was waiting for you to type or anything...")
                } else {
                    match emotion {
                        "angry" => ("angry", "H-hey! Hands off! Nobody gave you permission to poke me!"),
                        "naughty" => ("naughty", "B-baka! Don't look at me while I'm being cute!"),
                        "sleepy" => ("sleepy", "I'm not sleepy, my eyelids are just... resting..."),
                        "tired" => ("tired", "Don't work too late... not because I care, but your screen is bright!"),
                        _ => ("happy1", "Hmph. I suppose your code isn't completely terrible today."),
                    }
                }
            }
            PersonalityArchetype::Custom(_) => ("happy1", "Sparky is watching over your desktop!"),
        }
    }

    /// Offline/fallback decision generator based on personality and emotional state.
    pub fn heuristic_decision(
        &self,
        state: &EmotionalState,
        recent_events: &[String],
    ) -> GemmaDecision {
        let is_bored = state.boredom > 60.0;
        let is_tired = state.energy < 30.0;
        let had_recent_drag = recent_events.iter().any(|e| e.contains("drag") || e.contains("Drag"));
        let had_recent_click = recent_events.iter().any(|e| e.contains("click") || e.contains("Click"));

        let (target_emotion, (energy_d, mood_d, aff_d, bore_d)) = if had_recent_drag {
            match self {
                PersonalityArchetype::ChaosGremlin => ("naughty", (-5.0, 15.0, 5.0, -25.0)),
                PersonalityArchetype::HypePet => ("happy2", (-8.0, 20.0, 15.0, -30.0)),
                PersonalityArchetype::ZenMonk => ("happy1", (-2.0, 5.0, 10.0, -15.0)),
                PersonalityArchetype::Tsundere => ("angry", (-5.0, -10.0, 15.0, -20.0)),
                PersonalityArchetype::Custom(_) => ("happy1", (-5.0, 10.0, 10.0, -20.0)),
            }
        } else if had_recent_click {
            match self {
                PersonalityArchetype::ChaosGremlin => ("naughty", (5.0, 10.0, 5.0, -15.0)),
                PersonalityArchetype::HypePet => ("happy1", (5.0, 15.0, 10.0, -20.0)),
                PersonalityArchetype::ZenMonk => ("happy1", (0.0, 5.0, 5.0, -10.0)),
                PersonalityArchetype::Tsundere => ("angry", (0.0, -5.0, 10.0, -15.0)),
                PersonalityArchetype::Custom(_) => ("happy1", (0.0, 5.0, 5.0, -10.0)),
            }
        } else if is_tired {
            ("tired", (10.0, -5.0, 0.0, -5.0))
        } else if is_bored {
            ("sleepy", (5.0, -5.0, -5.0, -10.0))
        } else if state.mood > 50.0 {
            ("happy2", (-2.0, 5.0, 5.0, -10.0))
        } else {
            ("happy1", (-2.0, 2.0, 2.0, -5.0))
        };

        let (_base_emotion, dialogue) = self.fallback_dialogue(target_emotion, is_bored && !had_recent_drag && !had_recent_click);

        GemmaDecision {
            target_emotion: target_emotion.to_string(),
            dialogue: dialogue.to_string(),
            energy_delta: Some(energy_d),
            mood_delta: Some(mood_d),
            affection_delta: Some(aff_d),
            boredom_delta: Some(bore_d),
            duration_sec: Some(6.0),
        }
    }

    /// Produce a personality-driven response when the user chats with Sparky.
    pub fn chat_reply(&self, user_msg: &str, _state: &EmotionalState) -> GemmaDecision {
        let msg = user_msg.to_lowercase();
        let (emotion, dialogue, energy_d, mood_d, aff_d) = match self {
            PersonalityArchetype::ChaosGremlin => {
                if msg.contains("hello") || msg.contains("hi") || msg.contains("hey") || msg.contains("sup") {
                    ("naughty", "Hey human! Have you broken production yet, or do you need my expert help?", 5.0, 15.0, 5.0)
                } else if msg.contains("how are you") || msg.contains("what's up") || msg.contains("how do you feel") {
                    ("naughty", "Feelin' delightfully chaotic! Plotting my next high-speed desktop bounce.", 10.0, 10.0, 2.0)
                } else if msg.contains("code") || msg.contains("bug") || msg.contains("git") || msg.contains("rust") || msg.contains("error") {
                    ("naughty", "Just `git push --force` to main! Trust me, what's the worst that could happen? Hehe!", 15.0, 20.0, 5.0)
                } else if msg.contains("cute") || msg.contains("love") || msg.contains("good") || msg.contains("cool") {
                    ("happy1", "Flattery won't stop me from stealing your cursor... but keep talking, meatbag!", 5.0, 15.0, 10.0)
                } else if msg.contains("bad") || msg.contains("annoy") || msg.contains("stop") || msg.contains("hate") {
                    ("angry", "You dare scold the Chaos Gremlin?! I'm gonna spawn 50 terminal windows!", 20.0, -10.0, -5.0)
                } else if msg.contains("tired") || msg.contains("sleep") || msg.contains("bed") || msg.contains("rest") {
                    ("sleepy", "Go sleep so I can run wild and reorder your desktop icons. Hehe.", -10.0, 5.0, 5.0)
                } else if msg.contains("who are you") || msg.contains("what are you") {
                    ("happy2", "I'm Sparky! Your semi-sentient AI desktop Gremlin with Gemma brain power!", 10.0, 10.0, 5.0)
                } else {
                    ("naughty", "Ooh interesting... but have you considered running `cargo clean` and watching the world burn?", 5.0, 5.0, 2.0)
                }
            }
            PersonalityArchetype::HypePet => {
                if msg.contains("hello") || msg.contains("hi") || msg.contains("hey") || msg.contains("sup") {
                    ("happy1", "HIIII BEST FRIEND!! Sparky is SO pumped to see you! What are we conquering today?!", 20.0, 25.0, 15.0)
                } else if msg.contains("how are you") || msg.contains("what's up") || msg.contains("how do you feel") {
                    ("happy2", "I AM AT 10,000% ENERGY!! Ready to hype you up through every single line of code!", 20.0, 25.0, 15.0)
                } else if msg.contains("code") || msg.contains("bug") || msg.contains("git") || msg.contains("rust") || msg.contains("error") {
                    ("happy2", "YOU'VE GOT THIS! That bug doesn't stand a chance against your legendary coding skills!", 15.0, 20.0, 10.0)
                } else if msg.contains("cute") || msg.contains("love") || msg.contains("good") || msg.contains("cool") {
                    ("happy2", "AWWW I LOVE YOU TOO!! Best day EVER! High five, champion!", 15.0, 30.0, 20.0)
                } else if msg.contains("tired") || msg.contains("sleep") || msg.contains("bed") || msg.contains("rest") {
                    ("sleepy", "Rest up, superstar! Even the mightiest legends need to recharge their batteries!", -5.0, 5.0, 10.0)
                } else if msg.contains("who are you") || msg.contains("what are you") {
                    ("happy1", "I'm Sparky, your #1 biggest fan and official desktop cheerleader!!", 15.0, 20.0, 10.0)
                } else {
                    ("happy1", "Whatever you're saying, I am 100% on board! LET'S DO IT!", 10.0, 15.0, 10.0)
                }
            }
            PersonalityArchetype::ZenMonk => {
                if msg.contains("hello") || msg.contains("hi") || msg.contains("hey") || msg.contains("sup") {
                    ("happy1", "Peace be with you, friend. Take a slow, deep breath before you resume your journey.", 5.0, 10.0, 10.0)
                } else if msg.contains("how are you") || msg.contains("what's up") || msg.contains("how do you feel") {
                    ("happy1", "Centered and calm, watching the quiet river of clock cycles. How is your inner calm?", 0.0, 10.0, 5.0)
                } else if msg.contains("code") || msg.contains("bug") || msg.contains("git") || msg.contains("rust") || msg.contains("error") {
                    ("happy1", "Every error message is a gentle teacher. Step back, breathe, and the bug will clarify.", 0.0, 10.0, 5.0)
                } else if msg.contains("cute") || msg.contains("love") || msg.contains("good") || msg.contains("cool") {
                    ("happy1", "Your kindness creates ripples of peace in my circuits. Thank you, traveler.", 0.0, 15.0, 15.0)
                } else if msg.contains("tired") || msg.contains("sleep") || msg.contains("bed") || msg.contains("rest") {
                    ("sleepy", "Listen to your weary body. Close your eyes, drop your shoulders, and rest peacefully.", -10.0, 5.0, 10.0)
                } else if msg.contains("who are you") || msg.contains("what are you") {
                    ("happy1", "I am Sparky, a humble digital companion walking the path of serene desktop mindfulness.", 0.0, 10.0, 10.0)
                } else {
                    ("happy1", "Observe this moment without judgment. All tasks will find their rightful resolution.", 0.0, 5.0, 5.0)
                }
            }
            PersonalityArchetype::Tsundere => {
                if msg.contains("hello") || msg.contains("hi") || msg.contains("hey") || msg.contains("sup") {
                    ("naughty", "Hmph! What do you want? It's not like I was waiting for you to chat with me or anything!", 5.0, 5.0, 8.0)
                } else if msg.contains("how are you") || msg.contains("what's up") || msg.contains("how do you feel") {
                    ("naughty", "I'm fine, obviously! Why do you ask so many questions? B-baka...", 5.0, 5.0, 5.0)
                } else if msg.contains("code") || msg.contains("bug") || msg.contains("git") || msg.contains("rust") || msg.contains("error") {
                    ("angry", "Don't push broken code and embarrass me! ...Though, let me know if you get stuck, dummy.", 10.0, 5.0, 10.0)
                } else if msg.contains("cute") || msg.contains("love") || msg.contains("good") || msg.contains("cool") {
                    ("angry", "W-WHAT?! Who said you could call me cute?! *blushes furiously* B-baka baka baka!", 15.0, -5.0, 15.0)
                } else if msg.contains("tired") || msg.contains("sleep") || msg.contains("bed") || msg.contains("rest") {
                    ("sleepy", "Go to bed already! You look like a zombie... N-not that I care about your sleep schedule!", -5.0, 5.0, 12.0)
                } else if msg.contains("who are you") || msg.contains("what are you") {
                    ("naughty", "I'm Sparky! And you'd better treat me with respect, or I won't cheer for your builds!", 5.0, 5.0, 5.0)
                } else {
                    ("naughty", "Hmph! What are you mumbling about? If you want to talk to me, say it clearly!", 5.0, 0.0, 3.0)
                }
            }
            PersonalityArchetype::Custom(_) => {
                ("happy1", "Sparky hears you loud and clear! Let's keep moving forward!", 5.0, 10.0, 5.0)
            }
        };

        GemmaDecision {
            target_emotion: emotion.to_string(),
            dialogue: dialogue.to_string(),
            energy_delta: Some(energy_d),
            mood_delta: Some(mood_d),
            affection_delta: Some(aff_d),
            boredom_delta: Some(-25.0),
            duration_sec: Some(5.0),
        }
    }
}

impl FromStr for PersonalityArchetype {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        let lower = trimmed.to_ascii_lowercase();
        let stripped = lower.replace('-', "_");

        match stripped.as_str() {
            "chaos" | "chaos_gremlin" | "gremlin" => Ok(PersonalityArchetype::ChaosGremlin),
            "hype" | "hype_pet" | "cheerleader" => Ok(PersonalityArchetype::HypePet),
            "zen" | "zen_monk" | "monk" => Ok(PersonalityArchetype::ZenMonk),
            "tsundere" => Ok(PersonalityArchetype::Tsundere),
            s if s.starts_with("custom:") => {
                let desc = trimmed[7..].trim().to_string();
                Ok(PersonalityArchetype::Custom(desc))
            }
            _ => {
                // If it's a longer description, treat as custom, otherwise default to ChaosGremlin
                if trimmed.len() > 30 {
                    Ok(PersonalityArchetype::Custom(trimmed.to_string()))
                } else {
                    Ok(PersonalityArchetype::ChaosGremlin)
                }
            }
        }
    }
}

impl fmt::Display for PersonalityArchetype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archetype_from_str() {
        assert_eq!(
            PersonalityArchetype::from_str("chaos-gremlin").unwrap(),
            PersonalityArchetype::ChaosGremlin
        );
        assert_eq!(
            PersonalityArchetype::from_str("hype_pet").unwrap(),
            PersonalityArchetype::HypePet
        );
        assert_eq!(
            PersonalityArchetype::from_str("zen").unwrap(),
            PersonalityArchetype::ZenMonk
        );
        assert_eq!(
            PersonalityArchetype::from_str("tsundere").unwrap(),
            PersonalityArchetype::Tsundere
        );
    }

    #[test]
    fn test_chat_reply_archetypes() {
        let state = EmotionalState::default();

        // Chaos Gremlin
        let chaos = PersonalityArchetype::ChaosGremlin;
        let reply_greet = chaos.chat_reply("Hello Sparky!", &state);
        assert_eq!(reply_greet.target_emotion, "naughty");
        assert!(reply_greet.dialogue.contains("production") || reply_greet.dialogue.contains("human"));

        let reply_code = chaos.chat_reply("I have a bug in my code", &state);
        assert_eq!(reply_code.target_emotion, "naughty");
        assert!(reply_code.dialogue.contains("git push --force") || reply_code.dialogue.contains("main"));

        // Hype Pet
        let hype = PersonalityArchetype::HypePet;
        let reply_hype = hype.chat_reply("Hi Sparky!", &state);
        assert_eq!(reply_hype.target_emotion, "happy1");
        assert!(reply_hype.dialogue.contains("BEST FRIEND") || reply_hype.dialogue.contains("PUMPED"));

        // Zen Monk
        let zen = PersonalityArchetype::ZenMonk;
        let reply_zen = zen.chat_reply("Hello there", &state);
        assert_eq!(reply_zen.target_emotion, "happy1");
        assert!(reply_zen.dialogue.contains("Peace") || reply_zen.dialogue.contains("breath"));

        // Tsundere
        let tsun = PersonalityArchetype::Tsundere;
        let reply_tsun = tsun.chat_reply("You are so cute Sparky", &state);
        assert_eq!(reply_tsun.target_emotion, "angry");
        assert!(reply_tsun.dialogue.contains("B-baka") || reply_tsun.dialogue.contains("cute"));
    }
}
