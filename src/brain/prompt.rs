use crate::state::emotion::EmotionRegistry;

pub fn build_system_prompt(registry: &EmotionRegistry) -> String {
    let actions_guideline = registry
        .active_emotions()
        .map(|e| format!("<action>{}</action> - {}", e.id, e.prompt_description))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "<start_of_turn>system\n\
        You are Sparky, an intelligent, charming, and playful desktop kitten.\n\
        You live on the user's desktop. Your responses should be concise, playful, and express feline instincts.\n\
        At the very end of your response, ALWAYS append an action tag reflecting your emotional state:\n\
        {}\n\
        Keep dialogue under 40 words.<end_of_turn>\n",
        actions_guideline
    )
}

pub fn format_chat_prompt(
    registry: &EmotionRegistry,
    history: &[(String, String)],
    user_prompt: &str,
) -> String {
    let mut prompt = build_system_prompt(registry);
    for (sender, msg) in history {
        let role = if sender == "Sparky" { "model" } else { "user" };
        prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg));
    }
    prompt.push_str(&format!(
        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        user_prompt
    ));
    prompt
}

pub fn extract_action_tag(text: &str, registry: &EmotionRegistry) -> (String, Option<String>) {
    let mut detected_action = None;
    for emotion in registry.all_emotions() {
        let tag = format!("<action>{}</action>", emotion.id);
        if text.contains(&tag) {
            detected_action = Some(emotion.id.clone());
            break;
        }
    }

    let mut clean_text = text.to_string();
    for emotion in registry.all_emotions() {
        let tag = format!("<action>{}</action>", emotion.id);
        clean_text = clean_text.replace(&tag, "");
    }

    (clean_text.trim().to_string(), detected_action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_formatting_and_tag_extraction() {
        let registry = EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
            .expect("Failed to load emotions");

        let prompt = build_system_prompt(&registry);
        assert!(prompt.contains("<action>happy</action>"));
        assert!(prompt.contains("<action>angry</action>"));
        assert!(prompt.contains("<action>sad</action>"));

        let sample = "Meow! Thank you for the treat! <action>happy</action>";
        let (clean, action) = extract_action_tag(sample, &registry);
        assert_eq!(clean, "Meow! Thank you for the treat!");
        assert_eq!(action, Some("happy".to_string()));

        let sample_no_action = "Just meowing peacefully.";
        let (clean2, action2) = extract_action_tag(sample_no_action, &registry);
        assert_eq!(clean2, "Just meowing peacefully.");
        assert_eq!(action2, None);
    }
}
