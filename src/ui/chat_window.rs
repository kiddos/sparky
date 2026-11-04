use crate::state::emotion::EmotionRegistry;
use egui::{Color32, Context, RichText, ScrollArea, TextEdit, Vec2};

pub struct ChatState {
    pub is_visible: bool,
    pub current_input: String,
    pub messages: Vec<(String, String)>, // (sender, message)
    pub is_generating: bool,
    pub current_emotion_id: String,
    pub focus_input: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatState {
    pub fn new() -> Self {
        Self {
            is_visible: false,
            current_input: String::new(),
            messages: vec![(
                "Sparky".to_string(),
                "Meow! I'm Sparky. How are you today?".to_string(),
            )],
            is_generating: false,
            current_emotion_id: "sit".to_string(),
            focus_input: true,
        }
    }

    pub fn draw(
        &mut self,
        ctx: &Context,
        registry: &EmotionRegistry,
        mut on_send: impl FnMut(&str),
        mut on_trigger_emotion: impl FnMut(&str),
    ) {
        if !self.is_visible {
            return;
        }

        let emotion_def = registry.get(&self.current_emotion_id);
        let badge_text = emotion_def
            .map(|e| format!("{} {}", e.emoji, e.name))
            .unwrap_or_else(|| "🐱 Sparky".to_string());

        egui::Window::new("🐾 Sparky Chat")
            .default_size(Vec2::new(272.0, 260.0))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0))
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Mood:").strong());
                    ui.colored_label(
                        Color32::from_rgb(255, 180, 50),
                        RichText::new(badge_text).strong(),
                    );
                    if self.is_generating {
                        ui.colored_label(
                            Color32::from_rgb(180, 180, 180),
                            RichText::new("thinking...").italics(),
                        );
                    }
                });
                ui.separator();

                ui.vertical(|ui| {
                    ScrollArea::vertical()
                        .max_height(130.0)
                        .auto_shrink([false; 2])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for (sender, text) in &self.messages {
                                if sender == "Sparky" {
                                    ui.colored_label(
                                        Color32::from_rgb(255, 170, 0),
                                        format!("{}:", sender),
                                    );
                                } else if sender == "System" {
                                    ui.colored_label(
                                        Color32::from_rgb(255, 80, 80),
                                        format!("{}:", sender),
                                    );
                                } else {
                                    ui.colored_label(
                                        Color32::from_rgb(100, 200, 255),
                                        format!("{}:", sender),
                                    );
                                }
                                ui.label(RichText::new(text).size(13.0));
                                ui.add_space(3.0);
                            }
                        });

                    ui.separator();

                    // Dynamic Emotion Suggestion Chips
                    ui.horizontal_wrapped(|ui| {
                        for emotion in registry.all_emotions() {
                            let chip_label = format!("{} {}", emotion.emoji, emotion.name);
                            if ui.small_button(chip_label).clicked() {
                                on_trigger_emotion(&emotion.id);
                            }
                        }
                    });

                    ui.separator();

                    // Text Input Box
                    ui.horizontal(|ui| {
                        let send_button_width = 46.0;
                        let item_spacing = ui.spacing().item_spacing.x;
                        let text_width =
                            (ui.available_width() - send_button_width - item_spacing).max(60.0);
                        let response = ui.add_enabled(
                            !self.is_generating,
                            TextEdit::singleline(&mut self.current_input)
                                .hint_text("Talk to Sparky...")
                                .desired_width(text_width),
                        );

                        if self.focus_input && !self.is_generating {
                            response.request_focus();
                            self.focus_input = false;
                        }

                        let enter_pressed =
                            !self.is_generating && ui.input(|i| i.key_pressed(egui::Key::Enter));

                        let send_clicked = ui
                            .add_enabled(!self.is_generating, egui::Button::new("Send"))
                            .clicked();

                        if (send_clicked || enter_pressed) && !self.current_input.trim().is_empty()
                        {
                            let input = self.current_input.trim().to_string();
                            self.messages.push(("You".to_string(), input.clone()));
                            self.messages.push(("Sparky".to_string(), String::new()));
                            self.current_input.clear();
                            self.is_generating = true;
                            self.focus_input = true;
                            on_send(&input);
                        }
                    });
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_state_enter_key_submits_message() {
        let mut chat_state = ChatState::new();
        chat_state.is_visible = true;
        chat_state.current_input = "Hello Sparky".to_string();

        let registry =
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations").unwrap();

        let ctx = Context::default();
        let mut sent_prompt = None;

        // Frame 1: TextEdit is rendered and automatically requests focus
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(288.0, 466.0),
                )),
                ..Default::default()
            },
            |ctx| {
                chat_state.draw(
                    ctx,
                    &registry,
                    |p| sent_prompt = Some(p.to_string()),
                    |_| {},
                );
            },
        );

        // Frame 2: User presses Enter key to send
        let mut enter_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(288.0, 466.0),
            )),
            ..Default::default()
        };
        enter_input.events.push(egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            modifiers: egui::Modifiers::default(),
            repeat: false,
        });

        let _ = ctx.run(enter_input, |ctx| {
            chat_state.draw(
                ctx,
                &registry,
                |p| sent_prompt = Some(p.to_string()),
                |_| {},
            );
        });

        assert_eq!(sent_prompt.as_deref(), Some("Hello Sparky"));
        assert_eq!(chat_state.current_input, "");
        assert!(chat_state.is_generating);
        assert!(chat_state.focus_input);
    }
}
