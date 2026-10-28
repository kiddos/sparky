pub mod client;
pub mod decision;
pub mod local;
pub mod personality;
pub mod state;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::{Duration, Instant};

use log::{debug, info, warn};

pub use client::GemmaClient;
pub use decision::GemmaDecision;
#[allow(unused_imports)]
pub use local::LocalGemmaEngine;
pub use personality::PersonalityArchetype;
pub use state::EmotionalState;

/// Selection of the intelligence engine backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainEngineMode {
    /// Gemma 3 270M running locally in-process via Candle (CPU / low resource).
    Local,
    /// Gemma 4 API (Google AI Studio or OpenAI-compatible endpoint).
    Api,
}

/// Configuration for launching the Sparky Brain background service.
#[derive(Clone)]
pub struct BrainConfig {
    pub engine_mode: BrainEngineMode,
    pub personality: PersonalityArchetype,
    pub interval: Duration,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub api_model: String,
    #[allow(dead_code)]
    pub local_model_dir: Option<PathBuf>,
}

/// High-level events from Sparky UI/environment that influence the brain.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum BrainEvent {
    Started,
    Clicked { x: f64, y: f64 },
    Dragged { dx: f64, dy: f64 },
    DragEnded,
    AnimationCompleted { name: String },
    Custom(String),
}

/// Commands sent to the background brain worker thread.
#[allow(dead_code)]
pub enum BrainCommand {
    Event(BrainEvent),
    UserChat(String),
    TriggerNow,
    ForceEmotion(String),
    Shutdown,
}

/// Handle held by the main application thread to interact with the Gemma Brain.
pub struct BrainHandle {
    command_tx: tokio::sync::mpsc::UnboundedSender<BrainCommand>,
    decision_rx: Receiver<GemmaDecision>,
}

impl BrainHandle {
    /// Launch the background Gemma Brain service running on a Tokio runtime.
    pub fn spawn(config: BrainConfig) -> Self {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<BrainCommand>();
        let (dec_tx, dec_rx) = channel::<GemmaDecision>();

        thread::Builder::new()
            .name("sparky-gemma-tokio".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .expect("Failed to initialize Tokio runtime for Sparky Brain");

                rt.block_on(async move {
                    run_tokio_brain_worker(config, cmd_rx, dec_tx).await;
                });
            })
            .expect("Failed to spawn Sparky brain thread");

        Self {
            command_tx: cmd_tx,
            decision_rx: dec_rx,
        }
    }

    /// Send an interaction event to the brain.
    pub fn send_event(&self, event: BrainEvent) {
        let _ = self.command_tx.send(BrainCommand::Event(event));
    }

    /// Send a user chat message to the brain.
    pub fn send_chat(&self, message: impl Into<String>) {
        let _ = self.command_tx.send(BrainCommand::UserChat(message.into()));
    }

    /// Request an immediate Gemma thought/decision without waiting for timer.
    #[allow(dead_code)]
    pub fn trigger_now(&self) {
        let _ = self.command_tx.send(BrainCommand::TriggerNow);
    }

    /// Force an immediate emotion transition.
    #[allow(dead_code)]
    pub fn force_emotion(&self, emotion: impl Into<String>) {
        let _ = self.command_tx.send(BrainCommand::ForceEmotion(emotion.into()));
    }

    /// Non-blocking check for new decisions from Gemma.
    pub fn try_recv_decision(&self) -> Option<GemmaDecision> {
        self.decision_rx.try_recv().ok()
    }
}

impl Drop for BrainHandle {
    fn drop(&mut self) {
        let _ = self.command_tx.send(BrainCommand::Shutdown);
    }
}

/// The asynchronous brain worker loop running inside Tokio.
async fn run_tokio_brain_worker(
    config: BrainConfig,
    mut command_rx: tokio::sync::mpsc::UnboundedReceiver<BrainCommand>,
    decision_tx: Sender<GemmaDecision>,
) {
    info!(
        "Sparky Brain Tokio worker running (engine={:?}, model='{}', personality='{}', interval={:?})",
        config.engine_mode,
        config.api_model,
        config.personality.name(),
        config.interval
    );

    let mut state = EmotionalState::default();
    let mut recent_events: VecDeque<String> = VecDeque::with_capacity(20);
    let mut last_consultation = Instant::now();
    let mut last_decay = Instant::now();
    let mut last_interaction = Instant::now();
    let mut pending_immediate_trigger = false;

    // Send initial greeting thought immediately
    let initial_decision = config.personality.heuristic_decision(
        &state,
        &["Sparky woke up on the desktop!".to_string()],
    );
    apply_decision_deltas(&mut state, &initial_decision);
    let _ = decision_tx.send(initial_decision);

    let mut api_client: Option<GemmaClient> = if config.engine_mode == BrainEngineMode::Api {
        Some(GemmaClient::new(
            config.api_key.clone(),
            config.api_base.clone(),
            Some(config.api_model.clone()),
            config.personality.clone(),
        ))
    } else {
        None
    };

    let mut interval_timer = tokio::time::interval(config.interval);
    // Discard the initial immediate tick
    interval_timer.tick().await;

    let mut decay_interval = tokio::time::interval(Duration::from_secs(5));
    decay_interval.tick().await;

    loop {
        tokio::select! {
            biased;

            maybe_cmd = command_rx.recv() => {
                match maybe_cmd {
                    Some(BrainCommand::Shutdown) | None => {
                        debug!("Sparky brain worker received Shutdown command or channel closed.");
                        break;
                    }
                    Some(BrainCommand::ForceEmotion(emotion)) => {
                        let (target, dialogue) = config.personality.fallback_dialogue(&emotion, false);
                        let decision = GemmaDecision::new(target, dialogue);
                        let _ = decision_tx.send(decision);
                    }
                    Some(BrainCommand::TriggerNow) => {
                        pending_immediate_trigger = true;
                    }
                    Some(BrainCommand::UserChat(msg)) => {
                        last_interaction = Instant::now();
                        let event_desc = format!("User said in chat: \"{}\"", msg);
                        if recent_events.len() >= 15 {
                            recent_events.pop_front();
                        }
                        recent_events.push_back(event_desc);

                        state.apply_delta(2.0, 5.0, 8.0, -25.0);

                        let history: Vec<String> = recent_events.iter().cloned().collect();
                        let decision = decide_chat(
                            &config,
                            &mut api_client,
                            &state,
                            &history,
                            &msg,
                        ).await;

                        apply_decision_deltas(&mut state, &decision);
                        let _ = decision_tx.send(decision);
                        last_consultation = Instant::now();
                    }
                    Some(BrainCommand::Event(event)) => {
                        last_interaction = Instant::now();
                        let event_desc = format_event(&event);
                        if recent_events.len() >= 15 {
                            recent_events.pop_front();
                        }
                        recent_events.push_back(event_desc);

                        match event {
                            BrainEvent::Clicked { .. } | BrainEvent::DragEnded => {
                                if last_consultation.elapsed() >= Duration::from_secs(4) {
                                    pending_immediate_trigger = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            _ = decay_interval.tick() => {
                let decay_elapsed = last_decay.elapsed().as_secs_f32();
                state.decay_over_time(decay_elapsed);
                last_decay = Instant::now();

                let idle_secs = last_interaction.elapsed().as_secs();
                if idle_secs >= 60 && idle_secs % 60 < 5 {
                    let note = format!("User has been idle for {} seconds.", idle_secs);
                    if !recent_events.iter().any(|e| e == &note) {
                        recent_events.push_back(note);
                    }
                }
            }

            _ = interval_timer.tick() => {
                // Periodic timer trigger
            }
        }

        let should_consult = pending_immediate_trigger || last_consultation.elapsed() >= config.interval;
        if should_consult {
            pending_immediate_trigger = false;
            last_consultation = Instant::now();

            let history: Vec<String> = recent_events.iter().cloned().collect();
            let decision = decide_thought(
                &config,
                &mut api_client,
                &state,
                &history,
            ).await;

            apply_decision_deltas(&mut state, &decision);
            let _ = decision_tx.send(decision);
        }
    }

    info!("Sparky Brain Tokio worker stopped.");
}

/// Execute intelligence decision on Tokio thread pool according to active engine.
async fn decide_thought(
    config: &BrainConfig,
    api_client: &mut Option<GemmaClient>,
    state: &EmotionalState,
    history: &[String],
) -> GemmaDecision {
    match config.engine_mode {
        BrainEngineMode::Local => {
            config.personality.heuristic_decision(state, history)
        }
        BrainEngineMode::Api => {
            if let Some(client) = api_client {
                let mut client_clone = client.clone();
                let state_clone = state.clone();
                let history_clone = history.to_vec();

                let api_res = tokio::task::spawn_blocking(move || {
                    client_clone.decide(&state_clone, &history_clone)
                }).await;

                match api_res {
                    Ok(decision) => decision,
                    Err(e) => {
                        warn!("Gemma API task failure: {}. Falling back to heuristics.", e);
                        config.personality.heuristic_decision(state, history)
                    }
                }
            } else {
                config.personality.heuristic_decision(state, history)
            }
        }
    }
}

async fn decide_chat(
    config: &BrainConfig,
    api_client: &mut Option<GemmaClient>,
    state: &EmotionalState,
    _history: &[String],
    user_msg: &str,
) -> GemmaDecision {
    match config.engine_mode {
        BrainEngineMode::Local => {
            config.personality.chat_reply(user_msg, state)
        }
        BrainEngineMode::Api => {
            if let Some(client) = api_client {
                let mut client_clone = client.clone();
                let state_clone = state.clone();
                let history_clone = _history.to_vec();
                let user_msg_clone = user_msg.to_string();

                let api_res = tokio::task::spawn_blocking(move || {
                    client_clone.decide_chat(&state_clone, &history_clone, &user_msg_clone)
                }).await;

                match api_res {
                    Ok(decision) => decision,
                    Err(e) => {
                        warn!("Gemma API chat failure: {}. Falling back to heuristics.", e);
                        config.personality.chat_reply(user_msg, state)
                    }
                }
            } else {
                config.personality.chat_reply(user_msg, state)
            }
        }
    }
}

fn apply_decision_deltas(state: &mut EmotionalState, decision: &GemmaDecision) {
    let ed = decision.energy_delta.unwrap_or(0.0);
    let md = decision.mood_delta.unwrap_or(0.0);
    let ad = decision.affection_delta.unwrap_or(0.0);
    let bd = decision.boredom_delta.unwrap_or(0.0);
    state.apply_delta(ed, md, ad, bd);
}

fn format_event(event: &BrainEvent) -> String {
    match event {
        BrainEvent::Started => "Sparky application started.".to_string(),
        BrainEvent::Clicked { x, y } => format!("User clicked Sparky at cursor ({:.0}, {:.0}).", x, y),
        BrainEvent::Dragged { dx, dy } => {
            format!("User dragged Sparky across desktop (dx: {:.0}, dy: {:.0}).", dx, dy)
        }
        BrainEvent::DragEnded => "User released Sparky after dragging.".to_string(),
        BrainEvent::AnimationCompleted { name } => {
            format!("Completed loop of '{}' animation.", name)
        }
        BrainEvent::Custom(msg) => msg.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_chat_immediate_response() {
        let config = BrainConfig {
            engine_mode: BrainEngineMode::Local,
            personality: PersonalityArchetype::ChaosGremlin,
            interval: Duration::from_secs(60),
            api_key: None,
            api_base: None,
            api_model: "gemma-3-270m".to_string(),
            local_model_dir: None,
        };
        let brain = BrainHandle::spawn(config);

        // Receive initial greeting
        let mut got_greeting = false;
        for _ in 0..50 {
            if brain.try_recv_decision().is_some() {
                got_greeting = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(got_greeting, "Brain should send initial greeting immediately");

        // Send a chat message
        brain.send_chat("Hello Sparky!");

        // Receive chat response
        let mut got_reply = None;
        for _ in 0..50 {
            if let Some(dec) = brain.try_recv_decision() {
                got_reply = Some(dec);
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(got_reply.is_some(), "Brain should respond to chat promptly");
        let reply = got_reply.unwrap();
        assert!(!reply.dialogue.is_empty());
        assert!(!reply.target_emotion.is_empty());
    }
}
