pub mod candle_engine;
pub mod gguf_engine;
pub mod prompt;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::state::emotion::EmotionRegistry;

/// Selection for the local cognition engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum BrainBackend {
    /// Automatically detect backend based on model path (prefers GGUF if present)
    #[default]
    Auto,
    /// GGUF quantized model inference via llama.cpp (low RAM footprint)
    Gguf,
    /// Candle transformer forward pass (full-precision safetensors)
    Candle,
}

impl std::fmt::Display for BrainBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Gguf => write!(f, "gguf"),
            Self::Candle => write!(f, "candle"),
        }
    }
}

#[allow(unused_imports)]
pub use gguf_engine::{BrainRequest, BrainResponse};

pub enum BrainEngine {
    Gguf(gguf_engine::BrainEngine),
    Candle(candle_engine::BrainEngine),
}

impl BrainEngine {
    #[allow(dead_code)]
    pub fn spawn(model_path: PathBuf, registry: Arc<EmotionRegistry>) -> Self {
        Self::spawn_with_backend(model_path, BrainBackend::Auto, registry)
    }

    pub fn spawn_with_backend(
        model_path: PathBuf,
        backend: BrainBackend,
        registry: Arc<EmotionRegistry>,
    ) -> Self {
        let selected = match backend {
            BrainBackend::Gguf => BrainBackend::Gguf,
            BrainBackend::Candle => BrainBackend::Candle,
            BrainBackend::Auto => Self::detect_backend(&model_path),
        };

        match selected {
            BrainBackend::Gguf => {
                log::info!("Cognition backend: GGUF Quantized Inference (llama.cpp)");
                BrainEngine::Gguf(gguf_engine::BrainEngine::spawn(model_path, registry))
            }
            BrainBackend::Candle => {
                log::info!("Cognition backend: Candle Safetensors Engine (full precision)");
                BrainEngine::Candle(candle_engine::BrainEngine::spawn(model_path, registry))
            }
            BrainBackend::Auto => unreachable!(),
        }
    }

    pub fn detect_backend(model_path: &Path) -> BrainBackend {
        if model_path.is_file() {
            if let Some(ext) = model_path.extension().and_then(|s| s.to_str())
                && ext.eq_ignore_ascii_case("gguf")
            {
                return BrainBackend::Gguf;
            }
        } else if model_path.is_dir() {
            // Check for .gguf inside directory first (preferred for low RAM footprint)
            if let Ok(entries) = std::fs::read_dir(model_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension().and_then(|s| s.to_str())
                        && ext.eq_ignore_ascii_case("gguf")
                    {
                        return BrainBackend::Gguf;
                    }
                }
            }
            // Check for safetensors inside directory
            let safetensors = model_path.join("model.safetensors");
            if safetensors.exists() {
                return BrainBackend::Candle;
            }
        }
        // Default to GGUF if neither file exists yet (GGUF is primary target with low RAM footprint)
        BrainBackend::Gguf
    }

    pub fn send_chat(&self, prompt: &str) {
        match self {
            Self::Gguf(engine) => engine.send_chat(prompt),
            Self::Candle(engine) => engine.send_chat(prompt),
        }
    }

    #[allow(dead_code)]
    pub fn interrupt(&self) {
        match self {
            Self::Gguf(engine) => engine.interrupt(),
            Self::Candle(engine) => engine.interrupt(),
        }
    }

    pub fn shutdown(&self) {
        match self {
            Self::Gguf(engine) => engine.shutdown(),
            Self::Candle(engine) => engine.shutdown(),
        }
    }

    pub fn try_recv_response(&self) -> Option<BrainResponse> {
        match self {
            Self::Gguf(engine) => engine.try_recv_response(),
            Self::Candle(engine) => match engine.try_recv_response() {
                Some(candle_engine::BrainResponse::Token(t)) => Some(BrainResponse::Token(t)),
                Some(candle_engine::BrainResponse::Complete {
                    full_text,
                    action_tag,
                }) => Some(BrainResponse::Complete {
                    full_text,
                    action_tag,
                }),
                Some(candle_engine::BrainResponse::Error(e)) => Some(BrainResponse::Error(e)),
                None => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_brain_backend_display_and_detection() {
        assert_eq!(BrainBackend::Auto.to_string(), "auto");
        assert_eq!(BrainBackend::Gguf.to_string(), "gguf");
        assert_eq!(BrainBackend::Candle.to_string(), "candle");

        // GGUF file extension detection
        let gguf_path = PathBuf::from("models/test_model.gguf");
        assert_eq!(BrainEngine::detect_backend(&gguf_path), BrainBackend::Gguf);

        // Fallback detection for non-existent path defaults to GGUF
        let empty_path = PathBuf::from("non_existent_dir_or_model");
        assert_eq!(BrainEngine::detect_backend(&empty_path), BrainBackend::Gguf);
    }

    #[test]
    fn test_gguf_brain_fallback_streaming_and_actions() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions.toml"),
        );

        let brain = BrainEngine::spawn_with_backend(
            PathBuf::from("non_existent_model.gguf"),
            BrainBackend::Gguf,
            registry,
        );

        brain.send_chat("here is a fish treat for you");

        let mut received_tokens = Vec::new();
        let mut completed_action = None;
        let start = std::time::Instant::now();

        while start.elapsed() < Duration::from_secs(5) {
            if let Some(resp) = brain.try_recv_response() {
                match resp {
                    BrainResponse::Token(token) => received_tokens.push(token),
                    BrainResponse::Complete { action_tag, .. } => {
                        completed_action = action_tag;
                        break;
                    }
                    BrainResponse::Error(_) => {}
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(
            !received_tokens.is_empty(),
            "Expected streaming tokens from GGUF fallback"
        );
        assert_eq!(completed_action, Some("happy".to_string()));
        brain.shutdown();
    }

    #[test]
    fn test_candle_brain_fallback_streaming_and_actions() {
        let registry = Arc::new(
            EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
                .expect("Failed to load emotions.toml"),
        );

        let brain = BrainEngine::spawn_with_backend(
            PathBuf::from("non_existent_candle_dir"),
            BrainBackend::Candle,
            registry,
        );

        brain.send_chat("bad dog stop that");

        let mut received_tokens = Vec::new();
        let mut completed_action = None;
        let start = std::time::Instant::now();

        while start.elapsed() < Duration::from_secs(5) {
            if let Some(resp) = brain.try_recv_response() {
                match resp {
                    BrainResponse::Token(token) => received_tokens.push(token),
                    BrainResponse::Complete { action_tag, .. } => {
                        completed_action = action_tag;
                        break;
                    }
                    BrainResponse::Error(_) => {}
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(
            !received_tokens.is_empty(),
            "Expected streaming tokens from Candle fallback"
        );
        assert_eq!(completed_action, Some("sad".to_string()));
        brain.shutdown();
    }
}
