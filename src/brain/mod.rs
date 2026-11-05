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

/// Default GGUF filename for Gemma 3 270M Q4_K_M
pub const GEMMA3_GGUF_FILENAME: &str = "gemma-3-270m-it-q4_k_m.gguf";

/// Alternative upstream Hugging Face filename (uppercase Q4_K_M)
pub const GEMMA3_GGUF_UPSTREAM_FILENAME: &str = "gemma-3-270m-it-Q4_K_M.gguf";

/// Hugging Face repository containing Gemma 3 GGUF models
pub const GEMMA3_GGUF_REPO: &str = "google/gemma-3-270m-it-GGUF";

/// Direct Hugging Face download URL for Gemma 3 270M Q4_K_M GGUF
pub const GEMMA3_GGUF_DIRECT_URL: &str =
    "https://huggingface.co/google/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf";

/// Finds the path to the GGUF model file if it exists on disk.
pub fn find_gguf_model(model_path: &Path) -> Option<PathBuf> {
    if model_path.is_file() {
        if let Some(ext) = model_path.extension().and_then(|s| s.to_str())
            && ext.eq_ignore_ascii_case("gguf")
            && model_path.exists()
        {
            return Some(model_path.to_path_buf());
        }
    } else if model_path.is_dir() {
        // Look for any .gguf file in the directory
        if let Ok(entries) = std::fs::read_dir(model_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(ext) = p.extension().and_then(|s| s.to_str())
                    && ext.eq_ignore_ascii_case("gguf")
                    && p.is_file()
                {
                    return Some(p);
                }
            }
        }
        // Also check standard filenames directly
        let candidate1 = model_path.join(GEMMA3_GGUF_FILENAME);
        if candidate1.is_file() {
            return Some(candidate1);
        }
        let candidate2 = model_path.join(GEMMA3_GGUF_UPSTREAM_FILENAME);
        if candidate2.is_file() {
            return Some(candidate2);
        }
    }

    // Check standard default locations
    let default_path1 = Path::new("models/gemma-3-270m").join(GEMMA3_GGUF_FILENAME);
    if default_path1.is_file() {
        return Some(default_path1);
    }
    let default_path2 = Path::new("models/gemma-3-270m").join(GEMMA3_GGUF_UPSTREAM_FILENAME);
    if default_path2.is_file() {
        return Some(default_path2);
    }

    None
}

/// Checks if the GGUF model has already been downloaded.
pub fn is_gguf_downloaded(model_path: &Path) -> bool {
    find_gguf_model(model_path).is_some()
}

/// Resolves the file destination path where the GGUF model should be saved.
pub fn resolve_gguf_download_path(model_path: &Path) -> PathBuf {
    if let Some(ext) = model_path.extension().and_then(|s| s.to_str())
        && ext.eq_ignore_ascii_case("gguf")
    {
        return model_path.to_path_buf();
    }
    model_path.join(GEMMA3_GGUF_FILENAME)
}

/// Downloads the Gemma 3 270M GGUF model to the specified destination path.
pub fn download_gguf_model(dest_path: &Path) -> anyhow::Result<PathBuf> {
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // 1. Try existing scripts/download_gemma3_gguf.sh if present
    let script = Path::new("scripts/download_gemma3_gguf.sh");
    if script.is_file() {
        log::info!(
            "Attempting GGUF download via {:?} -> {:?}",
            script,
            dest_path
        );
        let status = std::process::Command::new("bash")
            .arg(script)
            .arg(dest_path)
            .status();
        if let Ok(st) = status
            && st.success()
            && dest_path.is_file()
        {
            log::info!("GGUF model downloaded successfully via download script.");
            return Ok(dest_path.to_path_buf());
        }
    }

    // 2. Try huggingface-cli
    if let Some(parent) = dest_path.parent() {
        log::info!("Attempting GGUF download via huggingface-cli...");
        let status = std::process::Command::new("huggingface-cli")
            .arg("download")
            .arg(GEMMA3_GGUF_REPO)
            .arg(GEMMA3_GGUF_UPSTREAM_FILENAME)
            .arg("--local-dir")
            .arg(parent)
            .status();
        if let Ok(st) = status
            && st.success()
        {
            let downloaded = parent.join(GEMMA3_GGUF_UPSTREAM_FILENAME);
            if downloaded.is_file() && downloaded != dest_path {
                let _ = std::fs::rename(&downloaded, dest_path);
            }
            if dest_path.is_file() {
                return Ok(dest_path.to_path_buf());
            }
        }
    }

    // 3. Try curl
    log::info!("Attempting GGUF download via curl to {:?}...", dest_path);
    let status = std::process::Command::new("curl")
        .arg("-f")
        .arg("-L")
        .arg(GEMMA3_GGUF_DIRECT_URL)
        .arg("-o")
        .arg(dest_path)
        .status();
    if let Ok(st) = status
        && st.success()
        && dest_path.is_file()
    {
        log::info!("GGUF model downloaded successfully via curl.");
        return Ok(dest_path.to_path_buf());
    }

    // 4. Try wget
    log::info!("Attempting GGUF download via wget to {:?}...", dest_path);
    let status = std::process::Command::new("wget")
        .arg("-O")
        .arg(dest_path)
        .arg(GEMMA3_GGUF_DIRECT_URL)
        .status();
    if let Ok(st) = status
        && st.success()
        && dest_path.is_file()
    {
        log::info!("GGUF model downloaded successfully via wget.");
        return Ok(dest_path.to_path_buf());
    }

    anyhow::bail!(
        "Failed to download GGUF model to {}. Please download manually from {}",
        dest_path.display(),
        GEMMA3_GGUF_DIRECT_URL
    )
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

    #[test]
    fn test_gguf_detection_and_path_resolution() {
        let temp_dir =
            std::env::temp_dir().join(format!("sparky_test_gguf_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // 1. Empty directory -> not downloaded
        assert!(!is_gguf_downloaded(&temp_dir));
        assert_eq!(find_gguf_model(&temp_dir), None);

        // 2. Resolve download path
        let resolved = resolve_gguf_download_path(&temp_dir);
        assert_eq!(resolved, temp_dir.join(GEMMA3_GGUF_FILENAME));

        let direct_path = temp_dir.join("my_model.gguf");
        assert_eq!(resolve_gguf_download_path(&direct_path), direct_path);

        // 3. Create dummy .gguf file in temp_dir
        std::fs::write(&resolved, b"dummy gguf content").unwrap();
        assert!(is_gguf_downloaded(&temp_dir));
        assert_eq!(find_gguf_model(&temp_dir), Some(resolved.clone()));

        // 4. Test direct file path
        assert!(is_gguf_downloaded(&resolved));
        assert_eq!(find_gguf_model(&resolved), Some(resolved));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
