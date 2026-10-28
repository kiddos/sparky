use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use log::{debug, info};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma3::{Config, Model};
use tokenizers::Tokenizer;

use super::decision::GemmaDecision;
use super::personality::PersonalityArchetype;
use super::state::EmotionalState;

/// Gemma 3 270M local inference engine.
pub struct LocalGemmaEngine {
    #[allow(dead_code)]
    inner: Arc<Mutex<GemmaInferenceInner>>,
    #[allow(dead_code)]
    pub model_name: String,
    #[allow(dead_code)]
    pub model_path: Option<PathBuf>,
}

#[allow(dead_code)]
struct GemmaInferenceInner {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    temperature: f64,
    top_p: f64,
    seed: u64,
}

#[allow(dead_code)]
impl LocalGemmaEngine {
    /// Attempts to load Gemma 3 270M from a specified directory or default search paths,
    /// or fetches from Hugging Face Hub if token is available.
    pub async fn load_or_fetch(
        model_dir: Option<&Path>,
        personality: &PersonalityArchetype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!("Initializing local Gemma 3 270M inference engine (pure Rust / Candle)...");

        // 1. Try local paths first
        match find_model_dir(model_dir) {
            Ok(resolved_dir) => {
                info!("Loading Gemma 3 270M weights from local directory: {}", resolved_dir.display());
                return Self::load_from_dir(&resolved_dir, personality);
            }
            Err(e) => {
                debug!("Local model directory search: {}", e);
            }
        }

        // 2. Try Hugging Face Hub async download if HF_TOKEN is provided or default repo
        info!("Checking Hugging Face Hub for Gemma 3 270M (google/gemma-3-270m-it)...");
        match fetch_from_hf_hub().await {
            Ok((config_path, tokenizer_path, weights_path)) => {
                info!("Successfully retrieved Gemma 3 270M from Hugging Face cache/hub!");
                return Self::load_from_files(&config_path, &tokenizer_path, &weights_path, personality);
            }
            Err(e) => {
                debug!("Hugging Face Hub fetch skipped or failed: {}", e);
            }
        }

        Err(
            "Gemma 3 270M model files not found locally. To run Gemma 3 270M locally: place config.json, model.safetensors, and tokenizer.json in './models/gemma-3-270m/' or pass --model-path <PATH> (or set HF_TOKEN to auto-download from Hugging Face).".into()
        )
    }

    /// Load directly from a local directory containing config.json, tokenizer.json, and model.safetensors.
    pub fn load_from_dir(
        dir: &Path,
        personality: &PersonalityArchetype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_path = dir.join("config.json");
        let tokenizer_path = if dir.join("tokenizer.json").exists() {
            dir.join("tokenizer.json")
        } else {
            dir.join("tokenizer.model")
        };

        let weights_path = if dir.join("model.safetensors").exists() {
            dir.join("model.safetensors")
        } else {
            find_safetensors(dir)
                .ok_or_else(|| format!("No .safetensors weights found in {}", dir.display()))?
        };

        Self::load_from_files(&config_path, &tokenizer_path, &weights_path, personality)
    }

    /// Load from explicit file paths.
    pub fn load_from_files(
        config_path: &Path,
        tokenizer_path: &Path,
        weights_path: &Path,
        personality: &PersonalityArchetype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_content = std::fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;
        let mut val: serde_json::Value = serde_json::from_str(&config_content)?;

        // Compatibility fix for Hugging Face Gemma 3 config variations
        if let Some(pattern) = val.get("_sliding_window_pattern").cloned() {
            val["sliding_window_pattern"] = pattern;
        }
        if val.get("sliding_window_pattern").is_none() {
            val["sliding_window_pattern"] = serde_json::json!(6);
        }
        if val.get("sliding_window").is_none() {
            val["sliding_window"] = serde_json::json!(512);
        }
        if val.get("query_pre_attn_scalar").is_none() {
            val["query_pre_attn_scalar"] = serde_json::json!(256);
        }
        if val.get("rope_local_base_freq").is_none() {
            val["rope_local_base_freq"] = serde_json::json!(10000.0);
        }
        if val.get("max_position_embeddings").is_none() {
            val["max_position_embeddings"] = serde_json::json!(32768);
        }

        let config: Config = serde_json::from_value(val)
            .map_err(|e| format!("Failed to parse Gemma 3 config: {}", e))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e))?;

        let device = Device::Cpu;
        let dtype = DType::F32;

        info!(
            "Memory-mapping Gemma 3 270M weights from {} (dtype={:?}, device={:?})...",
            weights_path.display(),
            dtype,
            device
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
                .map_err(|e| format!("Failed to mmap safetensors {}: {}", weights_path.display(), e))?
        };

        let model = Model::new(false, &config, vb)
            .map_err(|e| format!("Failed to build Gemma 3 model: {}", e))?;

        info!(
            "Gemma 3 270M initialized successfully! (layers={}, heads={}, hidden_size={}, personality='{}')",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size,
            personality.name()
        );

        let parent_dir = config_path.parent().map(|p| p.to_path_buf());

        Ok(Self {
            inner: Arc::new(Mutex::new(GemmaInferenceInner {
                model,
                tokenizer,
                device,
                temperature: 0.7,
                top_p: 0.9,
                seed: 42,
            })),
            model_name: "gemma-3-270m".to_string(),
            model_path: parent_dir,
        })
    }

    /// Run inference synchronously (meant to be executed on tokio::task::spawn_blocking).
    #[allow(dead_code)]
    pub fn infer_decision(
        &self,
        personality: &PersonalityArchetype,
        state: &EmotionalState,
        recent_events: &[String],
    ) -> Result<GemmaDecision, Box<dyn std::error::Error + Send + Sync>> {
        let mut inner = self.inner.lock().unwrap();
        let prompt = format_gemma3_prompt(personality, state, recent_events);

        let encoding = inner.tokenizer.encode(prompt, true)
            .map_err(|e| format!("Tokenization error: {}", e))?;
        let prompt_tokens = encoding.get_ids();

        if prompt_tokens.is_empty() {
            return Err("Tokenized prompt is empty".into());
        }

        let mut pos = 0;
        let mut logits_processor = LogitsProcessor::new(
            inner.seed,
            Some(inner.temperature),
            Some(inner.top_p),
        );
        inner.seed = inner.seed.wrapping_add(1);

        // Feed prompt tokens in one forward pass
        let input_tensor = Tensor::new(prompt_tokens, &inner.device)?.unsqueeze(0)?;
        let mut logits = inner.model.forward(&input_tensor, pos)?;
        pos += prompt_tokens.len();

        let mut generated_tokens = Vec::with_capacity(128);
        let max_tokens = 120;

        for _ in 0..max_tokens {
            let logits_squeezed = logits.squeeze(0)?.squeeze(0)?;
            let next_token = logits_processor.sample(&logits_squeezed)?;

            // Gemma special tokens: 1 = <eos>, 107 = <end_of_turn>, 2 = <bos>
            if next_token == 1 || next_token == 107 {
                break;
            }

            generated_tokens.push(next_token);

            // Early check: if we see closing brace '}', test if valid JSON is ready
            if let Ok(partial_str) = inner.tokenizer.decode(&generated_tokens, false) {
                if partial_str.contains('}') && partial_str.trim().ends_with('}') {
                    if let Ok(decision) = crate::brain::client::parse_json_decision(&partial_str) {
                        inner.model.clear_kv_cache();
                        return Ok(decision);
                    }
                }
            }

            let next_input = Tensor::new(&[next_token], &inner.device)?.unsqueeze(0)?;
            logits = inner.model.forward(&next_input, pos)?;
            pos += 1;
        }

        inner.model.clear_kv_cache();
        let decoded = inner.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| format!("Tokenizer decode error: {}", e))?;

        debug!("Gemma 3 270M raw generated text: {}", decoded);
        crate::brain::client::parse_json_decision(&decoded)
    }
}

/// Formats the prompt using Gemma 3 instruction tokens.
pub fn format_gemma3_prompt(
    personality: &PersonalityArchetype,
    state: &EmotionalState,
    recent_events: &[String],
) -> String {
    let system_instructions = personality.system_prompt();
    let events_str = if recent_events.is_empty() {
        "No major user interaction recently (Sparky is idling).".to_string()
    } else {
        recent_events
            .iter()
            .rev()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n- ")
    };

    let user_content = format!(
        "Current Internal State: {}\nDominant Vibe: {}\n\nRecent User Events:\n- {}\n\nDecide Sparky's next target_emotion, dialogue, and vector deltas as JSON:",
        state.prompt_summary(),
        state.dominant_vibe(),
        events_str
    );

    format!(
        "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
        system_instructions,
        user_content
    )
}

/// Search for Gemma 3 model directory in configured and standard paths.
pub fn find_model_dir(custom_path: Option<&Path>) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(p) = custom_path {
        if p.exists() {
            if p.is_dir() {
                return Ok(p.to_path_buf());
            } else if p.is_file() {
                if let Some(parent) = p.parent() {
                    return Ok(parent.to_path_buf());
                }
            }
        }
        return Err(format!("Specified model path does not exist: {}", p.display()).into());
    }

    // Check environment variable
    if let Ok(env_path) = std::env::var("GEMMA_MODEL_PATH")
        .or_else(|_| std::env::var("GEMMA_LOCAL_MODEL_PATH"))
        .or_else(|_| std::env::var("SPARKY_MODEL_PATH"))
    {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Ok(p);
        }
    }

    // Check project relative directories
    let candidates = [
        "models/gemma-3-270m",
        "models/gemma-3-270m-it",
        "models/gemma3-270m",
        "models/gemma-270m",
        "models",
    ];

    for candidate in candidates {
        let p = PathBuf::from(candidate);
        if p.exists() && (p.join("model.safetensors").exists() || p.join("config.json").exists()) {
            return Ok(p);
        }
    }

    // Check user cache dir ~/.cache/sparky/gemma-3-270m
    if let Ok(home) = std::env::var("HOME") {
        let cache_p = PathBuf::from(home).join(".cache/sparky/gemma-3-270m");
        if cache_p.exists() && cache_p.join("config.json").exists() {
            return Ok(cache_p);
        }
    }

    Err(
        "Gemma 3 270M model files not found in standard paths ('models/gemma-3-270m/', '~/.cache/sparky/gemma-3-270m/').".into()
    )
}

#[allow(dead_code)]
fn find_safetensors(dir: &Path) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                return Some(path);
            }
        }
    }
    None
}

/// Download or retrieve cached Gemma 3 270M files from Hugging Face Hub.
#[allow(dead_code)]
async fn fetch_from_hf_hub() -> Result<(PathBuf, PathBuf, PathBuf), Box<dyn std::error::Error + Send + Sync>> {
    let mut builder = hf_hub::api::tokio::ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        builder = builder.with_token(Some(token));
    }

    let api = builder.build()?;
    let repo_id = std::env::var("GEMMA_HF_REPO")
        .unwrap_or_else(|_| "google/gemma-3-270m-it".to_string());
    let repo = api.model(repo_id);

    let config_path = repo.get("config.json").await?;
    let tokenizer_path = repo.get("tokenizer.json").await?;
    let weights_path = repo.get("model.safetensors").await?;

    Ok((config_path, tokenizer_path, weights_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_gemma3_prompt() {
        let personality = PersonalityArchetype::ChaosGremlin;
        let state = EmotionalState::default();
        let events = vec!["User clicked Sparky".to_string()];
        let prompt = format_gemma3_prompt(&personality, &state, &events);

        assert!(prompt.starts_with("<start_of_turn>user\n"));
        assert!(prompt.ends_with("<end_of_turn>\n<start_of_turn>model\n"));
        assert!(prompt.contains("ARCHETYPE: The Chaos Gremlin"));
        assert!(prompt.contains("User clicked Sparky"));
    }

    #[test]
    fn test_find_model_dir_nonexistent() {
        let custom = PathBuf::from("/nonexistent/path/to/gemma");
        let res = find_model_dir(Some(&custom));
        assert!(res.is_err());
    }

    #[test]
    fn test_personality_heuristic_decision() {
        let personality = PersonalityArchetype::ZenMonk;
        let mut state = EmotionalState::default();
        state.boredom = 90.0;
        let decision = personality.heuristic_decision(&state, &[]);
        assert_eq!(decision.target_emotion, "sleepy");
        assert!(!decision.dialogue.is_empty());
    }
}
