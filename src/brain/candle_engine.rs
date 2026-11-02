use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use crossbeam_channel::{Receiver, Sender};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma3::{Config as Gemma3Config, Model as Gemma3Model};
use tokenizers::Tokenizer;

use crate::brain::prompt::{extract_action_tag, format_chat_prompt};
use crate::state::emotion::EmotionRegistry;

pub enum BrainRequest {
    Chat { prompt: String, max_tokens: usize },
    #[allow(dead_code)]
    Interrupt,
    Shutdown,
}

pub enum BrainResponse {
    Token(String),
    Complete {
        #[allow(dead_code)]
        full_text: String,
        action_tag: Option<String>,
    },
    Error(String),
}

pub struct BrainEngine {
    request_tx: Sender<BrainRequest>,
    response_rx: Receiver<BrainResponse>,
}

impl BrainEngine {
    pub fn spawn(model_dir: PathBuf, registry: Arc<EmotionRegistry>) -> Self {
        let (req_tx, req_rx) = crossbeam_channel::unbounded::<BrainRequest>();
        let (res_tx, res_rx) = crossbeam_channel::unbounded::<BrainResponse>();

        let reg_clone = registry.clone();
        std::thread::Builder::new()
            .name("sparky-brain".to_string())
            .spawn(move || {
                log::info!("Sparky brain worker ready (lazy loading enabled from {:?})", model_dir);
                let mut loaded_brain: Option<LoadedBrain> = None;
                let mut load_attempted = false;

                while let Ok(request) = req_rx.recv() {
                    match request {
                        BrainRequest::Chat { prompt, max_tokens } => {
                            if !load_attempted {
                                loaded_brain = Self::try_load_model(&model_dir);
                                load_attempted = true;
                            }

                            if let Some(ref mut brain) = loaded_brain {
                                if let Err(e) = brain.generate_reply(&prompt, max_tokens, &reg_clone, &res_tx) {
                                    log::warn!("Inference error: {}. Falling back to dynamic kitten brain.", e);
                                    let _ = res_tx.send(BrainResponse::Error(format!("Inference fallback: {}", e)));
                                    Self::generate_fallback(&prompt, &reg_clone, &res_tx);
                                }
                            } else {
                                Self::generate_fallback(&prompt, &reg_clone, &res_tx);
                            }
                        }
                        BrainRequest::Interrupt => {}
                        BrainRequest::Shutdown => break,
                    }
                }
                log::info!("Sparky brain thread terminated.");
            })
            .expect("Failed to spawn brain thread");

        Self {
            request_tx: req_tx,
            response_rx: res_rx,
        }
    }

    fn try_load_model(model_dir: &Path) -> Option<LoadedBrain> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        if !tokenizer_path.exists() || !config_path.exists() || !weights_path.exists() {
            log::info!(
                "Gemma 3 model files not found in {:?}. Running with embedded cognitive fallback.",
                model_dir
            );
            return None;
        }

        log::info!("Loading Gemma 3 tokenizer from {:?}", tokenizer_path);
        let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to load tokenizer: {}", e);
                return None;
            }
        };

        log::info!("Loading Gemma 3 config from {:?}", config_path);
        let config_str = match fs::read_to_string(&config_path) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Failed to read config: {}", e);
                return None;
            }
        };

        let mut config_val: serde_json::Value = match serde_json::from_str(&config_str) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("Failed to parse config json: {}", e);
                return None;
            }
        };

        if let Some(val) = config_val.get("_sliding_window_pattern") {
            if config_val.get("sliding_window_pattern").is_none() {
                config_val["sliding_window_pattern"] = val.clone();
            }
        }

        let config: Gemma3Config = match serde_json::from_value(config_val) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("Failed to deserialize Gemma 3 config: {}", e);
                return None;
            }
        };

        let device = Device::Cpu;
        log::info!("Loading Gemma 3 weights from {:?} onto CPU...", weights_path);
        let vb = match unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
        } {
            Ok(vb) => vb,
            Err(e) => {
                log::warn!("Failed to map safetensors: {}", e);
                return None;
            }
        };

        log::info!("Instantiating Gemma 3 model architecture...");
        let model = match Gemma3Model::new(false, &config, vb) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Failed to build Gemma 3 model: {}", e);
                return None;
            }
        };

        log::info!("Gemma 3 270M cognitive engine loaded successfully!");
        Some(LoadedBrain {
            tokenizer,
            model,
            device,
        })
    }

    fn generate_fallback(
        prompt: &str,
        registry: &EmotionRegistry,
        sender: &Sender<BrainResponse>,
    ) {
        let p = prompt.to_lowercase();
        let (reply, action) = if p.contains("fish")
            || p.contains("treat")
            || p.contains("food")
            || p.contains("good boy")
            || p.contains("good girl")
            || p.contains("pet")
            || p.contains("love")
            || p.contains("cute")
        {
            (
                "*purrs loudly and rubs against your cursor* Meow! That makes me so happy! Can I have another treat please?",
                "happy",
            )
        } else if p.contains("stop")
            || p.contains("bad")
            || p.contains("no")
            || p.contains("hate")
            || p.contains("scold")
            || p.contains("go away")
        {
            (
                "*droops ears and looks at you with big teary eyes* M-meow... Did I do something wrong? I'm sorry...",
                "sad",
            )
        } else if p.contains("angry")
            || p.contains("poke")
            || p.contains("hit")
            || p.contains("fight")
            || p.contains("dog")
        {
            (
                "*arches back and lets out a tiny defensive hiss* Hssss! Don't push your luck, two-legs!",
                "angry",
            )
        } else if p.contains("secret")
            || p.contains("clever")
            || p.contains("smart")
            || p.contains("joke")
            || p.contains("smirk")
        {
            (
                "*gives a knowing grin and winks* Heh, you think you're in charge here? Check your keyboard shortcuts again.",
                "smirk",
            )
        } else if p.contains("play")
            || p.contains("chase")
            || p.contains("jump")
            || p.contains("mouse")
            || p.contains("laser")
            || p.contains("naughty")
        {
            (
                "*wiggles hind legs and pounces at the screen* Meow haha! Catch me if you can!",
                "naughty",
            )
        } else if p.contains("why")
            || p.contains("what")
            || p.contains("how")
            || p.contains("who")
            || p.contains("tell me")
            || p.contains("?")
        {
            (
                "*tilts head curiously with perked ears* Mew? That sounds fascinating! Tell me more about it!",
                "curious",
            )
        } else if p.contains("sleep")
            || p.contains("tired")
            || p.contains("night")
            || p.contains("bed")
            || p.contains("rest")
        {
            (
                "*yawns widely and curls up into a tight ball* Zzz... so sleepy... wake me up at feeding time...",
                "sleep",
            )
        } else {
            (
                "*blinks serenely and swishes tail* Meow! I'm watching your code take shape. Everything is under control.",
                "sit",
            )
        };

        // Stream reply tokens
        let full_text_with_tag = format!("{} <action>{}</action>", reply, action);
        for word in reply.split_inclusive(' ') {
            std::thread::sleep(Duration::from_millis(40));
            let _ = sender.send(BrainResponse::Token(word.to_string()));
        }

        let (clean_text, action_tag) = extract_action_tag(&full_text_with_tag, registry);
        let _ = sender.send(BrainResponse::Complete {
            full_text: clean_text,
            action_tag,
        });
    }

    pub fn send_chat(&self, prompt: &str) {
        let _ = self.request_tx.send(BrainRequest::Chat {
            prompt: prompt.to_string(),
            max_tokens: 100,
        });
    }

    #[allow(dead_code)]
    pub fn interrupt(&self) {
        let _ = self.request_tx.send(BrainRequest::Interrupt);
    }

    pub fn shutdown(&self) {
        let _ = self.request_tx.send(BrainRequest::Shutdown);
    }

    pub fn try_recv_response(&self) -> Option<BrainResponse> {
        self.response_rx.try_recv().ok()
    }
}

struct LoadedBrain {
    tokenizer: Tokenizer,
    model: Gemma3Model,
    device: Device,
}

impl LoadedBrain {
    fn generate_reply(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        registry: &EmotionRegistry,
        sender: &Sender<BrainResponse>,
    ) -> anyhow::Result<()> {
        let formatted = format_chat_prompt(registry, &[], prompt);
        let tokens = self
            .tokenizer
            .encode(formatted, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {}", e))?;

        let prompt_token_ids = tokens.get_ids();
        if prompt_token_ids.is_empty() {
            anyhow::bail!("Empty prompt token sequence");
        }

        self.model.clear_kv_cache();

        let mut logits_processor = LogitsProcessor::new(1337, Some(0.7), Some(0.9));
        let mut all_tokens: Vec<u32> = prompt_token_ids.to_vec();
        let mut generated_text = String::new();

        let mut input_tensor =
            Tensor::new(prompt_token_ids, &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input_tensor, 0)?;

        let mut next_token = logits_processor.sample(&logits.squeeze(0)?.squeeze(0)?)?;
        all_tokens.push(next_token);

        let eos_id = self.tokenizer.token_to_id("<end_of_turn>").unwrap_or(1);

        for index in 0..max_tokens {
            if next_token == eos_id || next_token == 1 {
                break;
            }

            if let Ok(piece) = self.tokenizer.decode(&[next_token], false) {
                generated_text.push_str(&piece);
                let _ = sender.send(BrainResponse::Token(piece));
            }

            input_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input_tensor, prompt_token_ids.len() + index)?;
            next_token = logits_processor.sample(&logits.squeeze(0)?.squeeze(0)?)?;
            all_tokens.push(next_token);
        }

        let (clean_text, action_tag) = extract_action_tag(&generated_text, registry);
        let _ = sender.send(BrainResponse::Complete {
            full_text: clean_text,
            action_tag,
        });

        Ok(())
    }
}
