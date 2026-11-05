use crossbeam_channel::{Receiver, Sender};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::brain::prompt::{extract_action_tag, format_chat_prompt};
use crate::state::emotion::EmotionRegistry;

pub enum BrainRequest {
    Chat {
        prompt: String,
        max_tokens: usize,
    },
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
    pub fn spawn(model_path: PathBuf, registry: Arc<EmotionRegistry>) -> Self {
        log::info!(
            "Spawning cognitive worker thread 'sparky-brain' (GGUF model path: {:?})...",
            model_path
        );
        let (req_tx, req_rx) = crossbeam_channel::unbounded::<BrainRequest>();
        let (res_tx, res_rx) = crossbeam_channel::unbounded::<BrainResponse>();

        let reg_clone = registry.clone();
        std::thread::Builder::new()
            .name("sparky-brain".to_string())
            .spawn(move || {
                log::info!("Sparky GGUF brain worker ready (lazy loading enabled from {:?})", model_path);
                let mut loaded_brain: Option<LoadedBrain> = None;
                let mut load_attempted = false;

                while let Ok(request) = req_rx.recv() {
                    match request {
                        BrainRequest::Chat { prompt, max_tokens } => {
                            if !load_attempted {
                                loaded_brain = Self::try_load_model(&model_path);
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
                log::info!("Sparky GGUF brain thread terminated.");
            })
            .expect("Failed to spawn brain thread");

        Self {
            request_tx: req_tx,
            response_rx: res_rx,
        }
    }

    fn try_load_model(model_path: &Path) -> Option<LoadedBrain> {
        LoadedBrain::load(model_path)
    }

    fn generate_fallback(prompt: &str, registry: &EmotionRegistry, sender: &Sender<BrainResponse>) {
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
        } else if p.contains("tired")
            || p.contains("yawn")
            || p.contains("exhausted")
            || p.contains("drowsy")
        {
            (
                "*yawns widely and rubs eyes with a tiny paw* So tired... my eyelids feel so heavy...",
                "tired",
            )
        } else if p.contains("sleep")
            || p.contains("night")
            || p.contains("bed")
            || p.contains("nap")
            || p.contains("rest")
            || p.contains("shh")
        {
            (
                "*curls up into a tight ball* Zzz... so sleepy... wake me up at feeding time...",
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
    backend: llama_cpp_2::llama_backend::LlamaBackend,
    model: llama_cpp_2::model::LlamaModel,
}

impl LoadedBrain {
    fn load(model_path: &Path) -> Option<Self> {
        let target_path = if model_path.is_dir() {
            // Check for *.gguf files in directory
            let mut gguf_path = None;
            if let Ok(entries) = std::fs::read_dir(model_path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
                        gguf_path = Some(p);
                        break;
                    }
                }
            }
            gguf_path?
        } else if model_path.exists() {
            model_path.to_path_buf()
        } else {
            log::info!(
                "GGUF model file not found at {:?}. Running with embedded cognitive fallback.",
                model_path
            );
            return None;
        };

        log::info!("Initializing llama.cpp backend...");
        let backend = match llama_cpp_2::llama_backend::LlamaBackend::init() {
            Ok(b) => b,
            Err(e) => {
                log::warn!("Failed to initialize llama.cpp backend: {}", e);
                return None;
            }
        };

        log::info!("Loading Gemma 3 GGUF model from {:?}...", target_path);
        let mut model_params = llama_cpp_2::model::params::LlamaModelParams::default();
        model_params = model_params.with_use_mmap(true);

        let model = match llama_cpp_2::model::LlamaModel::load_from_file(
            &backend,
            &target_path,
            &model_params,
        ) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Failed to load GGUF model: {}", e);
                return None;
            }
        };

        log::info!("Gemma 3 GGUF cognitive model loaded successfully!");
        Some(Self { backend, model })
    }

    fn generate_reply(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        registry: &EmotionRegistry,
        sender: &Sender<BrainResponse>,
    ) -> anyhow::Result<()> {
        let n_ctx = NonZeroU32::new(1024).expect("1024 is non-zero");
        let ctx_params = llama_cpp_2::context::params::LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_threads(2)
            .with_n_threads_batch(2);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create llama context: {:?}", e))?;

        let vocab = self.model.vocab();
        let formatted = format_chat_prompt(registry, &[], prompt);
        let tokens = vocab.tokenize(formatted.as_bytes(), true, true);
        if tokens.is_empty() {
            anyhow::bail!("Empty prompt token sequence");
        }

        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(1024, 1);
        let last_idx = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i == last_idx)?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Failed to decode initial prompt batch: {:?}", e))?;

        let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
            llama_cpp_2::sampling::LlamaSampler::temp(0.7),
            llama_cpp_2::sampling::LlamaSampler::min_p(0.05, 1),
            llama_cpp_2::sampling::LlamaSampler::dist(1337),
        ]);

        let mut n_cur = tokens.len() as i32;
        let mut generated_text = String::new();

        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..max_tokens {
            let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(next_token);

            if vocab.is_eog(next_token) {
                break;
            }

            let piece_bytes = vocab.token_to_piece(next_token, false, None);
            let piece = String::from_utf8_lossy(&piece_bytes).to_string();

            generated_text.push_str(&piece);
            let _ = sender.send(BrainResponse::Token(piece));

            batch.clear();
            batch.add(next_token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Failed to decode token: {:?}", e))?;
        }

        let (clean_text, action_tag) = extract_action_tag(&generated_text, registry);
        let _ = sender.send(BrainResponse::Complete {
            full_text: clean_text,
            action_tag,
        });

        Ok(())
    }
}
