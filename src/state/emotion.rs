use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum LoopMode {
    Loop,
    OnceThenReturn,
}

impl Default for LoopMode {
    fn default() -> Self {
        LoopMode::OnceThenReturn
    }
}

fn default_emoji() -> String {
    "🐾".to_string()
}
fn default_tint() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_return_to() -> String {
    "sit".to_string()
}
fn default_duration() -> f32 {
    8.0
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmotionDefinition {
    #[serde(default, skip_deserializing)]
    pub id: String,
    pub name: String,
    #[serde(default = "default_emoji")]
    pub emoji: String,
    pub asset: String,
    pub fallback: Option<String>,
    #[serde(default = "default_tint")]
    pub shader_tint: [f32; 3],
    #[serde(default)]
    pub loop_mode: LoopMode,
    #[serde(default = "default_return_to")]
    pub return_to: String,
    #[serde(default = "default_duration")]
    pub duration_secs: f32,
    pub prompt_description: String,
}

#[derive(Deserialize)]
struct EmotionConfigFile {
    emotions: HashMap<String, EmotionDefinition>,
}

#[derive(Debug, Clone)]
pub struct EmotionRegistry {
    emotions: HashMap<String, EmotionDefinition>,
    order: Vec<String>,
    available_assets: Vec<String>,
}

impl EmotionRegistry {
    pub fn load_from_toml<P: AsRef<Path>, Q: AsRef<Path>>(
        config_path: P,
        assets_dir: Q,
    ) -> anyhow::Result<Self> {
        let config_path_ref = config_path.as_ref();
        let assets_dir_ref = assets_dir.as_ref();
        log::info!(
            "Loading emotions configuration from '{}'...",
            config_path_ref.display()
        );
        let content = fs::read_to_string(config_path_ref)?;
        let parsed: EmotionConfigFile = toml::from_str(&content)?;

        let mut available_assets = Vec::new();
        if let Ok(entries) = fs::read_dir(assets_dir_ref) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    available_assets.push(name.to_string());
                }
            }
        }

        let mut emotions = HashMap::new();
        let mut base_order = vec![
            "sit".to_string(),
            "happy".to_string(),
            "naughty".to_string(),
            "angry".to_string(),
        ];
        let mut extra_order = Vec::new();

        for (id, mut def) in parsed.emotions {
            def.id = id.clone();
            if !base_order.contains(&id) {
                extra_order.push(id.clone());
            }
            emotions.insert(id, def);
        }

        extra_order.sort();
        base_order.retain(|id| emotions.contains_key(id));
        base_order.extend(extra_order);

        log::info!(
            "Loaded {} emotion definitions from '{}' (registered asset files found: {})",
            emotions.len(),
            config_path_ref.display(),
            available_assets.len()
        );

        Ok(Self {
            emotions,
            order: base_order,
            available_assets,
        })
    }

    pub fn get(&self, id: &str) -> Option<&EmotionDefinition> {
        self.emotions.get(id)
    }

    pub fn all_emotions(&self) -> impl Iterator<Item = &EmotionDefinition> {
        self.order.iter().filter_map(|id| self.emotions.get(id))
    }

    pub fn active_emotions(&self) -> impl Iterator<Item = &EmotionDefinition> {
        self.all_emotions()
    }

    pub fn available_emotion_ids(&self) -> Vec<String> {
        self.order.clone()
    }

    #[allow(dead_code)]
    pub fn has_asset(&self, asset_filename: &str) -> bool {
        self.available_assets.iter().any(|a| a == asset_filename)
    }

    /// Resolves the effective animation asset name and shader tint, applying fallback if needed
    pub fn resolve_visuals(&self, emotion_id: &str) -> (String, [f32; 3]) {
        if let Some(def) = self.get(emotion_id) {
            if self.available_assets.contains(&def.asset) {
                let clip_name = def.asset.trim_end_matches(".webp").to_string();
                return (clip_name, def.shader_tint);
            }

            // Asset missing, resolve fallback
            if let Some(fallback_id) = &def.fallback {
                if let Some(fallback_def) = self.get(fallback_id) {
                    if self.available_assets.contains(&fallback_def.asset) {
                        let fallback_clip =
                            fallback_def.asset.trim_end_matches(".webp").to_string();
                        return (fallback_clip, def.shader_tint);
                    }
                }
            }

            // Ultimate fallback to sit if available
            if self.available_assets.contains(&"sit.webp".to_string()) {
                return ("sit".to_string(), def.shader_tint);
            }
        }

        ("sit".to_string(), [1.0, 1.0, 1.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_emotions_and_fallbacks() {
        let registry = EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
            .expect("Failed to load emotions.toml");

        assert!(registry.get("sit").is_some());
        assert!(registry.get("happy").is_some());
        assert!(registry.get("naughty").is_some());
        assert!(registry.get("angry").is_some());
        assert!(registry.get("sad").is_some());
        assert!(registry.get("smirk").is_some());

        // Visual resolution for present assets
        let (clip, tint) = registry.resolve_visuals("happy");
        assert_eq!(clip, "happy");
        assert_eq!(tint, [1.0, 1.0, 1.0]);

        // Visual resolution for fallback asset (sad -> fallback to sit with cold tint)
        let (clip, tint) = registry.resolve_visuals("sad");
        assert_eq!(clip, "sit");
        assert_eq!(tint, [0.80, 0.85, 1.05]);

        // Visual resolution for smirk -> resolves to smirk if asset present, else fallback to naughty
        let (clip, tint) = registry.resolve_visuals("smirk");
        if registry.has_asset("smirk.webp") {
            assert_eq!(clip, "smirk");
        } else {
            assert_eq!(clip, "naughty");
        }
        assert_eq!(tint, [1.0, 1.0, 1.0]);
    }
}
