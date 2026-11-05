use clap::Parser;
use std::path::PathBuf;

/// Sparky: Ultra-low-resource hardware-accelerated desktop pet powered by Gemma 3 270M.
#[derive(Parser, Debug, Clone)]
#[command(name = "sparky", author, version, about, long_about = None)]
pub struct Cli {
    /// Run automated 10-second benchmark and performance profiling session
    #[arg(long)]
    pub profile_run: bool,

    /// Path to emotions TOML configuration file
    #[arg(long, default_value = "assets/emotions.toml")]
    pub emotions_config: PathBuf,

    /// Path to directory containing WebP animation clips
    #[arg(long, default_value = "assets/animations")]
    pub animations_dir: PathBuf,

    /// Path to local Gemma 3 model weights directory or GGUF file
    #[arg(long, default_value = "models/gemma-3-270m")]
    pub model_path: PathBuf,

    /// Cognitive engine backend for local inference (auto, gguf, candle)
    #[arg(long, value_enum, default_value = "auto")]
    pub brain_backend: crate::brain::BrainBackend,

    /// Target animation playback frame rate in FPS
    #[arg(long, default_value_t = 24.0)]
    pub fps: f64,

    /// Starting emotion for the pet (e.g. sit, happy, angry, naughty)
    #[arg(long)]
    pub initial_emotion: Option<String>,

    /// Start in mouse click-through mode
    #[arg(long)]
    pub click_through: bool,

    /// Bind address for the JSON-RPC 2.0 server (HTTP and raw TCP)
    #[arg(long, default_value = "127.0.0.1:6006")]
    pub rpc_addr: String,

    /// Disable the JSON-RPC server
    #[arg(long)]
    pub no_rpc: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_default_args() {
        let args = Cli::parse_from(["sparky"]);
        assert!(!args.profile_run);
        assert_eq!(args.emotions_config, PathBuf::from("assets/emotions.toml"));
        assert_eq!(args.animations_dir, PathBuf::from("assets/animations"));
        assert_eq!(args.model_path, PathBuf::from("models/gemma-3-270m"));
        assert_eq!(args.brain_backend, crate::brain::BrainBackend::Auto);
        assert!((args.fps - 24.0).abs() < f64::EPSILON);
        assert_eq!(args.initial_emotion, None);
        assert!(!args.click_through);
        assert_eq!(args.rpc_addr, "127.0.0.1:6006");
        assert!(!args.no_rpc);
    }

    #[test]
    fn test_cli_custom_flags() {
        let args = Cli::parse_from([
            "sparky",
            "--profile-run",
            "--fps",
            "30.0",
            "--emotions-config",
            "custom_emotions.toml",
            "--animations-dir",
            "custom_assets",
            "--model-path",
            "custom_models",
            "--initial-emotion",
            "happy",
            "--click-through",
            "--rpc-addr",
            "0.0.0.0:8080",
            "--brain-backend",
            "gguf",
            "--no-rpc",
        ]);
        assert!(args.profile_run);
        assert_eq!(args.emotions_config, PathBuf::from("custom_emotions.toml"));
        assert_eq!(args.animations_dir, PathBuf::from("custom_assets"));
        assert_eq!(args.model_path, PathBuf::from("custom_models"));
        assert_eq!(args.brain_backend, crate::brain::BrainBackend::Gguf);
        assert!((args.fps - 30.0).abs() < f64::EPSILON);
        assert_eq!(args.initial_emotion.as_deref(), Some("happy"));
        assert!(args.click_through);
        assert_eq!(args.rpc_addr, "0.0.0.0:8080");
        assert!(args.no_rpc);
    }
}
