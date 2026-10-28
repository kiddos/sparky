use std::env;
use std::error::Error;
use std::path::PathBuf;

#[derive(Default, Debug, Clone)]
pub struct CliArgs {
    pub video_dir: Option<PathBuf>,
    pub scale: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub threshold: Option<f32>,
    pub fps: Option<f64>,
    pub personality: Option<String>,
    pub gemma_engine: Option<String>,
    pub gemma_model: Option<String>,
    pub gemma_api_base: Option<String>,
    pub model_path: Option<PathBuf>,
    pub brain_interval: Option<u64>,
    pub disable_brain: bool,
    pub chat: bool,
}

pub fn print_help() {
    println!("Sparky - Desktop Pet & Gemma AI Companion");
    println!("Usage: sparky [OPTIONS]");
    println!();
    println!("Visual & Playback Options:");
    println!(
        "  -v, --video-dir <PATH>          Directory with video files (default: auto-detects videos/v2, videos)"
    );
    println!(
        "  -s, --scale <FLOAT>             Display scale factor (e.g. 1.0 for 1280x720, 0.5 for 640x360)"
    );
    println!("  -w, --width <INT>               Window width in pixels");
    println!("  -h, --height <INT>              Window height in pixels");
    println!(
        "  -t, --threshold <FLOAT>         Black color threshold to ignore [0.0 - 1.0] (default: 0.08)"
    );
    println!(
        "      --fps <FLOAT>               Frame rate for animation (default: auto-detected or 24.0)"
    );
    println!("  -c, --chat                      Open chat window on startup");
    println!();
    println!("Gemma AI Intelligence Options:");
    println!(
        "  -p, --personality <ARCHETYPE>   Sparky personality (chaos_gremlin, hype_pet, zen_monk, tsundere)"
    );
    println!(
        "      --engine <local|api>        Intelligence engine backend: 'local' (Gemma 3 270M) or 'api' (Gemma 4 API)"
    );
    println!(
        "      --local, --gemma-3-270m     Use local Gemma 3 270M in-process neural engine (no API key needed)"
    );
    println!(
        "      --model-path <PATH>         Path to local Gemma 3 270M directory or safetensors weights"
    );
    println!(
        "      --gemma-model <NAME>        Gemma model name (e.g. gemma-3-270m, gemma-4, or $GEMMA_MODEL)"
    );
    println!(
        "      --gemma-api-base <URL>      Gemma API base endpoint URL (default: Google AI Studio)"
    );
    println!(
        "      --brain-interval <SECS>     Interval between periodic AI checks in seconds (default: 20)"
    );
    println!(
        "      --no-brain                  Disable Gemma AI brain and use random playback"
    );
    println!("      --help                      Print this help message");
}

pub fn parse_args() -> Result<CliArgs, Box<dyn Error>> {
    parse_args_from(env::args().skip(1))
}

pub fn parse_args_from<I, T>(args: I) -> Result<CliArgs, Box<dyn Error>>
where
    I: IntoIterator<Item = T>,
    T: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let mut cli = CliArgs::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-v" | "--video-dir" => {
                let val = args.next().ok_or("Missing value for --video-dir")?;
                cli.video_dir = Some(PathBuf::from(val));
            }
            "-s" | "--scale" => {
                let val = args.next().ok_or("Missing value for --scale")?;
                cli.scale = Some(val.parse()?);
            }
            "-w" | "--width" => {
                let val = args.next().ok_or("Missing value for --width")?;
                cli.width = Some(val.parse()?);
            }
            "-h" | "--height" => {
                let val = args.next().ok_or("Missing value for --height")?;
                cli.height = Some(val.parse()?);
            }
            "-t" | "--threshold" => {
                let val = args.next().ok_or("Missing value for --threshold")?;
                cli.threshold = Some(val.parse()?);
            }
            "--fps" => {
                let val = args.next().ok_or("Missing value for --fps")?;
                cli.fps = Some(val.parse()?);
            }
            "-p" | "--personality" => {
                let val = args.next().ok_or("Missing value for --personality")?;
                cli.personality = Some(val);
            }
            "--engine" => {
                let val = args.next().ok_or("Missing value for --engine")?;
                cli.gemma_engine = Some(val.to_ascii_lowercase());
            }
            "--local" | "--gemma-3" | "--gemma-3-270m" => {
                cli.gemma_engine = Some("local".to_string());
                if cli.gemma_model.is_none() {
                    cli.gemma_model = Some("gemma-3-270m".to_string());
                }
            }
            "--model-path" => {
                let val = args.next().ok_or("Missing value for --model-path")?;
                cli.model_path = Some(PathBuf::from(val));
                if cli.gemma_engine.is_none() {
                    cli.gemma_engine = Some("local".to_string());
                }
            }
            "--gemma-model" => {
                let val = args.next().ok_or("Missing value for --gemma-model")?;
                cli.gemma_model = Some(val);
            }
            "--gemma-api-base" => {
                let val = args.next().ok_or("Missing value for --gemma-api-base")?;
                cli.gemma_api_base = Some(val);
            }
            "--brain-interval" => {
                let val = args.next().ok_or("Missing value for --brain-interval")?;
                cli.brain_interval = Some(val.parse()?);
            }
            "--no-brain" => {
                cli.disable_brain = true;
            }
            "-c" | "--chat" => {
                cli.chat = true;
            }
            other => {
                return Err(format!("Unknown option: {}", other).into());
            }
        }
    }

    if cli.video_dir.is_none()
        && let Ok(v) = env::var("SPARKY_VIDEO_DIR")
    {
        cli.video_dir = Some(PathBuf::from(v));
    }
    if cli.scale.is_none()
        && let Ok(v) = env::var("SPARKY_SCALE")
        && let Ok(val) = v.parse()
    {
        cli.scale = Some(val);
    }
    if cli.threshold.is_none()
        && let Ok(v) = env::var("SPARKY_BLACK_THRESHOLD")
        && let Ok(val) = v.parse()
    {
        cli.threshold = Some(val);
    }
    if cli.personality.is_none()
        && let Ok(v) = env::var("SPARKY_PERSONALITY")
    {
        cli.personality = Some(v);
    }
    if cli.gemma_engine.is_none()
        && let Ok(v) = env::var("GEMMA_ENGINE").or_else(|_| env::var("SPARKY_BRAIN_ENGINE"))
    {
        cli.gemma_engine = Some(v.to_ascii_lowercase());
    }
    if cli.model_path.is_none()
        && let Ok(v) = env::var("GEMMA_MODEL_PATH")
            .or_else(|_| env::var("GEMMA_LOCAL_MODEL_PATH"))
            .or_else(|_| env::var("SPARKY_MODEL_PATH"))
    {
        cli.model_path = Some(PathBuf::from(v));
    }
    if cli.gemma_model.is_none()
        && let Ok(v) = env::var("GEMMA_MODEL")
    {
        cli.gemma_model = Some(v);
    }
    if cli.gemma_api_base.is_none()
        && let Ok(v) = env::var("GEMMA_API_BASE")
    {
        cli.gemma_api_base = Some(v);
    }
    if cli.brain_interval.is_none()
        && let Ok(v) = env::var("SPARKY_BRAIN_INTERVAL_SECS")
        && let Ok(val) = v.parse()
    {
        cli.brain_interval = Some(val);
    }
    if !cli.disable_brain
        && let Ok(v) = env::var("SPARKY_DISABLE_BRAIN")
        && (v.eq_ignore_ascii_case("true") || v == "1")
    {
        cli.disable_brain = true;
    }

    Ok(cli)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chat_flag() {
        let cli = parse_args_from(vec!["--chat"]).unwrap();
        assert!(cli.chat);

        let cli_short = parse_args_from(vec!["-c"]).unwrap();
        assert!(cli_short.chat);

        let cli_default = parse_args_from(Vec::<String>::new()).unwrap();
        assert!(!cli_default.chat);
    }
}
