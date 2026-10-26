use std::env;
use std::error::Error;
use std::path::PathBuf;

#[derive(Default)]
pub struct CliArgs {
    pub video_dir: Option<PathBuf>,
    pub scale: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub threshold: Option<f32>,
    pub fps: Option<f64>,
}

pub fn print_help() {
    println!("Sparky - Desktop Pet & Animated Character");
    println!("Usage: sparky [OPTIONS]");
    println!();
    println!("Options:");
    println!(
        "  -v, --video-dir <PATH>       Directory with video files (default: auto-detects videos/v2, videos)"
    );
    println!(
        "  -s, --scale <FLOAT>          Display scale factor (e.g. 1.0 for 1280x720, 0.5 for 640x360)"
    );
    println!("  -w, --width <INT>            Window width in pixels");
    println!("  -h, --height <INT>           Window height in pixels");
    println!(
        "  -t, --threshold <FLOAT>      Black color threshold to ignore [0.0 - 1.0] (default: 0.08)"
    );
    println!(
        "      --fps <FLOAT>            Frame rate for animation (default: auto-detected or 24.0)"
    );
    println!("      --help                   Print this help message");
}

pub fn parse_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut args = env::args().skip(1);
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

    Ok(cli)
}
