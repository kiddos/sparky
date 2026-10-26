use std::env;
use std::error::Error;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Clone)]
pub struct VideoAnimation {
    pub name: String,
    pub path: PathBuf,
}

pub struct VideoStream {
    child: std::process::Child,
    stdout: std::process::ChildStdout,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
    pub frame_size: usize,
    pub black_threshold: f32,
    pub fill_stack: Vec<u32>,
}

impl VideoStream {
    pub fn new(
        video_path: &Path,
        width: u32,
        height: u32,
        black_threshold: f32,
    ) -> Result<Self, Box<dyn Error>> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-v",
            "error",
            "-i",
            video_path.to_str().ok_or("Invalid video path")?,
            "-s",
            &format!("{}x{}", width, height),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ]);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::null());

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn ffmpeg: {}. Please ensure ffmpeg is installed and available in PATH.",
                e
            )
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or("Failed to capture ffmpeg stdout pipe")?;
        let frame_size = (width * height * 4) as usize;

        Ok(Self {
            child,
            stdout,
            width,
            height,
            frame_size,
            black_threshold,
            fill_stack: Vec::new(),
        })
    }

    pub fn read_frame(&mut self, frame_buf: &mut [u8]) -> Result<bool, Box<dyn Error>> {
        if frame_buf.len() < self.frame_size {
            return Err("Frame buffer too small".into());
        }

        match self.stdout.read_exact(&mut frame_buf[..self.frame_size]) {
            Ok(()) => {
                if self.black_threshold > 0.0 {
                    apply_background_mask(
                        &mut frame_buf[..self.frame_size],
                        self.width,
                        self.height,
                        self.black_threshold,
                        &mut self.fill_stack,
                    );
                }
                Ok(true)
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                let _ = self.child.wait();
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }
}

impl Drop for VideoStream {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Marks the black background as transparent (alpha = 0) in an RGBA frame.
///
/// Only dark pixels that are connected to the frame border are treated as
/// background (flood fill), so dark regions enclosed by Sparky — such as the
/// pupils — stay opaque.
pub fn apply_background_mask(
    frame: &mut [u8],
    width: u32,
    height: u32,
    black_threshold: f32,
    stack: &mut Vec<u32>,
) {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return;
    }
    let thresh = (black_threshold.clamp(0.0, 1.0) * 255.0) as u8;
    let is_dark = |frame: &[u8], i: usize| {
        let p = i * 4;
        frame[p].max(frame[p + 1]).max(frame[p + 2]) < thresh
    };

    // Start fully opaque; alpha 0 will mark visited background pixels.
    for px in frame.as_chunks_mut::<4>().0 {
        px[3] = 255;
    }

    stack.clear();
    let seed = |frame: &mut [u8], i: usize, stack: &mut Vec<u32>| {
        if frame[i * 4 + 3] != 0 && is_dark(frame, i) {
            frame[i * 4 + 3] = 0;
            stack.push(i as u32);
        }
    };
    for x in 0..w {
        seed(frame, x, stack);
        seed(frame, (h - 1) * w + x, stack);
    }
    for y in 0..h {
        seed(frame, y * w, stack);
        seed(frame, y * w + w - 1, stack);
    }

    while let Some(i) = stack.pop() {
        let i = i as usize;
        let (x, y) = (i % w, i / w);
        if x > 0 {
            seed(frame, i - 1, stack);
        }
        if x + 1 < w {
            seed(frame, i + 1, stack);
        }
        if y > 0 {
            seed(frame, i - w, stack);
        }
        if y + 1 < h {
            seed(frame, i + w, stack);
        }
    }
}

pub fn is_video_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        matches!(
            ext.to_ascii_lowercase().as_str(),
            "mp4" | "webm" | "mov" | "mkv" | "avi" | "m4v"
        )
    } else {
        false
    }
}

pub fn has_video_files(dir: &Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        entries
            .filter_map(|e| e.ok())
            .any(|e| is_video_file(&e.path()))
    } else {
        false
    }
}

pub fn find_videos_dir(cli_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = cli_dir
        && dir.is_dir()
        && has_video_files(dir)
    {
        return Some(dir.to_path_buf());
    }

    if let Ok(env_dir) = env::var("SPARKY_VIDEO_DIR") {
        let p = PathBuf::from(env_dir);
        if p.is_dir() && has_video_files(&p) {
            return Some(p);
        }
    }

    // Try relative paths in priority order: videos/v2 first, then videos
    let candidate_names = ["videos/v2", "videos"];

    for rel in &candidate_names {
        let p = PathBuf::from(rel);
        if p.is_dir() && has_video_files(&p) {
            return Some(p);
        }
    }

    // Relative to executable
    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        for rel in &candidate_names {
            let p = parent.join(rel);
            if p.is_dir() && has_video_files(&p) {
                return Some(p);
            }
            let p_target = parent.join("../../").join(rel);
            if p_target.is_dir() && has_video_files(&p_target) {
                return Some(p_target);
            }
        }
    }

    // Compile-time CARGO_MANIFEST_DIR
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for rel in &candidate_names {
        let p = manifest.join(rel);
        if p.is_dir() && has_video_files(&p) {
            return Some(p);
        }
    }

    None
}

pub fn load_video_animations(dir: &Path) -> Vec<VideoAnimation> {
    let mut animations = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| is_video_file(&e.path()))
            .collect();
        files.sort_by_key(|e| e.path());

        for file in files {
            let path = file.path();
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            animations.push(VideoAnimation { name, path });
        }
    }
    animations
}

pub fn probe_video_specs(path: &Path) -> (u32, u32, f64) {
    let default = (1280, 720, 24.0);

    let output = match Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0:s=x",
            path.to_str().unwrap_or(""),
        ])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => return default,
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.trim().split('x').collect();
    if parts.len() >= 3 {
        let width: u32 = parts[0].parse().unwrap_or(1280);
        let height: u32 = parts[1].parse().unwrap_or(720);

        let fps: f64 = if parts[2].contains('/') {
            let rat: Vec<&str> = parts[2].split('/').collect();
            let num: f64 = rat[0].parse().unwrap_or(24.0);
            let den: f64 = rat.get(1).and_then(|d| d.parse().ok()).unwrap_or(1.0);
            if den > 0.0 { num / den } else { 24.0 }
        } else {
            parts[2].parse().unwrap_or(24.0)
        };

        (width, height, fps)
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_video_file() {
        assert!(is_video_file(Path::new("test.mp4")));
        assert!(is_video_file(Path::new("test.MP4")));
        assert!(is_video_file(Path::new("test.webm")));
        assert!(is_video_file(Path::new("test.mov")));
        assert!(is_video_file(Path::new("test.mkv")));
        assert!(is_video_file(Path::new("test.avi")));
        assert!(is_video_file(Path::new("test.m4v")));
        assert!(!is_video_file(Path::new("test.png")));
        assert!(!is_video_file(Path::new("test.txt")));
    }

    #[test]
    fn test_apply_background_mask_simple() {
        // 2x2 image:
        // top-left black (border), top-right white (border)
        // bottom-left black (border), bottom-right black (border)
        let mut frame = vec![0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255];
        let mut stack = Vec::new();
        apply_background_mask(&mut frame, 2, 2, 0.1, &mut stack);

        // Black pixels connected to border should have alpha = 0
        assert_eq!(frame[3], 0);
        // White pixel should have alpha = 255
        assert_eq!(frame[7], 255);
        // Other black border pixels should have alpha = 0
        assert_eq!(frame[11], 0);
        assert_eq!(frame[15], 0);
    }
}
