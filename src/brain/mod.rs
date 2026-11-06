pub mod candle_engine;
pub mod gguf_engine;
pub mod prompt;

use std::io::{Read, Write};
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
#[allow(dead_code)]
pub const GEMMA3_GGUF_REPO: &str = "unsloth/gemma-3-270m-it-GGUF";

/// Direct Hugging Face download URL for Gemma 3 270M Q4_K_M GGUF (Primary)
pub const GEMMA3_GGUF_DIRECT_URL: &str =
    "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf";

/// Fallback Hugging Face mirror URL for Gemma 3 270M Q4_K_M GGUF (Secondary)
pub const GEMMA3_GGUF_FALLBACK_URL: &str = "https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-Q4_K_M.gguf";

/// Candidate download URLs for Gemma 3 270M Q4_K_M GGUF model
pub const GEMMA3_GGUF_DOWNLOAD_URLS: &[&str] = &[GEMMA3_GGUF_DIRECT_URL, GEMMA3_GGUF_FALLBACK_URL];

/// Minimum plausible size for the Gemma 3 270M GGUF model (~253 MB uncompressed)
pub const MIN_GGUF_FILE_SIZE: u64 = 50_000_000;

/// Progress and completion events emitted during background model download.
#[derive(Debug, Clone)]
pub enum DownloadProgressEvent {
    Progress { downloaded: u64, total: Option<u64> },
    Completed(Result<PathBuf, String>),
}

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

    // Check standard default locations only if model_path does not exist or is the default directory
    if !model_path.exists()
        || model_path == Path::new("models")
        || model_path == Path::new("models/gemma-3-270m")
    {
        let default_path1 = Path::new("models/gemma-3-270m").join(GEMMA3_GGUF_FILENAME);
        if default_path1.is_file() {
            return Some(default_path1);
        }
        let default_path2 = Path::new("models/gemma-3-270m").join(GEMMA3_GGUF_UPSTREAM_FILENAME);
        if default_path2.is_file() {
            return Some(default_path2);
        }
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

/// Verifies that a file exists, is at least `MIN_GGUF_FILE_SIZE`, and starts with GGUF magic bytes (`b"GGUF"`).
pub fn verify_gguf_file(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("File does not exist: {}", path.display()));
    }
    let metadata = path
        .metadata()
        .map_err(|e| format!("Failed to read metadata for {}: {e}", path.display()))?;
    if metadata.len() < MIN_GGUF_FILE_SIZE {
        return Err(format!(
            "File size ({} bytes) is too small (expected >= {} bytes)",
            metadata.len(),
            MIN_GGUF_FILE_SIZE
        ));
    }
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open file {}: {e}", path.display()))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| format!("Failed to read magic bytes from {}: {e}", path.display()))?;
    if &magic != b"GGUF" {
        return Err(format!(
            "File header does not match GGUF magic bytes: expected b\"GGUF\", got {:?}",
            String::from_utf8_lossy(&magic)
        ));
    }
    Ok(())
}

/// Downloads the Gemma 3 270M GGUF model using pure Rust HTTP/HTTPS client (`ureq`).
///
/// Implemented entirely in Rust without external scripts or external binaries:
/// - Streams HTTP response body into a temporary `.part` file.
/// - Periodically triggers `on_progress(downloaded_bytes, total_bytes)`.
/// - Supports `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` bearer authentication.
/// - Follows redirects automatically.
/// - Validates file size and magic header (`b"GGUF"`).
/// - Atomically moves the verified file to `dest_path`.
pub fn download_gguf_model_from_urls_with_progress<F>(
    dest_path: &Path,
    urls: &[&str],
    mut on_progress: F,
) -> anyhow::Result<PathBuf>
where
    F: FnMut(u64, Option<u64>),
{
    // If the destination file already exists and passes GGUF verification, no re-download needed
    if dest_path.is_file() && verify_gguf_file(dest_path).is_ok() {
        log::info!(
            "GGUF model already exists and is verified at {:?}",
            dest_path
        );
        let len = dest_path.metadata().map(|m| m.len()).unwrap_or(0);
        on_progress(len, Some(len));
        return Ok(dest_path.to_path_buf());
    }

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let agent = ureq::Agent::config_builder()
        .max_redirects(10)
        .build()
        .new_agent();

    let tmp_path = dest_path.with_extension(format!("part.{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp_path);

    let mut last_err = None;

    for &url in urls {
        log::info!("Downloading GGUF model from {} to {:?}", url, dest_path);

        let mut req = agent
            .get(url)
            .header("User-Agent", "sparky-desktop-pet/0.1.0");

        if let Ok(token) =
            std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        {
            let trimmed = token.trim();
            if !trimmed.is_empty() {
                req = req.header("Authorization", format!("Bearer {trimmed}"));
            }
        }

        let mut resp = match req.call() {
            Ok(r) => r,
            Err(e) => {
                log::warn!("Failed HTTP GET request to {}: {}", url, e);
                last_err = Some(anyhow::anyhow!("Request to {url} failed: {e}"));
                continue;
            }
        };

        let status = resp.status();
        if !status.is_success() {
            log::warn!("HTTP status {} received from {}", status, url);
            last_err = Some(anyhow::anyhow!("HTTP error {status} from {url}"));
            continue;
        }

        let total_bytes = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());

        let mut file = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to create temporary download file {:?}: {}",
                    tmp_path,
                    e
                ));
            }
        };

        let mut reader = resp.body_mut().as_reader();
        let mut buffer = [0u8; 64 * 1024]; // 64 KB buffer
        let mut downloaded: u64 = 0;
        let mut last_reported = 0u64;
        let mut last_report_time = std::time::Instant::now();
        let mut stream_err = None;

        on_progress(0, total_bytes);

        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = file.write_all(&buffer[..n]) {
                        stream_err = Some(anyhow::anyhow!("File write error: {e}"));
                        break;
                    }
                    downloaded += n as u64;

                    let elapsed = last_report_time.elapsed();
                    if elapsed >= std::time::Duration::from_millis(250)
                        || downloaded.saturating_sub(last_reported) >= 1_048_576
                    {
                        on_progress(downloaded, total_bytes);
                        last_reported = downloaded;
                        last_report_time = std::time::Instant::now();
                    }
                }
                Err(e) => {
                    log::warn!("Read error streaming from {}: {}", url, e);
                    stream_err = Some(anyhow::anyhow!("Stream read error from {url}: {e}"));
                    break;
                }
            }
        }

        if let Some(err) = stream_err {
            let _ = std::fs::remove_file(&tmp_path);
            last_err = Some(err);
            continue;
        }

        if let Err(e) = file.flush() {
            let _ = std::fs::remove_file(&tmp_path);
            last_err = Some(anyhow::anyhow!("File flush error: {e}"));
            continue;
        }
        drop(file);

        on_progress(downloaded, total_bytes);

        // Verify the downloaded file
        if let Err(err_msg) = verify_gguf_file(&tmp_path) {
            log::warn!(
                "Downloaded file from {} failed verification: {}",
                url,
                err_msg
            );
            let _ = std::fs::remove_file(&tmp_path);
            last_err = Some(anyhow::anyhow!("Verification failed: {err_msg}"));
            continue;
        }

        // Atomically rename temporary file to destination path
        if let Err(e) = std::fs::rename(&tmp_path, dest_path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(anyhow::anyhow!(
                "Failed to rename {:?} to {:?}: {}",
                tmp_path,
                dest_path,
                e
            ));
        }

        log::info!(
            "GGUF model downloaded and verified successfully at {:?}",
            dest_path
        );
        return Ok(dest_path.to_path_buf());
    }

    let _ = std::fs::remove_file(&tmp_path);
    Err(last_err.unwrap_or_else(|| {
        anyhow::anyhow!(
            "Failed to download GGUF model to {}. Please check your connection.",
            dest_path.display()
        )
    }))
}

/// Downloads the Gemma 3 270M GGUF model with a progress callback.
pub fn download_gguf_model_with_progress<F>(
    dest_path: &Path,
    on_progress: F,
) -> anyhow::Result<PathBuf>
where
    F: FnMut(u64, Option<u64>),
{
    download_gguf_model_from_urls_with_progress(dest_path, GEMMA3_GGUF_DOWNLOAD_URLS, on_progress)
}

/// Downloads the Gemma 3 270M GGUF model to the specified destination path.
#[allow(dead_code)]
pub fn download_gguf_model(dest_path: &Path) -> anyhow::Result<PathBuf> {
    download_gguf_model_with_progress(dest_path, |_, _| {})
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

    #[test]
    fn test_verify_gguf_file() {
        let temp_dir =
            std::env::temp_dir().join(format!("sparky_test_verify_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let non_existent = temp_dir.join("missing.gguf");
        assert!(verify_gguf_file(&non_existent).is_err());

        // File too small (< 50MB)
        let too_small = temp_dir.join("too_small.gguf");
        std::fs::write(&too_small, b"GGUF_small").unwrap();
        let err = verify_gguf_file(&too_small).unwrap_err();
        assert!(err.contains("too small"), "{err}");

        // Valid size (>= 50MB) but bad magic header
        let bad_magic = temp_dir.join("bad_magic.gguf");
        let file = std::fs::File::create(&bad_magic).unwrap();
        file.set_len(MIN_GGUF_FILE_SIZE).unwrap();
        drop(file);
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&bad_magic)
            .unwrap();
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(0)).unwrap();
        f.write_all(b"BAD!").unwrap();
        drop(f);
        let err = verify_gguf_file(&bad_magic).unwrap_err();
        assert!(err.contains("magic bytes"), "{err}");

        // Valid size and valid magic header
        let valid_gguf = temp_dir.join("valid.gguf");
        let file = std::fs::File::create(&valid_gguf).unwrap();
        file.set_len(MIN_GGUF_FILE_SIZE).unwrap();
        drop(file);
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&valid_gguf)
            .unwrap();
        f.seek(std::io::SeekFrom::Start(0)).unwrap();
        f.write_all(b"GGUF").unwrap();
        drop(f);
        assert!(verify_gguf_file(&valid_gguf).is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_gguf_pure_rust_with_mock_server() {
        use std::io::Write as _;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind listener");
        let local_addr = listener.local_addr().expect("local addr");
        let mock_url = format!("http://127.0.0.1:{}/model.gguf", local_addr.port());

        let total_size: usize = (MIN_GGUF_FILE_SIZE + 4) as usize;

        let server_thread = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("Accept failed");
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf);

            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                total_size
            );
            socket.write_all(header.as_bytes()).unwrap();
            socket.write_all(b"GGUF").unwrap();

            let zero_chunk = [0u8; 64 * 1024];
            let mut remaining = total_size - 4;
            while remaining > 0 {
                let to_write = remaining.min(zero_chunk.len());
                socket.write_all(&zero_chunk[..to_write]).unwrap();
                remaining -= to_write;
            }
            socket.flush().unwrap();
        });

        let temp_dir = std::env::temp_dir().join(format!("sparky_test_dl_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let dest_file = temp_dir.join("downloaded.gguf");

        let mut progress_calls = 0;
        let mut max_downloaded = 0;
        let result =
            download_gguf_model_from_urls_with_progress(&dest_file, &[&mock_url], |dl, total| {
                progress_calls += 1;
                max_downloaded = max_downloaded.max(dl);
                assert_eq!(total, Some(total_size as u64));
            });

        server_thread.join().unwrap();

        assert!(result.is_ok(), "Download failed: {:?}", result.err());
        assert!(dest_file.is_file());
        assert_eq!(dest_file.metadata().unwrap().len(), total_size as u64);
        assert!(verify_gguf_file(&dest_file).is_ok());
        assert!(progress_calls > 0);
        assert_eq!(max_downloaded, total_size as u64);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_gguf_rejects_corrupted_stream() {
        use std::io::Write as _;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind listener");
        let local_addr = listener.local_addr().expect("local addr");
        let mock_url = format!("http://127.0.0.1:{}/bad.gguf", local_addr.port());

        let total_size: usize = (MIN_GGUF_FILE_SIZE + 4) as usize;

        let server_thread = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("Accept failed");
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf);

            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                total_size
            );
            socket.write_all(header.as_bytes()).unwrap();
            socket.write_all(b"BAD!").unwrap();

            let zero_chunk = [0u8; 64 * 1024];
            let mut remaining = total_size - 4;
            while remaining > 0 {
                let to_write = remaining.min(zero_chunk.len());
                socket.write_all(&zero_chunk[..to_write]).unwrap();
                remaining -= to_write;
            }
            socket.flush().unwrap();
        });

        let temp_dir =
            std::env::temp_dir().join(format!("sparky_test_bad_dl_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let dest_file = temp_dir.join("corrupted.gguf");

        let result =
            download_gguf_model_from_urls_with_progress(&dest_file, &[&mock_url], |_, _| {});

        server_thread.join().unwrap();

        assert!(
            result.is_err(),
            "Expected corrupted stream to fail verification"
        );
        assert!(
            !dest_file.exists(),
            "Destination file must not exist on failed verification"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_gguf_fallback_mirrors() {
        use std::io::Write as _;
        use std::net::TcpListener;

        // Server 1: returns HTTP 500 error
        let listener1 = TcpListener::bind("127.0.0.1:0").expect("Failed to bind listener 1");
        let port1 = listener1.local_addr().unwrap().port();
        let url1 = format!("http://127.0.0.1:{}/error.gguf", port1);

        // Server 2: returns HTTP 200 with valid GGUF
        let listener2 = TcpListener::bind("127.0.0.1:0").expect("Failed to bind listener 2");
        let port2 = listener2.local_addr().unwrap().port();
        let url2 = format!("http://127.0.0.1:{}/good.gguf", port2);

        let total_size: usize = (MIN_GGUF_FILE_SIZE + 4) as usize;

        let t1 = std::thread::spawn(move || {
            if let Ok((mut socket, _)) = listener1.accept() {
                let mut buf = [0u8; 1024];
                let _ = socket.read(&mut buf);
                let resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
                let _ = socket.write_all(resp.as_bytes());
            }
        });

        let t2 = std::thread::spawn(move || {
            let (mut socket, _) = listener2.accept().expect("Accept failed on listener 2");
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf);

            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                total_size
            );
            socket.write_all(header.as_bytes()).unwrap();
            socket.write_all(b"GGUF").unwrap();

            let zero_chunk = [0u8; 64 * 1024];
            let mut remaining = total_size - 4;
            while remaining > 0 {
                let to_write = remaining.min(zero_chunk.len());
                socket.write_all(&zero_chunk[..to_write]).unwrap();
                remaining -= to_write;
            }
            socket.flush().unwrap();
        });

        let temp_dir =
            std::env::temp_dir().join(format!("sparky_test_fallback_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let dest_file = temp_dir.join("fallback.gguf");

        let result =
            download_gguf_model_from_urls_with_progress(&dest_file, &[&url1, &url2], |_, _| {});

        t1.join().unwrap();
        t2.join().unwrap();

        assert!(
            result.is_ok(),
            "Fallback download failed: {:?}",
            result.err()
        );
        assert!(dest_file.is_file());
        assert!(verify_gguf_file(&dest_file).is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_gguf_skips_if_already_verified() {
        let temp_dir =
            std::env::temp_dir().join(format!("sparky_test_skip_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let dest_file = temp_dir.join("already_verified.gguf");

        // Create valid GGUF file
        let file = std::fs::File::create(&dest_file).unwrap();
        file.set_len(MIN_GGUF_FILE_SIZE).unwrap();
        drop(file);
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&dest_file)
            .unwrap();
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(0)).unwrap();
        f.write_all(b"GGUF").unwrap();
        drop(f);

        // Dummy invalid URL that would fail if connected to
        let invalid_url = "http://127.0.0.1:9/never_connected.gguf";

        let mut progress_called = false;
        let result =
            download_gguf_model_from_urls_with_progress(&dest_file, &[invalid_url], |dl, total| {
                progress_called = true;
                assert_eq!(dl, MIN_GGUF_FILE_SIZE);
                assert_eq!(total, Some(MIN_GGUF_FILE_SIZE));
            });

        assert!(
            result.is_ok(),
            "Pre-verified model should return Ok immediately"
        );
        assert!(
            progress_called,
            "Progress should report 100% for pre-verified file"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
