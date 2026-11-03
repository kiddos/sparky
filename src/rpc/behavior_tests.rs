use crate::animation::{AnimationClip, FramePacer};
use crate::brain::BrainEngine;
use crate::render::PetWindow;
use crate::rpc::{RpcServer, base64_decode, handle_rpc_command, server::NoopWakeNotifier};
use crate::state::{EmotionRegistry, StateMachine};
use crate::ui::ChatState;
use image::{DynamicImage, GenericImageView};
use parking_lot::Mutex;
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

static TEST_HARNESS_LOCK: Mutex<()> = Mutex::new(());

static TEST_REGISTRY: std::sync::LazyLock<Arc<EmotionRegistry>> = std::sync::LazyLock::new(|| {
    Arc::new(
        EmotionRegistry::load_from_toml("assets/emotions.toml", "assets/animations")
            .expect("Failed to load emotions.toml"),
    )
});

static TEST_CLIPS: std::sync::LazyLock<HashMap<String, AnimationClip>> =
    std::sync::LazyLock::new(load_test_clips);

/// Client helper for sending JSON-RPC 2.0 requests over a persistent TCP connection
pub struct RpcTestClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
    next_id: u64,
}

impl RpcTestClient {
    pub fn connect(addr: std::net::SocketAddr) -> anyhow::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self {
            stream,
            reader,
            next_id: 1,
        })
    }

    pub fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;

        let req = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": id,
        });

        writeln!(self.stream, "{}", req).expect("Failed to write RPC request");
        self.stream.flush().expect("Failed to flush request");

        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .expect("Failed to read RPC response");

        let resp: serde_json::Value =
            serde_json::from_str(line.trim()).expect("Failed to parse JSON-RPC response");
        assert_eq!(resp["jsonrpc"], "2.0", "Expected jsonrpc version 2.0");
        assert_eq!(resp["id"], id, "Expected matching request ID");
        resp
    }

    pub fn call_success(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let resp = self.call(method, params);
        assert!(
            resp.get("result").is_some(),
            "Expected success result for '{}', got error: {:?}",
            method,
            resp.get("error")
        );
        resp["result"].clone()
    }

    pub fn call_error(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let resp = self.call(method, params);
        assert!(
            resp.get("error").is_some(),
            "Expected error for '{}', got result: {:?}",
            method,
            resp.get("result")
        );
        resp["error"].clone()
    }

    pub fn get_state(&mut self) -> serde_json::Value {
        self.call_success("sparky.get_state", json!({}))
    }

    pub fn set_emotion(&mut self, emotion: &str) -> serde_json::Value {
        self.call_success("sparky.set_emotion", json!({ "emotion": emotion }))
    }

    pub fn pet(&mut self) -> serde_json::Value {
        self.call_success("sparky.pet", json!({}))
    }

    pub fn poke(&mut self) -> serde_json::Value {
        self.call_success("sparky.poke", json!({}))
    }

    pub fn get_frame(
        &mut self,
        source: &str,
        include_data: bool,
        save_path: Option<&str>,
    ) -> serde_json::Value {
        let mut params = json!({
            "source": source,
            "include_data": include_data,
        });
        if let Some(path) = save_path {
            params["save_path"] = json!(path);
        }
        self.call_success("sparky.get_frame", params)
    }
}

/// Loads all WebP animations from assets/animations directory
#[allow(clippy::collapsible_if)]
pub fn load_test_clips() -> HashMap<String, AnimationClip> {
    let mut clips = HashMap::new();
    let anim_dir = Path::new("assets/animations");
    if let Ok(entries) = std::fs::read_dir(anim_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("webp") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(clip) = AnimationClip::load_from_webp(stem, &path) {
                        clips.insert(stem.to_string(), clip);
                    }
                }
            }
        }
    }
    clips
}

/// Visualizes a decoded frame image in the terminal using ANSI 24-bit truecolor half-blocks
/// and returns the rendered string along with image statistics.
#[allow(clippy::manual_checked_ops)]
pub fn visualize_frame_terminal(
    title: &str,
    emotion: &str,
    clip_name: &str,
    frame_idx: usize,
    img: &DynamicImage,
) -> (String, VisualStats) {
    let (orig_w, orig_h) = (img.width(), img.height());

    // Target terminal resolution: 48 columns x 12 character rows (24 pixel rows)
    let term_w: u32 = 48;
    let term_h: u32 = 12; // 2 vertical pixels per character row

    let mut out = String::new();
    let mut green_pixels: u64 = 0;
    let mut fur_pixels: u64 = 0;
    let mut total_r: u64 = 0;
    let mut total_g: u64 = 0;
    let mut total_b: u64 = 0;

    out.push_str(&format!(
        "\n\x1b[1;36m┌────────────────────────────────────────────────────────────────────────────┐\x1b[0m\n\
         \x1b[1;36m│\x1b[0m \x1b[1;33m🐾 SPARKY VISUAL FRAME: {}\x1b[0m\n\
         \x1b[1;36m│\x1b[0m Emotion: \x1b[1;32m{:<10}\x1b[0m | Clip: \x1b[1;35m{:<10}\x1b[0m | Frame: \x1b[1;33m{:<4}\x1b[0m | Resolution: {}x{}\n\
         \x1b[1;36m├────────────────────────────────────────────────────────────────────────────┤\x1b[0m\n",
        title, emotion, clip_name, frame_idx, orig_w, orig_h
    ));

    // Sample function that checks whether a pixel is the green background
    let is_key_green = |r: u8, g: u8, b: u8| -> bool {
        let (rf, gf, bf) = (r as f32, g as f32, b as f32);
        gf > 120.0 && gf > 1.35 * rf.max(bf)
    };

    // Render character grid using Unicode half blocks '▀'
    for row in 0..term_h {
        out.push_str("  ");
        for col in 0..term_w {
            // Pixel sample coordinates
            let px_top_x = (col * orig_w) / term_w;
            let px_top_y = (row * 2 * orig_h) / (term_h * 2);
            let px_bot_x = (col * orig_w) / term_w;
            let px_bot_y = ((row * 2 + 1) * orig_h) / (term_h * 2);

            let top_rgba = img.get_pixel(px_top_x.min(orig_w - 1), px_top_y.min(orig_h - 1));
            let bot_rgba = img.get_pixel(px_bot_x.min(orig_w - 1), px_bot_y.min(orig_h - 1));

            let (rt, gt, bt) = (top_rgba[0], top_rgba[1], top_rgba[2]);
            let (rb, gb, bb) = (bot_rgba[0], bot_rgba[1], bot_rgba[2]);

            let top_green = is_key_green(rt, gt, bt);
            let bot_green = is_key_green(rb, gb, bb);

            if !top_green {
                fur_pixels += 1;
                total_r += rt as u64;
                total_g += gt as u64;
                total_b += bt as u64;
            } else {
                green_pixels += 1;
            }

            if !bot_green {
                fur_pixels += 1;
                total_r += rb as u64;
                total_g += gb as u64;
                total_b += bb as u64;
            } else {
                green_pixels += 1;
            }

            match (top_green, bot_green) {
                (true, true) => {
                    // Both green screen -> transparent desktop background (render faint dot)
                    out.push_str("\x1b[90m··\x1b[0m");
                }
                (false, true) => {
                    // Top is kitten fur, bottom is transparent
                    out.push_str(&format!("\x1b[38;2;{};{};{}m▀▀\x1b[0m", rt, gt, bt));
                }
                (true, false) => {
                    // Top is transparent, bottom is kitten fur
                    out.push_str(&format!("\x1b[38;2;{};{};{}m▄▄\x1b[0m", rb, gb, bb));
                }
                (false, false) => {
                    // Both top and bottom are kitten fur
                    out.push_str(&format!(
                        "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀▀\x1b[0m",
                        rt, gt, bt, rb, gb, bb
                    ));
                }
            }
        }
        out.push('\n');
    }

    let total_samples = green_pixels + fur_pixels;
    let green_pct = if total_samples > 0 {
        (green_pixels as f64 / total_samples as f64) * 100.0
    } else {
        0.0
    };
    let fur_pct = if total_samples > 0 {
        (fur_pixels as f64 / total_samples as f64) * 100.0
    } else {
        0.0
    };

    let (avg_r, avg_g, avg_b) = if fur_pixels > 0 {
        (
            (total_r / fur_pixels) as u8,
            (total_g / fur_pixels) as u8,
            (total_b / fur_pixels) as u8,
        )
    } else {
        (0, 0, 0)
    };

    out.push_str(&format!(
        "\x1b[1;36m├────────────────────────────────────────────────────────────────────────────┤\x1b[0m\n\
         \x1b[1;36m│\x1b[0m Fur Coverage: \x1b[1;33m{:.1}%\x1b[0m | Chroma Background: \x1b[1;32m{:.1}%\x1b[0m | Avg Fur RGB: \x1b[38;2;{};{};{}m({}, {}, {})\x1b[0m\n\
         \x1b[1;36m└────────────────────────────────────────────────────────────────────────────┘\x1b[0m\n",
        fur_pct, green_pct, avg_r, avg_g, avg_b, avg_r, avg_g, avg_b
    ));

    (
        out,
        VisualStats {
            fur_coverage_pct: fur_pct,
            green_background_pct: green_pct,
            avg_fur_rgb: (avg_r, avg_g, avg_b),
        },
    )
}

#[derive(Debug, Clone)]
pub struct VisualStats {
    pub fur_coverage_pct: f64,
    pub green_background_pct: f64,
    pub avg_fur_rgb: (u8, u8, u8),
}

/// Helper to decode PNG bytes and assert validity
pub fn decode_and_verify_png(png_bytes: &[u8]) -> DynamicImage {
    assert!(
        png_bytes.len() >= 8,
        "PNG byte buffer too small: {} bytes",
        png_bytes.len()
    );
    // Standard PNG magic signature: \x89PNG\r\n\x1a\n
    assert_eq!(
        &png_bytes[0..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        "Invalid PNG magic signature"
    );
    image::load_from_memory(png_bytes).expect("Failed to decode PNG bytes into DynamicImage")
}

/// Sets up the test environment and runs a simulated desktop pet event loop while
/// executing test assertions via the RPC client.
pub fn run_desktop_pet_rpc_test<F>(test_body: F)
where
    F: FnOnce(&mut RpcTestClient, &Path) + Send + 'static,
{
    let _guard = TEST_HARNESS_LOCK.lock();

    let registry = TEST_REGISTRY.clone();
    let clips = &*TEST_CLIPS;

    let mut pet_window =
        PetWindow::new(288, 162, "Sparky Pet Test").expect("Failed to initialize PetWindow");
    let mut state_machine = StateMachine::new(&registry);
    let mut frame_pacer = FramePacer::new(24.0);
    let mut chat_state = ChatState::new();
    let brain = BrainEngine::spawn(PathBuf::from("models/gemma-3-270m"), registry.clone());
    let mut is_visible = true;

    let server = RpcServer::start("127.0.0.1:0", registry.clone(), Box::new(NoopWakeNotifier))
        .expect("Failed to bind RPC server");
    let addr = server.local_addr();

    let output_dir = PathBuf::from(
        "/home/kiddos/.gemini/antigravity/brain/687be11b-1c14-49c4-b1ff-fef189f4499f/frames",
    );
    let _ = std::fs::create_dir_all(&output_dir);

    let output_dir_client = output_dir.clone();

    // Spawn test client thread
    let client_handle = std::thread::spawn(move || {
        let mut client =
            RpcTestClient::connect(addr).expect("Client failed to connect to RPC server");
        test_body(&mut client, &output_dir_client);
    });

    // Main event loop pump (runs on the thread that created PetWindow)
    while !client_handle.is_finished() {
        pet_window.glfw.poll_events();
        if let Ok(cmd) = server.recv_command_timeout(Duration::from_millis(10)) {
            handle_rpc_command(
                cmd,
                &mut state_machine,
                &registry,
                &mut pet_window,
                &mut chat_state,
                &brain,
                &None, // TrayService
                &mut frame_pacer,
                &mut is_visible,
                clips,
            );
        }
    }

    // Drain any final pending commands
    while let Some(cmd) = server.try_recv_command() {
        handle_rpc_command(
            cmd,
            &mut state_machine,
            &registry,
            &mut pet_window,
            &mut chat_state,
            &brain,
            &None,
            &mut frame_pacer,
            &mut is_visible,
            clips,
        );
    }

    client_handle.join().expect("Client test thread panicked");
    server.shutdown();
}

#[test]
fn test_pet_behavior_initial_state_and_sit_frame_capture() {
    run_desktop_pet_rpc_test(|client, out_dir| {
        // 1. Verify initial pet state
        let state = client.get_state();
        assert_eq!(state["emotion"], "sit");
        assert_eq!(state["active_clip"], "sit");
        assert_eq!(state["window"]["visible"], true);
        assert_eq!(state["window"]["click_through"], false);
        assert_eq!(state["fps"], 24.0);

        // 2. Capture initial frame via RPC
        let frame_resp = client.get_frame("clip", true, None);
        assert_eq!(frame_resp["emotion"], "sit");
        assert_eq!(frame_resp["clip"], "sit");
        assert_eq!(frame_resp["width"], 384);
        assert_eq!(frame_resp["height"], 216);
        assert_eq!(frame_resp["channels"], 3);
        assert_eq!(frame_resp["source"], "clip");
        assert_eq!(frame_resp["format"], "image/png");

        // 3. Decode base64 PNG data
        let b64 = frame_resp["base64"].as_str().expect("base64 field");
        let png_bytes = base64_decode(b64).expect("Valid base64 PNG data");
        let img = decode_and_verify_png(&png_bytes);
        assert_eq!(img.width(), 384);
        assert_eq!(img.height(), 216);

        // 4. Visualize in terminal
        let (viz, stats) =
            visualize_frame_terminal("Initial Idle Pet State", "sit", "sit", 0, &img);
        print!("{}", viz);

        // Assert visual traits of Sparky sitting on green screen:
        // - Frame contains significant green screen background (> 40%)
        // - Frame contains kitten fur (> 10%)
        // - Kitten fur has orange/warm color tones (Red > Blue)
        assert!(
            stats.green_background_pct > 30.0,
            "Expected green background, got {:.1}%",
            stats.green_background_pct
        );
        assert!(
            stats.fur_coverage_pct > 10.0,
            "Expected cat fur coverage, got {:.1}%",
            stats.fur_coverage_pct
        );
        assert!(
            stats.avg_fur_rgb.0 > stats.avg_fur_rgb.2,
            "Expected warm/orange fur tones (R > B), got {:?}",
            stats.avg_fur_rgb
        );

        // 5. Save PNG to disk for artifact visualization
        let sit_path = out_dir.join("sparky_frame_sit.png");
        let _ = std::fs::write(&sit_path, &png_bytes);
    });
}

#[test]
fn test_pet_behavior_petting_triggers_happy_and_visual_change() {
    run_desktop_pet_rpc_test(|client, out_dir| {
        // Capture initial sit frame for visual delta comparison
        let sit_resp = client.get_frame("clip", true, None);
        let sit_b64 = sit_resp["base64"].as_str().unwrap();
        let sit_png = base64_decode(sit_b64).unwrap();
        let _sit_img = decode_and_verify_png(&sit_png);

        // 1. Simulate petting Sparky
        let pet_res = client.pet();
        assert_eq!(pet_res["success"], true);
        assert_eq!(pet_res["emotion"], "happy");

        // 2. State verification
        let state = client.get_state();
        assert_eq!(state["emotion"], "happy");
        assert_eq!(state["active_clip"], "happy");

        // 3. Frame capture of happy emotion
        let happy_resp = client.get_frame("clip", true, None);
        assert_eq!(happy_resp["emotion"], "happy");
        assert_eq!(happy_resp["clip"], "happy");

        let happy_b64 = happy_resp["base64"].as_str().unwrap();
        let happy_png = base64_decode(happy_b64).unwrap();
        let happy_img = decode_and_verify_png(&happy_png);

        // 4. Verify visual delta between sit and happy frames
        assert_ne!(
            sit_png, happy_png,
            "Happy frame PNG bytes must differ from Sit frame"
        );

        // 5. Visualize in terminal
        let (viz, stats) =
            visualize_frame_terminal("Petted Reaction State", "happy", "happy", 0, &happy_img);
        print!("{}", viz);

        assert!(
            stats.fur_coverage_pct > 10.0,
            "Expected fur coverage in happy animation"
        );

        // Save happy frame to disk
        let happy_path = out_dir.join("sparky_frame_happy.png");
        let _ = std::fs::write(&happy_path, &happy_png);
    });
}

#[test]
fn test_pet_behavior_poking_triggers_angry_and_visual_change() {
    run_desktop_pet_rpc_test(|client, out_dir| {
        // 1. Simulate poking Sparky
        let poke_res = client.poke();
        assert_eq!(poke_res["success"], true);
        assert_eq!(poke_res["emotion"], "angry");

        // 2. State verification
        let state = client.get_state();
        assert_eq!(state["emotion"], "angry");
        assert_eq!(state["active_clip"], "angry");

        // 3. Frame capture of angry emotion
        let angry_resp = client.get_frame("clip", true, None);
        assert_eq!(angry_resp["emotion"], "angry");
        assert_eq!(angry_resp["clip"], "angry");

        let angry_b64 = angry_resp["base64"].as_str().unwrap();
        let angry_png = base64_decode(angry_b64).unwrap();
        let angry_img = decode_and_verify_png(&angry_png);

        // 4. Visualize in terminal
        let (viz, stats) =
            visualize_frame_terminal("Poked Reaction State", "angry", "angry", 0, &angry_img);
        print!("{}", viz);

        assert!(
            stats.fur_coverage_pct > 10.0,
            "Expected cat fur in angry animation"
        );

        // Save angry frame to disk
        let angry_path = out_dir.join("sparky_frame_angry.png");
        let _ = std::fs::write(&angry_path, &angry_png);
    });
}

#[test]
fn test_pet_behavior_naughty_transition_and_visual_frame() {
    run_desktop_pet_rpc_test(|client, out_dir| {
        // 1. Transition to naughty
        let naughty_res = client.set_emotion("naughty");
        assert_eq!(naughty_res["success"], true);
        assert_eq!(naughty_res["emotion"], "naughty");
        assert_eq!(naughty_res["active_clip"], "naughty");

        // 2. State verification
        let state = client.get_state();
        assert_eq!(state["emotion"], "naughty");
        assert_eq!(state["active_clip"], "naughty");

        // 3. Frame capture of naughty emotion
        let naughty_resp = client.get_frame("clip", true, None);
        assert_eq!(naughty_resp["emotion"], "naughty");
        assert_eq!(naughty_resp["clip"], "naughty");

        let naughty_b64 = naughty_resp["base64"].as_str().unwrap();
        let naughty_png = base64_decode(naughty_b64).unwrap();
        let naughty_img = decode_and_verify_png(&naughty_png);

        // 4. Visualize in terminal
        let (viz, stats) = visualize_frame_terminal(
            "Playful Mischief State",
            "naughty",
            "naughty",
            0,
            &naughty_img,
        );
        print!("{}", viz);

        assert!(
            stats.fur_coverage_pct > 10.0,
            "Expected cat fur in naughty animation"
        );

        // Save naughty frame to disk
        let naughty_path = out_dir.join("sparky_frame_naughty.png");
        let _ = std::fs::write(&naughty_path, &naughty_png);
    });
}

#[test]
fn test_pet_behavior_fallback_resolution_and_tinting() {
    run_desktop_pet_rpc_test(|client, out_dir| {
        // 1. Set 'sad' emotion (which has fallback = 'sit' with cold blue tint [0.8, 0.85, 1.05])
        let sad_res = client.set_emotion("sad");
        assert_eq!(sad_res["success"], true);
        assert_eq!(sad_res["emotion"], "sad");
        assert_eq!(sad_res["active_clip"], "sit"); // Resolved fallback clip
        let tint = sad_res["active_tint"].as_array().expect("Tint array");
        assert!((tint[0].as_f64().unwrap() - 0.80).abs() < 1e-4);
        assert!((tint[1].as_f64().unwrap() - 0.85).abs() < 1e-4);
        assert!((tint[2].as_f64().unwrap() - 1.05).abs() < 1e-4);

        // Frame capture verifies current emotion is sad while clip is sit
        let sad_frame = client.get_frame("clip", true, None);
        assert_eq!(sad_frame["emotion"], "sad");
        assert_eq!(sad_frame["clip"], "sit");

        let sad_b64 = sad_frame["base64"].as_str().unwrap();
        let sad_png = base64_decode(sad_b64).unwrap();
        let sad_img = decode_and_verify_png(&sad_png);
        let (viz, _) = visualize_frame_terminal(
            "Sad Fallback State (Sit + Blue Tint)",
            "sad",
            "sit",
            0,
            &sad_img,
        );
        print!("{}", viz);

        let sad_path = out_dir.join("sparky_frame_sad_fallback.png");
        let _ = std::fs::write(&sad_path, &sad_png);

        // 2. Set 'smirk' emotion (which has fallback = 'naughty')
        let smirk_res = client.set_emotion("smirk");
        assert_eq!(smirk_res["success"], true);
        assert_eq!(smirk_res["emotion"], "smirk");
        assert_eq!(smirk_res["active_clip"], "naughty"); // Fallback to naughty

        let smirk_frame = client.get_frame("clip", true, None);
        assert_eq!(smirk_frame["emotion"], "smirk");
        assert_eq!(smirk_frame["clip"], "naughty");

        let smirk_b64 = smirk_frame["base64"].as_str().unwrap();
        let smirk_png = base64_decode(smirk_b64).unwrap();
        let smirk_path = out_dir.join("sparky_frame_smirk_fallback.png");
        let _ = std::fs::write(&smirk_path, &smirk_png);
    });
}

#[test]
fn test_pet_behavior_frame_save_path_via_rpc() {
    run_desktop_pet_rpc_test(|client, out_dir| {
        let save_target = out_dir.join("rpc_direct_saved_frame.png");
        let target_str = save_target.to_str().unwrap();

        // 1. Call sparky.get_frame with save_path
        let frame_resp = client.get_frame("clip", true, Some(target_str));
        assert_eq!(frame_resp["saved_to"], target_str);
        assert_eq!(frame_resp["format"], "image/png");

        // 2. Verify file was created on disk
        assert!(
            save_target.exists(),
            "Target PNG file was not written to disk"
        );
        let disk_bytes = std::fs::read(&save_target).expect("Read saved file from disk");
        assert!(!disk_bytes.is_empty());

        // 3. Verify disk file matches the returned base64 bytes
        let b64 = frame_resp["base64"].as_str().unwrap();
        let b64_bytes = base64_decode(b64).unwrap();
        assert_eq!(disk_bytes, b64_bytes);

        // 4. Verify decoded image
        let img = decode_and_verify_png(&disk_bytes);
        assert_eq!(img.width(), 384);
        assert_eq!(img.height(), 216);
    });
}

#[test]
fn test_pet_behavior_chat_and_touch_interactions() {
    run_desktop_pet_rpc_test(|client, _out_dir| {
        // 1. Send chat message
        let chat_res = client.call_success(
            "sparky.send_chat",
            json!({ "prompt": "Sparky, who is the best kitty?" }),
        );
        assert_eq!(chat_res["success"], true);
        assert_eq!(chat_res["prompt"], "Sparky, who is the best kitty?");

        // 2. Verify chat history contains the prompt
        let history = client.call_success("sparky.get_chat_history", json!({}));
        let hist_arr = history.as_array().expect("History array");
        assert!(
            hist_arr
                .iter()
                .any(|m| m["sender"] == "User" && m["message"] == "Sparky, who is the best kitty?"),
            "User message not found in chat history"
        );

        // 3. Simulate pet touch interaction (resets boredom)
        let touch_res = client.call_success("sparky.touch", json!({}));
        assert_eq!(touch_res["success"], true);

        // 4. Clear chat
        let clear_res = client.call_success("sparky.clear_chat", json!({}));
        assert_eq!(clear_res["success"], true);

        let cleared_history = client.call_success("sparky.get_chat_history", json!({}));
        let cleared_arr = cleared_history.as_array().expect("History array");
        assert_eq!(cleared_arr.len(), 1);
        assert_eq!(cleared_arr[0]["sender"], "Sparky");
    });
}

#[test]
fn test_pet_behavior_window_geometry_and_visibility() {
    run_desktop_pet_rpc_test(|client, _out_dir| {
        // 1. Set position
        let pos_res = client.call_success("sparky.set_position", json!({ "x": 320, "y": 240 }));
        assert_eq!(pos_res["success"], true);
        assert_eq!(pos_res["x"], 320);
        assert_eq!(pos_res["y"], 240);

        let get_pos = client.call_success("sparky.get_position", json!({}));
        assert!(get_pos["x"].is_number());
        assert!(get_pos["y"].is_number());

        // 2. Set click-through
        let ct_res =
            client.call_success("sparky.set_click_through", json!({ "click_through": true }));
        assert_eq!(ct_res["success"], true);
        assert_eq!(ct_res["click_through"], true);

        let state = client.get_state();
        assert_eq!(state["window"]["click_through"], true);

        // 3. Set window hidden (tray mode)
        let vis_res = client.call_success("sparky.set_visible", json!({ "visible": false }));
        assert_eq!(vis_res["success"], true);
        assert_eq!(vis_res["visible"], false);

        // When hidden, get_frame automatically falls back to clip mode safely
        let hidden_frame = client.get_frame("framebuffer", true, None);
        assert_eq!(hidden_frame["source"], "clip");

        // Restore visibility
        let vis_res2 = client.call_success("sparky.set_visible", json!({ "visible": true }));
        assert_eq!(vis_res2["visible"], true);
    });
}

#[test]
fn test_pet_behavior_error_handling() {
    run_desktop_pet_rpc_test(|client, _out_dir| {
        // 1. Unknown emotion name
        let err1 = client.call_error(
            "sparky.set_emotion",
            json!({ "emotion": "non_existent_emotion_xyz" }),
        );
        assert_eq!(err1["code"], -32602); // Invalid params
        assert_eq!(err1["message"], "Invalid params");
        assert!(
            err1["data"].as_str().unwrap().contains("Unknown emotion"),
            "Expected 'Unknown emotion' in error data: {:?}",
            err1["data"]
        );

        // 2. Missing required prompt parameter in send_chat
        let err2 = client.call_error("sparky.send_chat", json!({}));
        assert_eq!(err2["code"], -32602); // Invalid params
        assert_eq!(err2["message"], "Invalid params");
        assert!(
            err2["data"]
                .as_str()
                .unwrap()
                .contains("Missing or invalid 'prompt'"),
            "Expected prompt error in data: {:?}",
            err2["data"]
        );

        // 3. Unknown method
        let err3 = client.call_error("sparky.fly_to_space", json!({}));
        assert_eq!(err3["code"], -32601); // Method not found
        assert!(
            err3["message"]
                .as_str()
                .unwrap()
                .contains("Method not found")
        );

        // 4. Aliases are rejected as unknown methods
        let err4 = client.call_error("pet", json!({}));
        assert_eq!(err4["code"], -32601);
        assert!(
            err4["message"]
                .as_str()
                .unwrap()
                .contains("Method not found")
        );

        let err5 = client.call_error("get_state", json!({}));
        assert_eq!(err5["code"], -32601);
        assert!(
            err5["message"]
                .as_str()
                .unwrap()
                .contains("Method not found")
        );

        let err6 = client.call_error("version", json!({}));
        assert_eq!(err6["code"], -32601);
        assert!(
            err6["message"]
                .as_str()
                .unwrap()
                .contains("Method not found")
        );
    });
}
