#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use glfw::Context;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use sysinfo::{ProcessesToUpdate, System, get_current_pid};

mod animation;
mod brain;
mod cli;
mod render;
mod rpc;
mod state;
mod tray;
mod ui;

use animation::{AnimationClip, FramePacer};
use brain::BrainEngine;
use clap::Parser;
use cli::Cli;
use render::{PetWindow, RenderQuad, ShaderProgram};
use rpc::{GlfwWakeNotifier, RpcServer};
use state::{EmotionRegistry, StateMachine};
use tray::TrayService;
use ui::ChatState;

const BASE_WIDTH: u32 = 288;
const BASE_HEIGHT: u32 = 162;
const CHAT_EXTRA_HEIGHT: u32 = 304;
const EXPANDED_HEIGHT: u32 = BASE_HEIGHT + CHAT_EXTRA_HEIGHT;

const EMBEDDED_VERT_SHADER: &str = include_str!("../shaders/quad.vert");
const EMBEDDED_FRAG_SHADER: &str = include_str!("../shaders/chroma.frag");

const EMBEDDED_CLIPS: &[(&str, &[u8])] = &[
    ("sit", include_bytes!("../assets/animations/sit.webp")),
    ("happy", include_bytes!("../assets/animations/happy.webp")),
    (
        "naughty",
        include_bytes!("../assets/animations/naughty.webp"),
    ),
    ("angry", include_bytes!("../assets/animations/angry.webp")),
    ("sad", include_bytes!("../assets/animations/sad.webp")),
    ("smirk", include_bytes!("../assets/animations/smirk.webp")),
    (
        "curious",
        include_bytes!("../assets/animations/curious.webp"),
    ),
    ("tired", include_bytes!("../assets/animations/tired.webp")),
];

fn main() -> Result<()> {
    #[cfg(target_os = "windows")]
    unsafe {
        extern "system" {
            fn AttachConsole(dwProcessId: u32) -> i32;
        }
        const ATTACH_PARENT_PROCESS: u32 = 0xFFFFFFFF;
        AttachConsole(ATTACH_PARENT_PROCESS);
    }

    let cli = Cli::parse();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stdout)
        .init();
    log::info!("Starting Sparky the Kitten Desktop Pet (Extensible Emotion Edition)...");

    let is_profile_run = cli.profile_run;
    if is_profile_run {
        log::info!("Profile run mode enabled (--profile-run: 10s benchmark session)");
    }

    // 1. Initialize Window & OpenGL
    log::info!("Initializing PetWindow (target base resolution: {BASE_WIDTH}x{BASE_HEIGHT})...");
    let mut pet_window = PetWindow::new(BASE_WIDTH, BASE_HEIGHT, "Sparky")?;
    let quad = RenderQuad::new();
    let shader = if std::path::Path::new("shaders/quad.vert").exists()
        && std::path::Path::new("shaders/chroma.frag").exists()
    {
        ShaderProgram::new("shaders/quad.vert", "shaders/chroma.frag")?
    } else {
        log::info!("Shaders not found on disk; using embedded shaders");
        ShaderProgram::from_source(EMBEDDED_VERT_SHADER, EMBEDDED_FRAG_SHADER)?
    };

    // 2. Load Declarative Emotion Registry (falls back to embedded if file not on disk)
    let emotion_registry = Arc::new(EmotionRegistry::load_from_toml(
        &cli.emotions_config,
        &cli.animations_dir,
    )?);

    // 3. Load Available Animations dynamically from disk, then backfill with embedded clips
    log::info!("Loading animation clips from {:?}...", cli.animations_dir);
    let mut clips: HashMap<String, AnimationClip> = HashMap::new();
    if let Ok(entries) = std::fs::read_dir(&cli.animations_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("webp") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    match AnimationClip::load_from_webp(stem, &path) {
                        Ok(clip) => {
                            clips.insert(stem.to_string(), clip);
                        }
                        Err(e) => {
                            log::warn!("Failed to load animation clip from {:?}: {}", path, e);
                        }
                    }
                }
            }
        }
    }

    // Backfill missing animation clips from embedded assets
    for (name, bytes) in EMBEDDED_CLIPS {
        if !clips.contains_key(*name) {
            match AnimationClip::load_from_bytes(name, bytes) {
                Ok(clip) => {
                    clips.insert(name.to_string(), clip);
                }
                Err(e) => {
                    log::warn!("Failed to load embedded animation clip '{}': {}", name, e);
                }
            }
        }
    }

    if clips.is_empty() {
        log::warn!(
            "No WebP animations loaded from {:?} or embedded assets!",
            cli.animations_dir
        );
    } else {
        log::info!(
            "Loaded {} animation clips: {:?}",
            clips.len(),
            clips.keys().collect::<Vec<_>>()
        );
    }

    // 4. Initialize Tray, Brain & State Machine
    log::info!("Initializing system tray service...");
    let tray_service = match TrayService::from_model_path(&emotion_registry, &cli.model_path) {
        Ok(t) => {
            log::info!(
                "System tray service initialized (GGUF model downloaded: {}).",
                t.download_gguf_item.is_checked()
            );
            Some(t)
        }
        Err(e) => {
            log::warn!("Failed to initialize tray service: {}", e);
            None
        }
    };

    if cli.click_through {
        pet_window.set_click_through(true);
        if let Some(ref tray) = tray_service {
            tray.click_through_item.set_checked(true);
        }
    }

    let brain = BrainEngine::spawn_with_backend(
        cli.model_path.clone(),
        cli.brain_backend,
        emotion_registry.clone(),
    );
    let mut state_machine = StateMachine::new(&emotion_registry);
    if let Some(ref initial_emotion) = cli.initial_emotion {
        if emotion_registry.get(initial_emotion).is_some() {
            state_machine.transition_to(initial_emotion, &emotion_registry);
        } else {
            log::warn!(
                "Initial emotion '{}' not found in registry; defaulting to 'sit'",
                initial_emotion
            );
        }
    }

    let mut frame_pacer = FramePacer::new(cli.fps);
    log::info!("Frame pacer configured for {:.1} FPS", cli.fps);
    let mut chat_state = ChatState::new();

    // 5. Initialize JSON-RPC 2.0 Server
    let rpc_server = if !cli.no_rpc {
        let thread_safe_glfw = glfw::ThreadSafeGlfw::from(&mut pet_window.glfw);
        match RpcServer::start(
            &cli.rpc_addr,
            emotion_registry.clone(),
            Box::new(GlfwWakeNotifier::new(thread_safe_glfw)),
        ) {
            Ok(server) => {
                log::info!(
                    "JSON-RPC server running on http://{} and tcp://{}",
                    server.local_addr(),
                    server.local_addr()
                );
                Some(server)
            }
            Err(e) => {
                log::warn!("Failed to start JSON-RPC server on {}: {}", cli.rpc_addr, e);
                None
            }
        }
    } else {
        None
    };

    // 6. Initialize Egui for Chat overlay
    log::info!("Initializing egui overlay painter and input state...");
    let mut painter = egui_gl_glfw::Painter::new(&pet_window.window);
    let egui_ctx = egui::Context::default();
    let (fb_w, _) = pet_window.window.get_framebuffer_size();
    let (win_w, win_h) = pet_window.window.get_size();
    let pixels_per_point = fb_w as f32 / win_w as f32;
    let mut egui_input_state = egui_gl_glfw::EguiInputState::new(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(win_w as f32, win_h as f32),
            )),
            ..Default::default()
        },
        pixels_per_point,
    );

    let mut is_visible = true;
    let mut was_chat_visible = false;

    // Profiling monitor setup
    let mut sys = System::new();
    let current_pid = get_current_pid().ok();
    let profile_start = Instant::now();
    let mut last_profile_sample = Instant::now();
    let mut frame_count: u64 = 0;
    let mut cpu_samples = Vec::new();
    let mut ram_samples = Vec::new();

    // Shader Uniforms
    log::info!(
        "Configuring chroma-keying shader uniforms (threshold: 0.32, softness: 0.12, despill: 0.85, border_margin: 0.015, border_feather: 0.005)..."
    );
    shader.use_program();
    shader.set_int("u_Texture", 0);
    shader.set_vec3("u_KeyColorRGB", 0.0, 1.0, 0.0);
    shader.set_float("u_Threshold", 0.32);
    shader.set_float("u_Softness", 0.12);
    shader.set_float("u_Despill", 0.85);
    shader.set_vec2("u_BorderMargin", 0.015, 0.015);
    shader.set_vec2("u_BorderFeather", 0.005, 0.005);

    let (download_tx, download_rx) = crossbeam_channel::unbounded::<Result<PathBuf, String>>();
    let mut is_downloading_gguf = false;

    log::info!("Application loading complete! Entering main event loop.");

    // Main Event Loop
    while !pet_window.window.should_close() {
        // Sleep until next frame to preserve CPU; if actively dragging, use short timeout for smooth tracking
        let timeout = if pet_window.is_dragging {
            0.005 // 5ms for smooth dragging
        } else if is_visible {
            frame_pacer.time_until_next_frame().as_secs_f64()
        } else {
            0.1 // Deep sleep when hidden in tray
        };

        pet_window.glfw.wait_events_timeout(timeout);

        // Pump system tray event loop for responsiveness
        if let Some(ref tray) = tray_service {
            tray.pump_events();
        }

        // Process JSON-RPC Commands
        if let Some(ref server) = rpc_server {
            while let Some(cmd) = server.try_recv_command() {
                rpc::handle_rpc_command(
                    cmd,
                    &mut state_machine,
                    &emotion_registry,
                    &mut pet_window,
                    &mut chat_state,
                    &brain,
                    &tray_service,
                    &mut frame_pacer,
                    &mut is_visible,
                    &clips,
                );
            }
        }

        // Adjust window size dynamically when chat is toggled
        if chat_state.is_visible != was_chat_visible {
            let (wx, wy) = pet_window.get_pos();
            if chat_state.is_visible {
                pet_window
                    .window
                    .set_size(BASE_WIDTH as i32, EXPANDED_HEIGHT as i32);
                pet_window.set_pos(wx, wy - CHAT_EXTRA_HEIGHT as i32);
                pet_window.height = EXPANDED_HEIGHT;
                chat_state.focus_input = true;
            } else {
                pet_window
                    .window
                    .set_size(BASE_WIDTH as i32, BASE_HEIGHT as i32);
                pet_window.set_pos(wx, wy + CHAT_EXTRA_HEIGHT as i32);
                pet_window.height = BASE_HEIGHT;
            }
            let (nfb_w, nfb_h) = pet_window.window.get_framebuffer_size();
            let (nwin_w, nwin_h) = pet_window.window.get_size();
            let ppp = nfb_w as f32 / nwin_w as f32;
            egui_input_state.pixels_per_point = ppp;
            painter.set_size(nfb_w as u32, nfb_h as u32);
            egui_input_state.input.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(nwin_w as f32, nwin_h as f32),
            ));
            was_chat_visible = chat_state.is_visible;
        }

        // Process Window Events
        let events: Vec<_> = glfw::flush_messages(&pet_window.events).collect();
        for (_, event) in events {
            egui_gl_glfw::handle_event(event.clone(), &mut egui_input_state);

            let pointer_captured_by_egui =
                !pet_window.is_dragging && chat_state.is_visible && egui_ctx.wants_pointer_input();
            if !pointer_captured_by_egui {
                pet_window.handle_mouse_drag(&event);
                state_machine.touch_interaction();
            }

            match event {
                glfw::WindowEvent::MouseButton(
                    glfw::MouseButton::Button2,
                    glfw::Action::Press,
                    _,
                ) => {
                    chat_state.is_visible = !chat_state.is_visible;
                }
                glfw::WindowEvent::FramebufferSize(w, h) => {
                    let (win_w, win_h) = pet_window.window.get_size();
                    if win_w > 0 {
                        egui_input_state.pixels_per_point = w as f32 / win_w as f32;
                    }
                    painter.set_size(w as u32, h as u32);
                    egui_input_state.input.screen_rect = Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(win_w as f32, win_h as f32),
                    ));
                }
                glfw::WindowEvent::Key(
                    glfw::Key::KpEnter,
                    _,
                    glfw::Action::Press | glfw::Action::Repeat,
                    _,
                ) => {
                    egui_input_state.input.events.push(egui::Event::Key {
                        key: egui::Key::Enter,
                        physical_key: None,
                        pressed: true,
                        modifiers: egui_input_state.modifiers,
                        repeat: false,
                    });
                }
                glfw::WindowEvent::Key(glfw::Key::KpEnter, _, glfw::Action::Release, _) => {
                    egui_input_state.input.events.push(egui::Event::Key {
                        key: egui::Key::Enter,
                        physical_key: None,
                        pressed: false,
                        modifiers: egui_input_state.modifiers,
                        repeat: false,
                    });
                }
                _ => {}
            }
        }

        // Process GGUF model download results
        while let Ok(result) = download_rx.try_recv() {
            is_downloading_gguf = false;
            match result {
                Ok(path) => {
                    log::info!("GGUF model downloaded successfully to {:?}", path);
                    if let Some(ref tray) = tray_service {
                        tray.download_gguf_item.set_text("Download GGUF Model");
                        tray.download_gguf_item.set_checked(true);
                        tray.download_gguf_item.set_enabled(true);
                    }
                    if chat_state.is_visible {
                        chat_state.messages.push((
                            "Sparky".to_string(),
                            "Meow! Gemma 3 GGUF model downloaded successfully!".to_string(),
                        ));
                    }
                }
                Err(err) => {
                    log::error!("GGUF model download failed: {}", err);
                    if let Some(ref tray) = tray_service {
                        tray.download_gguf_item.set_text("Download GGUF Model");
                        tray.download_gguf_item.set_checked(false);
                        tray.download_gguf_item.set_enabled(true);
                    }
                    if chat_state.is_visible {
                        chat_state.messages.push((
                            "System".to_string(),
                            format!("Failed to download GGUF model: {}", err),
                        ));
                    }
                }
            }
        }

        // Process Tray Menu Events
        if let Some(ref tray) = tray_service
            && let Ok(menu_event) = muda::MenuEvent::receiver().try_recv()
        {
            if menu_event.id == tray.show_hide_item.id() {
                is_visible = tray.show_hide_item.is_checked();
                if is_visible {
                    pet_window.window.show();
                } else {
                    pet_window.window.hide();
                }
            } else if menu_event.id == tray.chat_item.id() {
                chat_state.is_visible = !chat_state.is_visible;
            } else if menu_event.id == tray.click_through_item.id() {
                let click_through = tray.click_through_item.is_checked();
                pet_window.set_click_through(click_through);
            } else if menu_event.id == tray.download_gguf_item.id() {
                let already_downloaded = brain::is_gguf_downloaded(&cli.model_path);
                if already_downloaded {
                    log::info!("GGUF model is already downloaded.");
                    tray.download_gguf_item.set_checked(true);
                } else if is_downloading_gguf {
                    log::info!("GGUF model download is already in progress...");
                    tray.download_gguf_item.set_checked(false);
                } else {
                    is_downloading_gguf = true;
                    tray.download_gguf_item
                        .set_text("Downloading GGUF Model...");
                    tray.download_gguf_item.set_checked(false);
                    tray.download_gguf_item.set_enabled(false);

                    let dest_path = brain::resolve_gguf_download_path(&cli.model_path);
                    let tx = download_tx.clone();
                    std::thread::Builder::new()
                        .name("gguf-downloader".to_string())
                        .spawn(move || {
                            log::info!(
                                "Starting background download of GGUF model to {:?}",
                                dest_path
                            );
                            let res =
                                brain::download_gguf_model(&dest_path).map_err(|e| e.to_string());
                            let _ = tx.send(res);
                        })
                        .expect("Failed to spawn downloader thread");

                    if chat_state.is_visible {
                        chat_state.messages.push((
                            "System".to_string(),
                            "Downloading Gemma 3 270M GGUF model in the background...".to_string(),
                        ));
                    }
                }
            } else if menu_event.id == tray.quit_item.id() {
                pet_window.window.set_should_close(true);
            } else if let Some(emotion_id) = tray.emotion_menu_items.get(&menu_event.id) {
                state_machine.push_emotion(emotion_id, &emotion_registry);
            }
        }

        // Process Brain Responses
        while let Some(resp) = brain.try_recv_response() {
            match resp {
                brain::BrainResponse::Token(token) => {
                    if let Some(last) = chat_state.messages.last_mut() {
                        if last.0 == "Sparky" {
                            last.1.push_str(&token);
                        }
                    }
                }
                brain::BrainResponse::Complete {
                    full_text: _,
                    action_tag,
                } => {
                    chat_state.is_generating = false;
                    chat_state.focus_input = true;
                    if let Some(tag) = action_tag {
                        state_machine.push_emotion(&tag, &emotion_registry);
                    }
                }
                brain::BrainResponse::Error(err) => {
                    chat_state.is_generating = false;
                    chat_state.focus_input = true;
                    chat_state
                        .messages
                        .push(("System".to_string(), format!("Error: {}", err)));
                }
            }
        }

        state_machine.tick(&emotion_registry);

        // Render Frame
        if is_visible {
            let active_clip = &state_machine.active_clip_name;
            let mut should_render = false;

            if let Some(clip) = clips.get(active_clip) {
                let (advanced, completed) = frame_pacer.advance_frame_cycle(clip.frames.len());
                if frame_count == 0 || advanced {
                    let frame_idx = frame_pacer.current_index();
                    if let Some(frame) = clip.frames.get(frame_idx) {
                        quad.update_texture(frame.width, frame.height, &frame.rgb_data);
                        should_render = true;
                    }
                }
                if completed && state_machine.on_animation_completed(&emotion_registry) {
                    chat_state.current_emotion_id = state_machine.current_emotion.clone();
                    frame_pacer.reset();
                }
            }

            if should_render || chat_state.is_visible {
                unsafe {
                    gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT);
                    gl::Disable(gl::SCISSOR_TEST);
                    gl::Enable(gl::BLEND);
                    gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                }

                // If chat is open, kitten quad occupies the bottom 162 px
                let (fb_w, fb_h) = pet_window.window.get_framebuffer_size();
                let kitten_fb_h =
                    ((BASE_HEIGHT as f32 / pet_window.height as f32) * fb_h as f32) as i32;
                unsafe {
                    gl::Viewport(0, 0, fb_w, kitten_fb_h);
                }

                shader.use_program();
                shader.set_vec3(
                    "u_EmotionTint",
                    state_machine.active_tint[0],
                    state_machine.active_tint[1],
                    state_machine.active_tint[2],
                );
                quad.draw();

                // Render Egui Chat UI if visible
                if chat_state.is_visible {
                    let (curr_fb_w, curr_fb_h) = pet_window.window.get_framebuffer_size();
                    let (curr_win_w, curr_win_h) = pet_window.window.get_size();
                    if curr_win_w > 0 && curr_win_h > 0 {
                        let ppp = curr_fb_w as f32 / curr_win_w as f32;
                        egui_input_state.pixels_per_point = ppp;
                        painter.set_size(curr_fb_w as u32, curr_fb_h as u32);
                        egui_input_state.input.screen_rect = Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(curr_win_w as f32, curr_win_h as f32),
                        ));
                    }
                    let raw_input = egui_input_state.input.take();
                    let mut triggered_emotion: Option<String> = None;

                    let full_output = egui_ctx.run(raw_input, |ctx| {
                        chat_state.draw(
                            ctx,
                            &emotion_registry,
                            |prompt| {
                                brain.send_chat(prompt);
                            },
                            |emotion_id| {
                                triggered_emotion = Some(emotion_id.to_string());
                            },
                        );
                    });

                    if let Some(emotion_id) = triggered_emotion {
                        state_machine.push_emotion(&emotion_id, &emotion_registry);
                    }

                    unsafe {
                        gl::Viewport(0, 0, fb_w, fb_h);
                    }

                    let ppp = egui_input_state.pixels_per_point;
                    let shapes = egui_ctx.tessellate(full_output.shapes, ppp);
                    painter.paint_and_update_textures(ppp, &shapes, &full_output.textures_delta);

                    unsafe {
                        gl::Disable(gl::SCISSOR_TEST);
                        gl::Enable(gl::BLEND);
                        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                    }
                }

                pet_window.window.swap_buffers();
                frame_count += 1;
            }
        }

        // Periodic resource profiling sampling
        if is_profile_run && last_profile_sample.elapsed().as_secs_f32() >= 1.0 {
            if let Some(pid) = current_pid {
                sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
                if let Some(proc) = sys.process(pid) {
                    let cpu = proc.cpu_usage();
                    let rss_mb = proc.memory() as f64 / (1024.0 * 1024.0);
                    let fps = frame_count as f64 / profile_start.elapsed().as_secs_f64();
                    cpu_samples.push(cpu);
                    ram_samples.push(rss_mb);
                    println!(
                        "[PROFILE] Elapsed: {:.1}s | Frames: {} | FPS: {:.1} | CPU: {:.2}% | RSS: {:.2} MB",
                        profile_start.elapsed().as_secs_f64(),
                        frame_count,
                        fps,
                        cpu,
                        rss_mb
                    );
                }
            }
            last_profile_sample = Instant::now();

            // End profile run after 10 seconds if automated
            if is_profile_run && profile_start.elapsed().as_secs() >= 10 {
                let avg_cpu = if !cpu_samples.is_empty() {
                    cpu_samples.iter().sum::<f32>() / cpu_samples.len() as f32
                } else {
                    0.0
                };
                let peak_ram = ram_samples.iter().cloned().fold(0.0f64, f64::max);
                let avg_fps = frame_count as f64 / profile_start.elapsed().as_secs_f64();

                println!("\n====================================================================");
                println!("SPARKY PROFILE RUN SUMMARY");
                println!("====================================================================");
                println!(
                    "Elapsed Time:     {:.2}s",
                    profile_start.elapsed().as_secs_f64()
                );
                println!("Total Frames:     {}", frame_count);
                println!("Average FPS:      {:.2} FPS", avg_fps);
                println!("Average CPU:      {:.2}%", avg_cpu);
                println!("Peak RSS Memory:  {:.2} MB", peak_ram);
                println!("====================================================================\n");
                break;
            }
        }
    }

    if let Some(ref server) = rpc_server {
        server.shutdown();
    }

    brain.shutdown();
    Ok(())
}
