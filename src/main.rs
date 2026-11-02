use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use anyhow::Result;
use glfw::Context;
use sysinfo::{get_current_pid, ProcessesToUpdate, System};

mod animation;
mod brain;
mod render;
mod state;
mod tray;
mod ui;

use animation::{AnimationClip, FramePacer};
use brain::BrainEngine;
use render::{PetWindow, RenderQuad, ShaderProgram};
use state::{EmotionRegistry, StateMachine};
use tray::TrayService;
use ui::ChatState;

const BASE_WIDTH: u32 = 384;
const BASE_HEIGHT: u32 = 216;
const CHAT_EXTRA_HEIGHT: u32 = 304;
const EXPANDED_HEIGHT: u32 = BASE_HEIGHT + CHAT_EXTRA_HEIGHT;

fn main() -> Result<()> {
    env_logger::init();
    log::info!("Starting Sparky the Kitten Desktop Pet (Extensible Emotion Edition)...");

    let args: Vec<String> = std::env::args().collect();
    let is_profile_run = args.iter().any(|a| a == "--profile-run");

    // 1. Initialize Window & OpenGL
    let mut pet_window = PetWindow::new(BASE_WIDTH, BASE_HEIGHT, "Sparky")?;
    let quad = RenderQuad::new();
    let shader = ShaderProgram::new("shaders/quad.vert", "shaders/chroma.frag")?;

    // 2. Load Declarative Emotion Registry
    let emotion_registry = Arc::new(EmotionRegistry::load_from_toml(
        "assets/emotions.toml",
        "assets/animations",
    )?);

    // 3. Load Available Animations dynamically
    let mut clips: HashMap<String, AnimationClip> = HashMap::new();
    if let Ok(entries) = std::fs::read_dir("assets/animations") {
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

    if clips.is_empty() {
        log::warn!("No WebP animations loaded from assets/animations! Check transcoding.");
    } else {
        log::info!(
            "Loaded {} animation clips: {:?}",
            clips.len(),
            clips.keys().collect::<Vec<_>>()
        );
    }

    // 4. Initialize Tray, Brain & State Machine
    let tray_service = match TrayService::new(&emotion_registry) {
        Ok(t) => {
            log::info!("System tray service initialized.");
            Some(t)
        }
        Err(e) => {
            log::warn!("Failed to initialize tray service: {}", e);
            None
        }
    };

    let brain = BrainEngine::spawn("models/gemma-3-270m".into(), emotion_registry.clone());
    let mut state_machine = StateMachine::new(&emotion_registry);
    let mut frame_pacer = FramePacer::new(24.0);
    let mut chat_state = ChatState::new();

    // 5. Initialize Egui for Chat overlay
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
    shader.use_program();
    shader.set_int("u_Texture", 0);
    shader.set_vec3("u_KeyColorRGB", 0.0, 1.0, 0.0);
    shader.set_float("u_Threshold", 0.32);
    shader.set_float("u_Softness", 0.12);
    shader.set_float("u_Despill", 0.85);

    // Main Event Loop
    while !pet_window.window.should_close() {
        // Sleep until next frame to preserve CPU
        let timeout = if is_visible {
            frame_pacer.time_until_next_frame().as_secs_f64()
        } else {
            0.1 // Deep sleep when hidden in tray
        };

        pet_window.glfw.wait_events_timeout(timeout);

        // Pump system tray event loop for responsiveness
        if let Some(ref tray) = tray_service {
            tray.pump_events();
        }

        // Adjust window size dynamically when chat is toggled
        if chat_state.is_visible != was_chat_visible {
            let (wx, wy) = pet_window.window.get_pos();
            if chat_state.is_visible {
                pet_window
                    .window
                    .set_size(BASE_WIDTH as i32, EXPANDED_HEIGHT as i32);
                pet_window.window.set_pos(wx, wy - CHAT_EXTRA_HEIGHT as i32);
                pet_window.height = EXPANDED_HEIGHT;
            } else {
                pet_window
                    .window
                    .set_size(BASE_WIDTH as i32, BASE_HEIGHT as i32);
                pet_window.window.set_pos(wx, wy + CHAT_EXTRA_HEIGHT as i32);
                pet_window.height = BASE_HEIGHT;
            }
            let (nfb_w, _) = pet_window.window.get_framebuffer_size();
            let (nwin_w, nwin_h) = pet_window.window.get_size();
            let ppp = nfb_w as f32 / nwin_w as f32;
            egui_input_state.pixels_per_point = ppp;
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

            let pointer_captured_by_egui = chat_state.is_visible && egui_ctx.wants_pointer_input();
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
                glfw::WindowEvent::FramebufferSize(w, _) => {
                    let (win_w, _) = pet_window.window.get_size();
                    if win_w > 0 {
                        egui_input_state.pixels_per_point = w as f32 / win_w as f32;
                    }
                }
                _ => {}
            }
        }

        // Process Tray Menu Events
        if let Some(ref tray) = tray_service {
            if let Ok(menu_event) = muda::MenuEvent::receiver().try_recv() {
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
                } else if menu_event.id == tray.quit_item.id() {
                    pet_window.window.set_should_close(true);
                } else if let Some(emotion_id) = tray.emotion_menu_items.get(&menu_event.id) {
                    state_machine.transition_to(emotion_id, &emotion_registry);
                    chat_state.current_emotion_id = emotion_id.clone();
                    frame_pacer.reset();
                }
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
                    if let Some(tag) = action_tag {
                        state_machine.transition_to(&tag, &emotion_registry);
                        chat_state.current_emotion_id = tag;
                        frame_pacer.reset();
                    }
                }
                brain::BrainResponse::Error(err) => {
                    chat_state.is_generating = false;
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
                if frame_count == 0 || frame_pacer.advance_frame(clip.frames.len()) {
                    let frame_idx = frame_pacer.current_index();
                    if let Some(frame) = clip.frames.get(frame_idx) {
                        quad.update_texture(frame.width, frame.height, &frame.rgb_data);
                        should_render = true;
                    }
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

                // If chat is open, kitten quad occupies the bottom 216 px
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
                        state_machine.transition_to(&emotion_id, &emotion_registry);
                        chat_state.current_emotion_id = emotion_id;
                        frame_pacer.reset();
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
                println!("Elapsed Time:     {:.2}s", profile_start.elapsed().as_secs_f64());
                println!("Total Frames:     {}", frame_count);
                println!("Average FPS:      {:.2} FPS", avg_fps);
                println!("Average CPU:      {:.2}%", avg_cpu);
                println!("Peak RSS Memory:  {:.2} MB", peak_ram);
                println!("====================================================================\n");
                break;
            }
        }
    }

    brain.shutdown();
    Ok(())
}
