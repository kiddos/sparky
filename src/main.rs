#![windows_subsystem = "windows"]

mod app;
mod brain;
mod chat;
mod cli;
mod shader;
mod tray;
mod util;
mod video;

use std::error::Error;
use std::time::{Duration, Instant};

use glfw::{Action, Context, Key, WindowEvent};
use log::info;
use tray_icon::menu::MenuEvent;

use crate::app::App;
use crate::brain::{BrainConfig, BrainEngineMode, BrainEvent, BrainHandle, PersonalityArchetype};
use crate::chat::{ChatMessage, ChatWindow, MessageSender};
use crate::cli::parse_args;
use crate::tray::{QUIT_MENU_ID, SHOW_MENU_ID, TOGGLE_CHAT_MENU_ID, create_tray_icon};
use crate::util::init_log;
use crate::video::{find_videos_dir, load_video_animations, probe_video_specs};

const DEFAULT_FRAME_WIDTH: u32 = 128;
const DEFAULT_FRAME_HEIGHT: u32 = 128;
const DEFAULT_ANIMATION_SPEED: Duration = Duration::from_millis(42);

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn XInitThreads() -> std::ffi::c_int;
}

pub fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(target_os = "linux")]
    unsafe {
        XInitThreads();
    }

    dotenvy::dotenv().ok();
    init_log();
    info!("Starting Sparky with Gemma 4 Intelligence Engine...");

    let cli = parse_args()?;

    #[cfg(target_os = "linux")]
    {
        if let Err(e) = gtk::init() {
            eprintln!("Failed to initialize GTK for tray icon: {}", e);
            std::process::exit(1);
        }
    }

    let mut glfw = glfw::init(glfw::log_errors).unwrap();

    glfw.window_hint(glfw::WindowHint::Resizable(false));
    glfw.window_hint(glfw::WindowHint::Decorated(false));
    glfw.window_hint(glfw::WindowHint::TransparentFramebuffer(true));
    glfw.window_hint(glfw::WindowHint::Floating(true));
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    // Determine playback specs before window creation
    let video_dir_opt = find_videos_dir(cli.video_dir.as_deref());
    let (frame_width, frame_height, animation_speed, black_threshold) =
        if let Some(ref v_dir) = video_dir_opt {
            let videos = load_video_animations(v_dir);
            if let Some(first_video) = videos.first() {
                let (v_w, v_h, fps) = probe_video_specs(&first_video.path);
                let scale = cli.scale.unwrap_or(1.0);
                let w = cli
                    .width
                    .unwrap_or_else(|| (v_w as f32 * scale).round() as u32);
                let h = cli
                    .height
                    .unwrap_or_else(|| (v_h as f32 * scale).round() as u32);
                let target_fps = cli.fps.unwrap_or(fps);
                let speed = Duration::from_secs_f64(1.0 / target_fps.max(1.0));
                let thresh = cli.threshold.unwrap_or(0.08);
                info!(
                    "Video mode: target resolution {}x{}, {:.1} FPS, black threshold: {}",
                    w, h, target_fps, thresh
                );
                (w, h, speed, thresh)
            } else {
                (
                    DEFAULT_FRAME_WIDTH,
                    DEFAULT_FRAME_HEIGHT,
                    DEFAULT_ANIMATION_SPEED,
                    0.0,
                )
            }
        } else {
            (
                DEFAULT_FRAME_WIDTH,
                DEFAULT_FRAME_HEIGHT,
                DEFAULT_ANIMATION_SPEED,
                0.0,
            )
        };

    let (mut window, events) = glfw
        .create_window(
            frame_width,
            frame_height,
            "Sparky",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create GLFW window.");

    window.make_current();
    window.set_all_polling(true);

    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    // Initial position: bottom-right corner of primary monitor
    let (monitor_x, monitor_y, monitor_width, monitor_height) =
        glfw.with_primary_monitor(|_, m| {
            m.map(|m| {
                let video_mode = m.get_video_mode().unwrap();
                let (x, y) = m.get_pos();
                (x, y, video_mode.width, video_mode.height)
            })
            .unwrap_or((0, 0, 1920, 1080))
        });

    let initial_x = monitor_x + monitor_width as i32 - frame_width as i32;
    let initial_y = monitor_y + monitor_height as i32 - frame_height as i32 - 30;
    window.set_pos(initial_x, initial_y);

    let mut app = App::new(window, frame_width, frame_height, black_threshold);
    app.window_pos = (initial_x, initial_y);
    app.init_opengl();
    app.load_animations(cli.video_dir.as_deref())?;
    app.render();

    // Initialize Gemma AI Intelligence Engine if enabled
    if !cli.disable_brain {
        let api_key = std::env::var("GEMMA_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .ok();
        let personality: PersonalityArchetype = cli
            .personality
            .as_deref()
            .unwrap_or("chaos_gremlin")
            .parse()
            .unwrap_or_default();
        let interval = Duration::from_secs(cli.brain_interval.unwrap_or(20));

        let engine_mode = match cli.gemma_engine.as_deref() {
            Some("local") | Some("gemma-3") | Some("gemma-3-270m") => BrainEngineMode::Local,
            Some("api") | Some("gemma-4") => BrainEngineMode::Api,
            _ => {
                if let Some(ref m) = cli.gemma_model {
                    if m.contains("gemma-3") || m.contains("270m") {
                        BrainEngineMode::Local
                    } else {
                        BrainEngineMode::Api
                    }
                } else if api_key.is_some() {
                    BrainEngineMode::Api
                } else {
                    BrainEngineMode::Local
                }
            }
        };

        let chosen_model = cli.gemma_model.unwrap_or_else(|| match engine_mode {
            BrainEngineMode::Local => "gemma-3-270m".to_string(),
            BrainEngineMode::Api => "gemma-4".to_string(),
        });

        info!(
            "Configuring AI Brain: engine={:?}, model='{}', personality='{}'",
            engine_mode,
            chosen_model,
            personality.name()
        );

        let brain_config = BrainConfig {
            engine_mode,
            personality,
            interval,
            api_key,
            api_base: cli.gemma_api_base,
            api_model: chosen_model,
            local_model_dir: cli.model_path,
        };

        let brain = BrainHandle::spawn(brain_config);
        brain.send_event(BrainEvent::Started);
        app.set_brain(brain);
    }

    let personality_name = cli.personality.as_deref().unwrap_or("chaos_gremlin");
    let chat_window = ChatWindow::new(personality_name);
    chat_window.add_message(&ChatMessage {
        sender: MessageSender::System,
        text: "Sparky Chat ready! Right-click Sparky, press 'C', or use the tray menu to toggle.".to_string(),
        emotion: None,
    });
    chat_window.add_sparky_message("happy1", "Hey! I'm Sparky! What are we working on today?");
    app.set_chat_window(chat_window);
    if cli.chat {
        app.set_chat_visible(true, monitor_height as i32);
    }

    let mut tray_icon = create_tray_icon();
    let tray_receiver = MenuEvent::receiver();
    let mut last_update = Instant::now();

    while !app.window.should_close() {
        #[cfg(target_os = "linux")]
        {
            while gtk::events_pending() {
                gtk::main_iteration();
            }
        }

        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                WindowEvent::Close => {
                    info!("sparky exit");
                    app.window.set_should_close(true);
                }
                WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    info!("sparky exit");
                    app.window.set_should_close(true);
                }
                WindowEvent::Key(Key::C, _, Action::Press, _)
                | WindowEvent::Key(Key::T, _, Action::Press, _) => {
                    app.toggle_chat(monitor_height as i32);
                }
                WindowEvent::MouseButton(glfw::MouseButtonRight, Action::Press, _) => {
                    let (x, y) = app.window.get_cursor_pos();
                    if app.is_pixel_opaque(x, y) {
                        app.toggle_chat(monitor_height as i32);
                    }
                }
                WindowEvent::MouseButton(glfw::MouseButtonLeft, Action::Press, _) => {
                    let (x, y) = app.window.get_cursor_pos();
                    if app.is_pixel_opaque(x, y) {
                        app.start_dragging(x, y);
                        if let Some(ref brain) = app.brain {
                            brain.send_event(BrainEvent::Clicked { x, y });
                        }
                    }
                }
                WindowEvent::MouseButton(glfw::MouseButtonLeft, Action::Release, _) => {
                    if app.is_dragging {
                        app.stop_dragging();
                        if let Some(ref brain) = app.brain {
                            brain.send_event(BrainEvent::DragEnded);
                        }
                    }
                }
                WindowEvent::CursorPos(..) if app.is_dragging => {
                    if let Some((dx, dy)) = app.update_drag(monitor_height as i32) {
                        if let Some(ref brain) = app.brain {
                            brain.send_event(BrainEvent::Dragged { dx, dy });
                        }
                    }
                }
                _ => {}
            }
        }

        // Tray events
        if let Ok(event) = tray_receiver.try_recv() {
            if event.id == QUIT_MENU_ID {
                info!("sparky exit from tray");
                app.window.set_should_close(true);
            } else if event.id == SHOW_MENU_ID {
                app.window.show();
            } else if event.id == TOGGLE_CHAT_MENU_ID {
                app.toggle_chat(monitor_height as i32);
            }
        }

        // Handle outgoing user messages typed into the chat window
        if let Some(ref mut chat) = app.chat_window {
            while let Some(user_msg) = chat.try_recv_user_message() {
                info!("[Sparky Chat] User asked: '{}'", user_msg);
                chat.add_user_message(&user_msg);
                if let Some(ref brain) = app.brain {
                    chat.set_status("[thinking...]");
                    brain.send_chat(user_msg);
                } else {
                    chat.add_sparky_message("happy1", "Chirp! (Brain is offline, but I see you!)");
                    chat.set_status("[idle]");
                }
            }
        }

        // Poll Gemma 4 Brain for thoughts and emotion changes
        if let Some((emotion, dialogue)) = app.poll_brain() {
            if let Some(ref mut tray) = tray_icon {
                let tooltip = format!("Sparky [{}]: \"{}\"", emotion, dialogue);
                let _ = tray.set_tooltip(Some(tooltip));
            }
        }

        let elapsed = last_update.elapsed();
        if elapsed >= animation_speed {
            app.update();
            last_update = Instant::now();
            app.render();
        }

        if app.is_dragging {
            // Safety check: if mouse button was released outside window or release event was missed
            if app.window.get_mouse_button(glfw::MouseButtonLeft) == Action::Release {
                app.stop_dragging();
                if let Some(ref brain) = app.brain {
                    brain.send_event(BrainEvent::DragEnded);
                }
            } else if let Some((dx, dy)) = app.update_drag(monitor_height as i32) {
                if let Some(ref brain) = app.brain {
                    brain.send_event(BrainEvent::Dragged { dx, dy });
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        } else {
            let time_to_next = animation_speed.saturating_sub(elapsed);
            std::thread::sleep(time_to_next.min(Duration::from_millis(10)));
        }
    }

    Ok(())
}
