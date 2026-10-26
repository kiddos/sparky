#![windows_subsystem = "windows"]

mod app;
mod cli;
mod shader;
mod sprite;
mod tray;
mod util;
mod video;

use std::error::Error;
use std::time::{Duration, Instant};

use glfw::{Action, Context, Key, WindowEvent};
use log::info;
use tray_icon::menu::MenuEvent;

use crate::app::App;
use crate::cli::parse_args;
use crate::sprite::{DEFAULT_ANIMATION_SPEED, DEFAULT_SPRITE_HEIGHT, DEFAULT_SPRITE_WIDTH};
use crate::tray::{QUIT_MENU_ID, SHOW_MENU_ID, create_tray_icon};
use crate::util::init_log;
use crate::video::{find_videos_dir, load_video_animations, probe_video_specs};

pub fn main() -> Result<(), Box<dyn Error>> {
    init_log();
    info!("Starting Sparky...");

    let cli = parse_args()?;

    #[cfg(target_os = "linux")]
    {
        if let Err(e) = gtk::init() {
            eprintln!("Failed to initialize GTK for tray icon: {}", e);
            std::process::exit(1);
        }
    }

    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();

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
                    DEFAULT_SPRITE_WIDTH,
                    DEFAULT_SPRITE_HEIGHT,
                    DEFAULT_ANIMATION_SPEED,
                    0.0,
                )
            }
        } else {
            (
                DEFAULT_SPRITE_WIDTH,
                DEFAULT_SPRITE_HEIGHT,
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

    window.set_pos(
        monitor_x + monitor_width as i32 - frame_width as i32,
        monitor_y + monitor_height as i32 - frame_height as i32 - 30,
    );

    let mut app = App::new(window, frame_width, frame_height, black_threshold);
    app.init_opengl();
    app.load_animations(cli.video_dir.as_deref())?;
    app.render();

    let _tray_icon = create_tray_icon();
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
                WindowEvent::MouseButton(glfw::MouseButtonLeft, Action::Press, _) => {
                    let (x, y) = app.window.get_cursor_pos();
                    if app.is_pixel_opaque(x, y) {
                        app.is_dragging = true;
                        app.drag_start_pos = (x, y);
                    }
                }
                WindowEvent::MouseButton(glfw::MouseButtonLeft, Action::Release, _) => {
                    app.is_dragging = false;
                }
                WindowEvent::CursorPos(x, y) if app.is_dragging => {
                    let (wx, wy) = app.window.get_pos();
                    let dx = x - app.drag_start_pos.0;
                    let dy = y - app.drag_start_pos.1;
                    app.window.set_pos(wx + dx as i32, wy + dy as i32);
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
            }
        }

        let elapsed = last_update.elapsed();
        if elapsed >= animation_speed {
            app.update();
            last_update = Instant::now();
            app.render();
        }

        if app.is_dragging {
            std::thread::sleep(Duration::from_millis(1));
        } else {
            let time_to_next = animation_speed.saturating_sub(elapsed);
            std::thread::sleep(time_to_next.min(Duration::from_millis(10)));
        }
    }

    Ok(())
}
