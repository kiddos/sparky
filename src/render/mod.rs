pub mod gl_helpers;
pub mod quad;
pub mod shader;

pub use quad::RenderQuad;
pub use shader::ShaderProgram;

use glfw::{Action, Context, GlfwReceiver, WindowEvent, WindowHint, WindowMode};

pub struct PetWindow {
    pub glfw: glfw::Glfw,
    pub window: glfw::PWindow,
    pub events: GlfwReceiver<(f64, WindowEvent)>,
    #[allow(dead_code)]
    pub width: u32,
    pub height: u32,
    pub is_dragging: bool,
    pub drag_start_cursor: (f64, f64),
    pub drag_start_pos: (i32, i32),
    pub is_click_through: bool,
}

impl PetWindow {
    pub fn new(width: u32, height: u32, title: &str) -> anyhow::Result<Self> {
        let mut glfw = glfw::init(glfw::fail_on_errors)?;

        // Window Hints for Desktop Pet
        glfw.window_hint(WindowHint::ContextVersion(3, 3));
        glfw.window_hint(WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
        glfw.window_hint(WindowHint::TransparentFramebuffer(true));
        glfw.window_hint(WindowHint::Decorated(false));
        glfw.window_hint(WindowHint::Floating(true)); // Always-on-top
        glfw.window_hint(WindowHint::Resizable(false));
        glfw.window_hint(WindowHint::Samples(Some(0))); // No MSAA needed for quad

        let (mut window, events) = glfw
            .create_window(width, height, title, WindowMode::Windowed)
            .ok_or_else(|| anyhow::anyhow!("Failed to create GLFW window"))?;

        window.make_current();
        window.set_all_polling(true);

        // Load OpenGL function pointers
        gl::load_with(|s| window.get_proc_address(s) as *const _);

        // Enable Alpha Blending for Transparency
        unsafe {
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
        }

        // Set initial position near bottom right of screen
        let (screen_w, screen_h) = glfw
            .with_primary_monitor(|_, m| {
                m.and_then(|mon| mon.get_video_mode())
                    .map(|vm| (vm.width as i32, vm.height as i32))
            })
            .unwrap_or((1920, 1080));

        let initial_x = (screen_w - width as i32 - 80).max(0);
        let initial_y = (screen_h - height as i32 - 80).max(0);
        window.set_pos(initial_x, initial_y);

        Ok(Self {
            glfw,
            window,
            events,
            width,
            height,
            is_dragging: false,
            drag_start_cursor: (0.0, 0.0),
            drag_start_pos: (initial_x, initial_y),
            is_click_through: false,
        })
    }

    pub fn handle_mouse_drag(&mut self, event: &WindowEvent) {
        if self.is_click_through {
            return;
        }

        match event {
            WindowEvent::MouseButton(glfw::MouseButton::Button1, Action::Press, _) => {
                let (cx, cy) = self.window.get_cursor_pos();
                let (wx, wy) = self.window.get_pos();
                self.is_dragging = true;
                self.drag_start_cursor = (cx, cy);
                self.drag_start_pos = (wx, wy);
            }
            WindowEvent::MouseButton(glfw::MouseButton::Button1, Action::Release, _) => {
                self.is_dragging = false;
            }
            WindowEvent::CursorPos(cx, cy) => {
                if self.is_dragging {
                    let dx = (cx - self.drag_start_cursor.0) as i32;
                    let dy = (cy - self.drag_start_cursor.1) as i32;
                    self.window
                        .set_pos(self.drag_start_pos.0 + dx, self.drag_start_pos.1 + dy);
                }
            }
            _ => {}
        }
    }

    pub fn set_click_through(&mut self, enabled: bool) {
        self.is_click_through = enabled;
        log::info!("Click-through mode set to: {}", enabled);
    }
}
