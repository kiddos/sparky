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
    pub current_pos: (i32, i32),
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

        // Query and log OpenGL context details
        unsafe {
            let get_gl_string = |name| {
                let p = gl::GetString(name);
                if p.is_null() {
                    "Unknown".to_string()
                } else {
                    std::ffi::CStr::from_ptr(p as *const _)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            let vendor = get_gl_string(gl::VENDOR);
            let renderer = get_gl_string(gl::RENDERER);
            let version = get_gl_string(gl::VERSION);
            log::info!("OpenGL context loaded: {vendor} | {renderer} | Version: {version}");
        }

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
        log::info!(
            "PetWindow initialized (size: {width}x{height}, screen: {screen_w}x{screen_h}, position: ({initial_x}, {initial_y}))"
        );

        Ok(Self {
            glfw,
            window,
            events,
            width,
            height,
            is_dragging: false,
            drag_start_cursor: (0.0, 0.0),
            drag_start_pos: (initial_x, initial_y),
            current_pos: (initial_x, initial_y),
            is_click_through: false,
        })
    }

    pub fn set_pos(&mut self, x: i32, y: i32) {
        self.current_pos = (x, y);
        self.window.set_pos(x, y);
    }

    pub fn get_pos(&self) -> (i32, i32) {
        self.window.get_pos()
    }

    pub fn handle_mouse_drag(&mut self, event: &WindowEvent) {
        if self.is_click_through {
            return;
        }

        match event {
            WindowEvent::MouseButton(glfw::MouseButton::Button1, Action::Press, _) => {
                let (cx, cy) = self.window.get_cursor_pos();
                let (wx, wy) = self.current_pos;
                self.is_dragging = true;
                self.drag_start_cursor = (cx, cy);
                self.drag_start_pos = (wx, wy);
            }
            WindowEvent::MouseButton(glfw::MouseButton::Button1, Action::Release, _) => {
                self.is_dragging = false;
            }
            WindowEvent::Focus(false) => {
                self.is_dragging = false;
            }
            WindowEvent::Pos(x, y) => {
                self.current_pos = (*x, *y);
            }
            WindowEvent::CursorPos(cx, cy) if self.is_dragging => {
                // Correct the window-relative cursor position by adding back
                // the window's accumulated displacement since drag start.
                // When the window moves under the cursor, (cx, cy) shifts by the same amount,
                // which previously caused a jittery feedback loop resetting to drag_start_pos.
                let actual_cx = cx + (self.current_pos.0 - self.drag_start_pos.0) as f64;
                let actual_cy = cy + (self.current_pos.1 - self.drag_start_pos.1) as f64;

                let dx = (actual_cx - self.drag_start_cursor.0) as i32;
                let dy = (actual_cy - self.drag_start_cursor.1) as i32;

                let target_x = self.drag_start_pos.0 + dx;
                let target_y = self.drag_start_pos.1 + dy;

                if target_x != self.current_pos.0 || target_y != self.current_pos.1 {
                    self.set_pos(target_x, target_y);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn test_mouse_drag_lifecycle_and_jitter_prevention() {
        let mut pet_window = PetWindow::new(288, 162, "TestSparky").unwrap();

        // 1. Initial state
        assert!(!pet_window.is_dragging);
        let start_pos = pet_window.current_pos;

        // 2. Press triggers drag
        pet_window.handle_mouse_drag(&WindowEvent::MouseButton(
            glfw::MouseButton::Button1,
            Action::Press,
            glfw::Modifiers::empty(),
        ));
        assert!(pet_window.is_dragging);

        // 3. Focus loss cancels drag
        pet_window.handle_mouse_drag(&WindowEvent::Focus(false));
        assert!(!pet_window.is_dragging);

        // 4. Release cancels drag
        pet_window.handle_mouse_drag(&WindowEvent::MouseButton(
            glfw::MouseButton::Button1,
            Action::Press,
            glfw::Modifiers::empty(),
        ));
        assert!(pet_window.is_dragging);
        pet_window.handle_mouse_drag(&WindowEvent::MouseButton(
            glfw::MouseButton::Button1,
            Action::Release,
            glfw::Modifiers::empty(),
        ));
        assert!(!pet_window.is_dragging);

        // 5. Click-through disables dragging
        pet_window.set_click_through(true);
        pet_window.handle_mouse_drag(&WindowEvent::MouseButton(
            glfw::MouseButton::Button1,
            Action::Press,
            glfw::Modifiers::empty(),
        ));
        assert!(!pet_window.is_dragging);
        pet_window.set_click_through(false);

        // 6. set_pos updates current_pos immediately
        pet_window.set_pos(start_pos.0 + 50, start_pos.1 + 30);
        assert_eq!(pet_window.current_pos, (start_pos.0 + 50, start_pos.1 + 30));

        // 7. Jitter prevention: simulate cursor movement and compensating event
        pet_window.set_pos(200, 200);
        pet_window.handle_mouse_drag(&WindowEvent::MouseButton(
            glfw::MouseButton::Button1,
            Action::Press,
            glfw::Modifiers::empty(),
        ));
        assert!(pet_window.is_dragging);
        let (start_cx, start_cy) = pet_window.drag_start_cursor;
        let pos_before_move = pet_window.current_pos;

        // Simulate mouse move: cursor moved by (+20, +10)
        pet_window.handle_mouse_drag(&WindowEvent::CursorPos(start_cx + 20.0, start_cy + 10.0));
        let moved_pos = pet_window.current_pos;
        assert_eq!(moved_pos, (pos_before_move.0 + 20, pos_before_move.1 + 10));

        // When the window moves under the mouse, compensating relative CursorPos event arrives
        // where cursor is back at (start_cx, start_cy) relative to the new window.
        // It MUST NOT snap back to pos_before_move!
        pet_window.handle_mouse_drag(&WindowEvent::CursorPos(start_cx, start_cy));
        assert_eq!(
            pet_window.current_pos, moved_pos,
            "Window must remain at moved_pos and not snap back"
        );

        // Move 2: further +15, +5 (relative cursor becomes start_cx + 15, start_cy + 5)
        pet_window.handle_mouse_drag(&WindowEvent::CursorPos(start_cx + 15.0, start_cy + 5.0));
        let moved_pos2 = pet_window.current_pos;
        assert_eq!(moved_pos2, (pos_before_move.0 + 35, pos_before_move.1 + 15));

        // Compensating event arrives for Move 2
        pet_window.handle_mouse_drag(&WindowEvent::CursorPos(start_cx, start_cy));
        assert_eq!(pet_window.current_pos, moved_pos2);

        // Move 3: drag backwards -10, -5
        pet_window.handle_mouse_drag(&WindowEvent::CursorPos(start_cx - 10.0, start_cy - 5.0));
        let moved_pos3 = pet_window.current_pos;
        assert_eq!(moved_pos3, (pos_before_move.0 + 25, pos_before_move.1 + 10));

        // Compensating event arrives for Move 3
        pet_window.handle_mouse_drag(&WindowEvent::CursorPos(start_cx, start_cy));
        assert_eq!(pet_window.current_pos, moved_pos3);

        // Release drag
        pet_window.handle_mouse_drag(&WindowEvent::MouseButton(
            glfw::MouseButton::Button1,
            Action::Release,
            glfw::Modifiers::empty(),
        ));
        assert!(!pet_window.is_dragging);
    }
}
