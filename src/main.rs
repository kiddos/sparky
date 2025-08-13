#![windows_subsystem = "windows"]

use std::error::Error;
use std::ffi::CString;
use std::ffi::c_void;
use std::num::NonZeroU32;
use std::time::Duration;
use std::time::Instant;

use glutin::config::Config;
use glutin::config::ConfigTemplateBuilder;
use glutin::config::GetGlConfig;
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentContext, PossiblyCurrentContext, Version,
};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{Surface, SurfaceAttributesBuilder, WindowSurface};
use glutin_winit::DisplayBuilder;
use rand::Rng;
use rand::rngs::ThreadRng;
use raw_window_handle::HasWindowHandle;
use tray_icon::TrayIcon;
use tray_icon::TrayIconBuilder;
use tray_icon::menu::Menu;
use tray_icon::menu::MenuEvent;
use tray_icon::menu::MenuId;
use tray_icon::menu::MenuItem;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::dpi::PhysicalSize;
use winit::event::ElementState;
use winit::event::MouseButton;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes, WindowLevel};

#[cfg(target_os = "linux")]
use winit::platform::x11::WindowAttributesExtX11;

#[cfg(target_os = "windows")]
use winit::platform::windows::WindowAttributesExtWindows;

// --- Sprites ---
const BLINK1: &[u8] = include_bytes!("../assets/blink1.png");
const BLINK2: &[u8] = include_bytes!("../assets/blink2.png");
const BLINK3: &[u8] = include_bytes!("../assets/blink3.png");
const HELLO1: &[u8] = include_bytes!("../assets/hello1.png");
const HIGH_FIVE1: &[u8] = include_bytes!("../assets/highfive1.png");
const HIGH_FIVE2: &[u8] = include_bytes!("../assets/highfive2.png");
const IDLE1: &[u8] = include_bytes!("../assets/idle1.png");
const IDLE2: &[u8] = include_bytes!("../assets/idle2.png");
const LAUGH1: &[u8] = include_bytes!("../assets/laugh1.png");
const LAUGH2: &[u8] = include_bytes!("../assets/laugh2.png");
const SMILE1: &[u8] = include_bytes!("../assets/smile1.png");
const SMILE2: &[u8] = include_bytes!("../assets/smile2.png");
const SMILE3: &[u8] = include_bytes!("../assets/smile3.png");
const TALK1: &[u8] = include_bytes!("../assets/talk1.png");
const TALK2: &[u8] = include_bytes!("../assets/talk2.png");
const TIRED1: &[u8] = include_bytes!("../assets/tired1.png");
const TIRED2: &[u8] = include_bytes!("../assets/tired2.png");
const TIRED3: &[u8] = include_bytes!("../assets/tired3.png");
const ICON: &[u8] = include_bytes!("../images/sparky.png");

const FRAME_WIDTH: u32 = 128;
const FRAME_HEIGHT: u32 = 128;
const NUM_FRAMES: u32 = 65;
const ANIMATION_SPEED: Duration = Duration::from_millis(66);

// menu
const SHOW_MENU_ID: &str = "SHOW_MENU";
const QUIT_MENU_ID: &str = "QUIT_MENU";

// Vertex shader: Positions the vertices of our quad and sets their texture coordinates.
const VS_SRC: &str = r#"
    #version 440 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main() {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
"#;

// Fragment shader: Samples the texture at a given coordinate and outputs the color.
const FS_SRC: &str = r#"
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;

    uniform sampler2D ourTexture;
    uniform float frame_offset; // Horizontal offset for the current animation frame

    void main() {
        // Apply the horizontal offset to the texture coordinate
        vec2 newTexCoord = vec2(TexCoord.x + frame_offset, TexCoord.y);
        vec4 texColor = texture(ourTexture, newTexCoord);
        
        // Discard pixels that are mostly transparent to maintain the shape
        if(texColor.a < 0.1)
            discard;

        FragColor = texColor;
    }
"#;

#[derive(Debug)]
enum UserEvent {
    MenuEvent(tray_icon::menu::MenuEvent),
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;

    let mut app = App::new();
    app.init(&event_loop);
    app.create_tray_icon(&event_loop);
    event_loop.run_app(&mut app)?;

    Ok(())
}

pub fn gl_config_picker(configs: Box<dyn Iterator<Item = Config> + '_>) -> Config {
    configs
        .reduce(|accum, config| {
            let transparency_check = config.supports_transparency().unwrap_or(false)
                & !accum.supports_transparency().unwrap_or(false);

            if transparency_check || config.num_samples() > accum.num_samples() {
                config
            } else {
                accum
            }
        })
        .unwrap()
}

struct App {
    window: Option<Window>,
    gl_context: Option<PossiblyCurrentContext>,
    gl_surface: Option<Surface<WindowSurface>>,
    shader_program: u32,
    vao: u32,
    vbo: u32,
    ebo: u32,
    current_frame: u32,
    textures: Vec<u32>,
    active_texture_index: usize,
    bind_texture: bool,
    last_update: Instant,
    rng: ThreadRng,
    #[cfg(not(target_os = "linux"))]
    tray_icon: Option<TrayIcon>,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            gl_context: None,
            gl_surface: None,
            shader_program: 0,
            vao: 0,
            vbo: 0,
            ebo: 0,
            current_frame: 0,
            textures: Vec::new(),
            active_texture_index: 0,
            bind_texture: false,
            last_update: Instant::now(),
            rng: rand::rng(),
            #[cfg(not(target_os = "linux"))]
            tray_icon: None,
        }
    }

    pub fn init(&mut self, event_loop: &EventLoop<UserEvent>) {
        let template = ConfigTemplateBuilder::new().with_alpha_size(8);

        let display_builder =
            DisplayBuilder::new().with_window_attributes(Some(window_attributes()));

        let result = display_builder.build(event_loop, template, gl_config_picker);
        if let Ok((window, gl_config)) = result {
            self.window = window;
            self.init_window_position();

            self.gl_context = Some(self.create_context(&gl_config).treat_as_possibly_current());
            self.gl_surface = Some(self.create_surface(&gl_config));
            self.init_opengl_buffers();
            let _ = self.load_textures();
        }
    }

    fn init_window_position(&self) {
        let window = self.window.as_ref().unwrap();

        let monitor = window.current_monitor().unwrap();
        let monitor_size = monitor.size();
        let window_size = window.outer_size();

        // Compute bottom-right position
        let x = monitor_size.width.saturating_sub(window_size.width);
        let y = monitor_size.height.saturating_sub(window_size.height);

        // Move window
        window.set_outer_position(PhysicalPosition::new(x as i32, y as i32 - 30));
    }

    fn create_context(&self, gl_config: &Config) -> NotCurrentContext {
        let gl_display = gl_config.display();
        let raw_window_handle = self
            .window
            .as_ref()
            .unwrap()
            .window_handle()
            .ok()
            .map(|wh| wh.as_raw());

        let fallback_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(raw_window_handle);

        let legacy_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
            .build(raw_window_handle);

        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(None))
            .build(raw_window_handle);
        unsafe {
            gl_display
                .create_context(gl_config, &context_attributes)
                .unwrap_or_else(|_| {
                    gl_display
                        .create_context(gl_config, &fallback_context_attributes)
                        .unwrap_or_else(|_| {
                            gl_display
                                .create_context(gl_config, &legacy_context_attributes)
                                .expect("failed to create context")
                        })
                })
        }
    }

    fn create_surface(&self, gl_config: &Config) -> Surface<WindowSurface> {
        let window = self.window.as_ref().unwrap();
        let raw_window_handle = window.window_handle().ok().map(|wh| wh.as_raw()).unwrap();
        let gl_display = gl_config.display();

        let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle,
            NonZeroU32::new(FRAME_WIDTH).unwrap(),
            NonZeroU32::new(FRAME_HEIGHT).unwrap(),
        );
        unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attributes)
                .expect("Failed to create window surface")
        }
    }

    fn init_opengl_buffers(&mut self) {
        let gl_context = self.gl_context.as_ref().unwrap();
        let gl_surface = self.gl_surface.as_ref().unwrap();
        gl_context.make_current(gl_surface).unwrap();

        let gl_config = self.gl_context.as_ref().unwrap().config();
        let gl_display = gl_config.display();

        gl::load_with(|symbol| {
            let c_str = CString::new(symbol).unwrap();
            gl_display.get_proc_address(&c_str)
        });

        self.shader_program = unsafe {
            let vertex_shader = create_shader(gl::VERTEX_SHADER, VS_SRC);
            let fragment_shader = create_shader(gl::FRAGMENT_SHADER, FS_SRC);

            let program = gl::CreateProgram();
            gl::AttachShader(program, vertex_shader);
            gl::AttachShader(program, fragment_shader);
            gl::LinkProgram(program);

            gl::UseProgram(program);

            gl::DeleteShader(vertex_shader);
            gl::DeleteShader(fragment_shader);
            program
        };

        let vertices: [f32; 16] = [
            // positions   // texture coords
            1.0,
            1.0,
            1.0 / NUM_FRAMES as f32,
            0.0, // top right
            1.0,
            -1.0,
            1.0 / NUM_FRAMES as f32,
            1.0, // bottom right
            -1.0,
            -1.0,
            0.0,
            1.0, // bottom left
            -1.0,
            1.0,
            0.0,
            0.0, // top left
        ];
        let indices: [u32; 6] = [
            0, 1, 3, // first triangle
            1, 2, 3, // second triangle
        ];

        let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);
            gl::GenBuffers(1, &mut ebo);

            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * std::mem::size_of::<f32>()) as isize,
                vertices.as_ptr() as *const c_void,
                gl::STATIC_DRAW,
            );

            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (indices.len() * std::mem::size_of::<u32>()) as isize,
                indices.as_ptr() as *const c_void,
                gl::STATIC_DRAW,
            );

            // Position attribute
            gl::VertexAttribPointer(
                0,
                2,
                gl::FLOAT,
                gl::FALSE,
                4 * std::mem::size_of::<f32>() as i32,
                std::ptr::null(),
            );
            gl::EnableVertexAttribArray(0);
            // Texture coord attribute
            gl::VertexAttribPointer(
                1,
                2,
                gl::FLOAT,
                gl::FALSE,
                4 * std::mem::size_of::<f32>() as i32,
                (2 * std::mem::size_of::<f32>()) as *const c_void,
            );
            gl::EnableVertexAttribArray(1);

            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            gl::Viewport(0, 0, FRAME_WIDTH as i32, FRAME_HEIGHT as i32);
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        self.vao = vao;
        self.vbo = vbo;
        self.ebo = ebo;

        self.swap_buffers();
    }

    fn swap_buffers(&self) {
        let gl_surface = self.gl_surface.as_ref().unwrap();
        let gl_context = self.gl_context.as_ref().unwrap();
        gl_surface.swap_buffers(gl_context).unwrap();
    }

    fn load_textures(&mut self) -> Result<(), Box<dyn Error>> {
        let blink1_texture = load_texture_from_bytes(BLINK1)?;
        let blink2_texture = load_texture_from_bytes(BLINK2)?;
        let blink3_texture = load_texture_from_bytes(BLINK3)?;
        let hello1_texture = load_texture_from_bytes(HELLO1)?;
        let highfive1_texture = load_texture_from_bytes(HIGH_FIVE1)?;
        let highfive2_texture = load_texture_from_bytes(HIGH_FIVE2)?;
        let idle1_texture = load_texture_from_bytes(IDLE1)?;
        let idle2_texture = load_texture_from_bytes(IDLE2)?;
        let laugh1_texture = load_texture_from_bytes(LAUGH1)?;
        let laugh2_texture = load_texture_from_bytes(LAUGH2)?;
        let smile1_texture = load_texture_from_bytes(SMILE1)?;
        let smile2_texture = load_texture_from_bytes(SMILE2)?;
        let smile3_texture = load_texture_from_bytes(SMILE3)?;
        let talk1_texture = load_texture_from_bytes(TALK1)?;
        let talk2_texture = load_texture_from_bytes(TALK2)?;
        let tired1_texture = load_texture_from_bytes(TIRED1)?;
        let tired2_texture = load_texture_from_bytes(TIRED2)?;
        let tired3_texture = load_texture_from_bytes(TIRED3)?;

        let textures = [
            blink1_texture,
            blink2_texture,
            blink3_texture,
            hello1_texture,
            highfive1_texture,
            highfive2_texture,
            idle1_texture,
            idle2_texture,
            laugh1_texture,
            laugh2_texture,
            smile1_texture,
            smile2_texture,
            smile3_texture,
            talk1_texture,
            talk2_texture,
            tired1_texture,
            tired2_texture,
            tired3_texture,
        ];

        self.textures = textures.to_vec();
        self.active_texture_index = 0;
        self.bind_texture = true;

        Ok(())
    }

    fn render(&mut self) {
        unsafe {
            // Clear the screen to transparent black
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::UseProgram(self.shader_program);

            if self.bind_texture {
                gl::BindTexture(gl::TEXTURE_2D, self.textures[self.active_texture_index]);
                self.bind_texture = false;
            }

            // Set the uniform for the current animation frame
            let offset = (self.current_frame as f32) / (NUM_FRAMES as f32);
            let uniform_location = gl::GetUniformLocation(
                self.shader_program,
                CString::new("frame_offset").unwrap().as_ptr(),
            );
            gl::Uniform1f(uniform_location, offset);

            gl::BindVertexArray(self.vao);
            gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
        }

        self.swap_buffers();
    }

    pub fn create_tray_icon(&mut self, event_loop: &EventLoop<UserEvent>) {
        #[cfg(target_os = "linux")]
        std::thread::spawn(|| {
            gtk::init().unwrap();
            let _tray_icon = Some(create_tray_icon());
            gtk::main();
        });

        #[cfg(not(target_os = "macos"))]
        {
            let proxy = event_loop.create_proxy();
            MenuEvent::set_event_handler(Some(move |event| {
                let _ = proxy.send_event(UserEvent::MenuEvent(event));
            }));
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let gl_config = self.gl_context.as_ref().unwrap().config();
        let _ = glutin_winit::finalize_window(event_loop, window_attributes(), &gl_config);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                let gl_surface = self.gl_surface.as_ref().unwrap();
                let gl_context = self.gl_context.as_ref().unwrap();
                gl_surface.resize(
                    &gl_context,
                    NonZeroU32::new(physical_size.width).unwrap(),
                    NonZeroU32::new(physical_size.height).unwrap(),
                );
                let size = physical_size.width.min(physical_size.height);
                unsafe {
                    gl::Viewport(0, 0, size as i32, size as i32);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let window = self.window.as_ref().unwrap();
                window.drag_window().unwrap();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let _ = event_loop;
        if self.last_update.elapsed() >= ANIMATION_SPEED {
            self.current_frame = (self.current_frame + 1) % NUM_FRAMES;
            self.last_update = Instant::now();
            if self.current_frame == 0 {
                self.active_texture_index = self.rng.random_range(0..self.textures.len());
                self.bind_texture = true;
            }
            let window = self.window.as_ref().unwrap();
            window.request_redraw();
        } else {
            let time_to_next_frame = ANIMATION_SPEED - self.last_update.elapsed();
            event_loop
                .set_control_flow(ControlFlow::WaitUntil(Instant::now() + time_to_next_frame));
        }
    }

    fn new_events(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        // We create the icon once the event loop is actually running
        // to prevent issues like https://github.com/tauri-apps/tray-icon/issues/90
        if winit::event::StartCause::Init == cause {
            #[cfg(not(target_os = "linux"))]
            {
                self.tray_icon = Some(create_tray_icon());
            }
        }
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        let window = self.window.as_ref().unwrap();
        match event {
            UserEvent::MenuEvent(event) => {
                if event.id() == SHOW_MENU_ID {
                    window.set_visible(true);
                } else if event.id() == QUIT_MENU_ID {
                    event_loop.exit();
                }
            }
        }
    }
}

fn window_attributes() -> WindowAttributes {
    let mut attr = Window::default_attributes()
        .with_inner_size(PhysicalSize::new(FRAME_WIDTH, FRAME_HEIGHT))
        .with_decorations(false)
        .with_transparent(true)
        .with_window_level(WindowLevel::AlwaysOnTop);

    #[cfg(target_os = "linux")]
    {
        use winit::platform::x11::WindowType;
        attr = attr.with_x11_window_type(vec![WindowType::Utility]);
    }
    #[cfg(target_os = "windows")]
    {
        attr = attr.with_skip_taskbar(true);
    }

    attr
}

unsafe fn create_shader(shader_type: gl::types::GLenum, source: &str) -> gl::types::GLuint {
    let shader = unsafe { gl::CreateShader(shader_type) };
    let c_str = CString::new(source.as_bytes()).unwrap();
    unsafe {
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);

        // check shader compile status
        let mut success = gl::FALSE as gl::types::GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as gl::types::GLint {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);

            let error_message = if len > 0 {
                let mut buffer: Vec<u8> = Vec::with_capacity(len as usize);
                buffer.set_len(len as usize); // Set length to capacity
                gl::GetShaderInfoLog(
                    shader,
                    len,
                    std::ptr::null_mut(),
                    buffer.as_mut_ptr() as *mut gl::types::GLchar,
                );
                String::from_utf8_lossy(&buffer).into_owned()
            } else {
                "Unknown shader compilation error".to_string()
            };

            gl::DeleteShader(shader);
            println!("error: {}", error_message);
        }
    }

    shader
}

fn load_texture_from_bytes(bytes: &[u8]) -> Result<u32, image::ImageError> {
    let image = image::load_from_memory(bytes)?.to_rgba8();
    let (img_width, img_height) = image.dimensions();
    let image_data = image.into_raw();

    let mut texture_id = 0;
    unsafe {
        gl::GenTextures(1, &mut texture_id);
        gl::BindTexture(gl::TEXTURE_2D, texture_id);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);

        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGBA as i32,
            img_width as i32,
            img_height as i32,
            0,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            image_data.as_ptr() as *const c_void,
        );
        gl::GenerateMipmap(gl::TEXTURE_2D);
    }
    Ok(texture_id)
}

fn create_tray_icon() -> TrayIcon {
    let show_menu_item = MenuItem::with_id(MenuId::new(SHOW_MENU_ID), "Show", true, None);
    let quit_menu_item = MenuItem::with_id(MenuId::new(QUIT_MENU_ID), "Quit", true, None);
    let menu = Menu::new();
    let _ = menu.append(&show_menu_item);
    let _ = menu.append(&quit_menu_item);
    let tray_icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Sparky")
        .with_icon(load_icon())
        .build()
        .unwrap();
    let _ = tray_icon.set_visible(true);
    tray_icon
}

fn load_icon() -> tray_icon::Icon {
    let image = image::load_from_memory(ICON).expect("").to_rgba8();
    let image = image::imageops::resize(&image, 256, 256, image::imageops::FilterType::Gaussian);
    let (img_width, img_height) = image.dimensions();
    let image_data = image.into_raw();
    tray_icon::Icon::from_rgba(image_data, img_width, img_height).expect("Failed to open icon")
}
