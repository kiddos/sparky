#![windows_subsystem = "windows"]

use std::ffi::{CString, c_void};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;
use glutin::config::{Config, ConfigTemplateBuilder};
use glutin::context::{ContextApi, ContextAttributesBuilder};
use glutin::display::{Display, GetGlDisplay};
use glutin::prelude::*;
use glutin::surface::{Surface, SurfaceAttributesBuilder, WindowSurface};
use raw_window_handle::HasRawWindowHandle;
use tray_icon::menu::{Menu, MenuItem};
use tray_icon::TrayIconBuilder;
use tray_icon::menu::MenuEvent;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::window::{Window, WindowBuilder, WindowLevel};

#[cfg(target_os = "linux")]
use winit::platform::x11::WindowBuilderExtX11;

#[cfg(target_os = "windows")]
use winit::platform::windows::WindowBuilderExtWindows;

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

// --- OpenGL Shader Source Code ---

// Vertex shader: Positions the vertices of our quad and sets their texture coordinates.
const VS_SRC: &str = r#"
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main() {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
"#;

// Fragment shader: Samples the texture at a given coordinate and outputs the color.
// It receives a 'frame_offset' to select the correct part of the sprite sheet.
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

#[derive(Debug)]
enum UserEvent {
    MenuEvent(tray_icon::menu::MenuEvent),
}

fn load_icon() -> tray_icon::Icon {
    let image = image::load_from_memory(ICON).expect("").to_rgba8();
    let image = image::imageops::resize(&image, 256, 256, image::imageops::FilterType::Gaussian);
    let (img_width, img_height) = image.dimensions();
    let image_data = image.into_raw();
    tray_icon::Icon::from_rgba(image_data, img_width, img_height).expect("Failed to open icon")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Window and Event Loop Setup ---
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build()?;
    let mut window_builder = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(FRAME_WIDTH, FRAME_HEIGHT))
        .with_decorations(false)
        .with_transparent(true)
        .with_window_level(WindowLevel::AlwaysOnTop);
    #[cfg(target_os = "linux")]
    {
        use winit::platform::x11::XWindowType;
        window_builder = window_builder.with_x11_window_type(vec![XWindowType::Utility]);
    }
    #[cfg(target_os = "windows")]
    {
        window_builder = window_builder.with_skip_taskbar(true);
    }

    // 1. Create a template for the OpenGL configuration.
    let template = ConfigTemplateBuilder::new().with_alpha_size(8); // Enable alpha for transparency

    let (window, gl_config) = glutin_winit::DisplayBuilder::new()
        .with_window_builder(Some(window_builder))
        .build(&event_loop, template, gl_config_picker)?;

    let window = window.unwrap();

    let monitor = window.current_monitor().unwrap();
    let monitor_size = monitor.size();
    // Get window size
    let window_size = window.outer_size();

    // Compute bottom-right position
    let x = monitor_size.width.saturating_sub(window_size.width);
    let y = monitor_size.height.saturating_sub(window_size.height);

    // Move window
    window.set_outer_position(PhysicalPosition::new(x as i32, y as i32 - 30));

    let window: Arc<Window> = Arc::new(window);
    let raw_window_handle = window.raw_window_handle();

    let show_menu_item = MenuItem::new("Show", true, None);
    let quit_menu_item = MenuItem::new("Quit", true, None);
    let menu = Menu::new();
    menu.append(&show_menu_item)?;
    menu.append(&quit_menu_item)?;
    let tray_icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Sparky")
        .with_title("Sparky")
        .with_icon(load_icon())
        .build()
        .unwrap();
    tray_icon.set_visible(true)?;

    #[cfg(not(target_os = "macos"))]
    {
        let proxy = event_loop.create_proxy();
        MenuEvent::set_event_handler(Some(move |event| {
            let _ = proxy.send_event(UserEvent::MenuEvent(event));
        }));
    }

    // 3. Get the GL display and create a GL context.
    let gl_display: Display = gl_config.display();
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(None))
        .build(Some(raw_window_handle));
    let gl_context = unsafe {
        gl_display
            .create_context(&gl_config, &context_attributes)
            .expect("failed to create context")
    };

    // 4. Create the GL surface.
    let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        raw_window_handle,
        NonZeroU32::new(FRAME_WIDTH).unwrap(),
        NonZeroU32::new(FRAME_HEIGHT).unwrap(),
    );
    let gl_surface: Surface<WindowSurface> = unsafe {
        gl_display
            .create_window_surface(&gl_config, &surface_attributes)
            .expect("Failed to create window surface")
    };

    // 5. Make the context current.
    let gl_context = gl_context.make_current(&gl_surface)?;

    // --- Load OpenGL Functions ---
    gl::load_with(|symbol| {
        let c_str = CString::new(symbol).unwrap();
        gl_display.get_proc_address(&c_str)
    });

    // --- OpenGL Rendering Setup ---
    let shader_program = unsafe {
        let vertex_shader = create_shader(gl::VERTEX_SHADER, VS_SRC);
        let fragment_shader = create_shader(gl::FRAGMENT_SHADER, FS_SRC);

        let program = gl::CreateProgram();
        gl::AttachShader(program, vertex_shader);
        gl::AttachShader(program, fragment_shader);
        gl::LinkProgram(program);
        // Check for linking errors...

        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
        program
    };

    // --- Vertex Data and Buffers ---
    // A quad (two triangles) that fills the window.
    // Vertices are in pairs: (x, y, tex_coord_x, tex_coord_y)
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
    }

    // Enable blending for transparency
    unsafe {
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    }

    unsafe {
        gl::ClearColor(0.0, 0.0, 0.0, 0.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);
    }

    gl_surface.swap_buffers(&gl_context).unwrap();

    // --- Texture Loading ---
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
    let mut active_texture_index = 0;
    let mut rebind_texture = true;


    // --- Animation State ---
    let mut current_frame: u32 = 0;
    let mut last_update = Instant::now();
    let mut rng = rand::rng();

    // --- Event Loop ---
    event_loop.run(move |event, elwt| {
        match event {
            Event::UserEvent(event) => match event {
                UserEvent::MenuEvent(event) => {
                    if event.id() == show_menu_item.id() {
                        window.set_visible(true);
                    } else if event.id() == quit_menu_item.id() {
                        elwt.exit();
                    }
                },
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    gl_surface.resize(
                        &gl_context,
                        NonZeroU32::new(physical_size.width).unwrap(),
                        NonZeroU32::new(physical_size.height).unwrap(),
                    );
                }
                WindowEvent::RedrawRequested => {
                    unsafe {
                        // Clear the screen to transparent black
                        gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                        gl::Clear(gl::COLOR_BUFFER_BIT);

                        gl::UseProgram(shader_program);

                        if rebind_texture {
                            gl::BindTexture(gl::TEXTURE_2D, textures[active_texture_index]);
                            rebind_texture = false;
                        }

                        // Set the uniform for the current animation frame
                        let offset = (current_frame as f32) / (NUM_FRAMES as f32);
                        let uniform_location = gl::GetUniformLocation(
                            shader_program,
                            CString::new("frame_offset").unwrap().as_ptr(),
                        );
                        gl::Uniform1f(uniform_location, offset);

                        gl::BindVertexArray(vao);
                        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
                    }
                    gl_surface.swap_buffers(&gl_context).unwrap();
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    window.drag_window().unwrap();
                }
                _ => (),
            },
            Event::AboutToWait => {
                if last_update.elapsed() >= ANIMATION_SPEED {
                    current_frame = (current_frame + 1) % NUM_FRAMES;
                    last_update = Instant::now();
                    if current_frame == 0 {
                        active_texture_index = rng.random_range(0..textures.len());
                        rebind_texture = true;
                    }
                    window.request_redraw();
                } else {
                    let time_to_next_frame = ANIMATION_SPEED - last_update.elapsed();
                    elwt.set_control_flow(ControlFlow::WaitUntil(
                        Instant::now() + time_to_next_frame,
                    ));
                }
            }
            _ => (),
        }
    })?;

    Ok(())
}

// Helper function to compile a shader from source.
unsafe fn create_shader(shader_type: gl::types::GLenum, source: &str) -> gl::types::GLuint {
    let shader = unsafe { gl::CreateShader(shader_type) };
    let c_str = CString::new(source.as_bytes()).unwrap();
    unsafe {
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);
    }

    // You should add error checking here in a real application!
    // let mut success = gl::FALSE as gl::types::GLint;
    // gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);

    shader
}
