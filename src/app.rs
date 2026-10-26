use std::error::Error;
use std::ffi::{CString, c_void};
use std::path::Path;

use glfw::Context;
use log::info;
use rand::rngs::ThreadRng;

use crate::shader::{FS_SRC, VS_SRC, create_shader};
use crate::sprite::{SpriteAnimation, find_sprites_dir, load_texture_from_file};
use crate::util::weighted_random_index;
use crate::video::{VideoAnimation, VideoStream, find_videos_dir, load_video_animations};

pub enum PlaybackMode {
    None,
    Video {
        animations: Vec<VideoAnimation>,
        active_stream: Option<VideoStream>,
        frame_buf: Vec<u8>,
        texture_id: u32,
    },
    Sprite {
        animations: Vec<SpriteAnimation>,
    },
}

pub struct App {
    pub window: glfw::PWindow,
    pub shader_program: u32,
    pub u_black_threshold_loc: i32,
    pub vao: u32,
    #[allow(dead_code)]
    pub vbo: u32,
    #[allow(dead_code)]
    pub ebo: u32,
    pub width: u32,
    pub height: u32,
    pub black_threshold: f32,
    pub mode: PlaybackMode,
    pub select_prob: Vec<f32>,
    pub active_animation_index: usize,
    pub current_frame: usize,
    pub rng: ThreadRng,
    pub is_dragging: bool,
    pub drag_start_pos: (f64, f64),
}

impl App {
    pub fn new(window: glfw::PWindow, width: u32, height: u32, black_threshold: f32) -> Self {
        Self {
            window,
            shader_program: 0,
            u_black_threshold_loc: -1,
            vao: 0,
            vbo: 0,
            ebo: 0,
            width,
            height,
            black_threshold,
            mode: PlaybackMode::None,
            select_prob: Vec::new(),
            active_animation_index: 0,
            current_frame: 0,
            rng: rand::rng(),
            is_dragging: false,
            drag_start_pos: (0.0, 0.0),
        }
    }

    pub fn init_opengl(&mut self) {
        self.shader_program = unsafe {
            let vertex_shader = create_shader(gl::VERTEX_SHADER, VS_SRC);
            let fragment_shader = create_shader(gl::FRAGMENT_SHADER, FS_SRC);

            let program = gl::CreateProgram();
            gl::AttachShader(program, vertex_shader);
            gl::AttachShader(program, fragment_shader);
            gl::LinkProgram(program);

            gl::DeleteShader(vertex_shader);
            gl::DeleteShader(fragment_shader);
            program
        };

        let u_name = CString::new("uBlackThreshold").unwrap();
        self.u_black_threshold_loc =
            unsafe { gl::GetUniformLocation(self.shader_program, u_name.as_ptr()) };

        let vertices: [f32; 16] = [
            // positions   // texture coords
            1.0, 1.0, 1.0, 0.0, // top right
            1.0, -1.0, 1.0, 1.0, // bottom right
            -1.0, -1.0, 0.0, 1.0, // bottom left
            -1.0, 1.0, 0.0, 0.0, // top left
        ];
        let indices: [u32; 6] = [0, 1, 3, 1, 2, 3];

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

            gl::Viewport(0, 0, self.width as i32, self.height as i32);
        }
        self.vao = vao;
        self.vbo = vbo;
        self.ebo = ebo;
    }

    pub fn load_animations(&mut self, cli_video_dir: Option<&Path>) -> Result<(), Box<dyn Error>> {
        // 1. Try to load videos directly
        if let Some(video_dir) = find_videos_dir(cli_video_dir) {
            let videos = load_video_animations(&video_dir);
            if !videos.is_empty() {
                info!(
                    "Found {} video animation(s) in {:?}",
                    videos.len(),
                    video_dir
                );
                for v in &videos {
                    info!("  - Animation: '{}' ({})", v.name, v.path.display());
                }

                let mut texture_id = 0;
                unsafe {
                    gl::GenTextures(1, &mut texture_id);
                    gl::BindTexture(gl::TEXTURE_2D, texture_id);
                    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
                    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
                    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
                    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

                    let initial_data = vec![0u8; (self.width * self.height * 4) as usize];
                    gl::TexImage2D(
                        gl::TEXTURE_2D,
                        0,
                        gl::RGBA as i32,
                        self.width as i32,
                        self.height as i32,
                        0,
                        gl::RGBA,
                        gl::UNSIGNED_BYTE,
                        initial_data.as_ptr() as *const c_void,
                    );
                }

                let frame_buf = vec![0u8; (self.width * self.height * 4) as usize];
                self.select_prob = vec![1.0; videos.len()];
                self.active_animation_index = 0;
                self.mode = PlaybackMode::Video {
                    animations: videos,
                    active_stream: None,
                    frame_buf,
                    texture_id,
                };

                self.start_current_video_stream()?;
                return Ok(());
            }
        }

        // 2. Fallback to legacy sprite directories
        self.load_legacy_sprites()?;
        Ok(())
    }

    pub fn start_current_video_stream(&mut self) -> Result<(), Box<dyn Error>> {
        if let PlaybackMode::Video {
            animations,
            active_stream,
            frame_buf,
            texture_id,
        } = &mut self.mode
        {
            if animations.is_empty() {
                return Ok(());
            }
            *active_stream = None;

            let anim = &animations[self.active_animation_index];
            info!("Starting playback of video animation: '{}'", anim.name);

            let mut stream =
                VideoStream::new(&anim.path, self.width, self.height, self.black_threshold)?;
            if let Ok(true) = stream.read_frame(frame_buf) {
                unsafe {
                    gl::BindTexture(gl::TEXTURE_2D, *texture_id);
                    gl::TexSubImage2D(
                        gl::TEXTURE_2D,
                        0,
                        0,
                        0,
                        self.width as i32,
                        self.height as i32,
                        gl::RGBA,
                        gl::UNSIGNED_BYTE,
                        frame_buf.as_ptr() as *const c_void,
                    );
                }
            }
            *active_stream = Some(stream);
        }
        Ok(())
    }

    pub fn load_legacy_sprites(&mut self) -> Result<(), Box<dyn Error>> {
        let sprites_dir = find_sprites_dir();
        info!("Loading legacy sprites from: {:?}", sprites_dir);

        let mut animations = Vec::new();

        if sprites_dir.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(&sprites_dir)
                .map_err(|e| {
                    format!(
                        "Failed to read sprites directory at {:?}: {}",
                        sprites_dir, e
                    )
                })?
                .filter_map(|e| e.ok())
                .collect();
            entries.sort_by_key(|e| e.path());

            for entry in entries {
                let path = entry.path();
                if path.is_dir() {
                    let dir_name = match path.file_name().and_then(|n| n.to_str()) {
                        Some(name) if !name.starts_with('.') => name.to_string(),
                        _ => continue,
                    };

                    let mut frame_files: Vec<_> = std::fs::read_dir(&path)?
                        .filter_map(|e| e.ok())
                        .filter(|e| {
                            e.path()
                                .extension()
                                .and_then(|ext| ext.to_str())
                                .map(|ext| ext.eq_ignore_ascii_case("png"))
                                .unwrap_or(false)
                        })
                        .collect();

                    frame_files.sort_by_key(|e| e.file_name());

                    if frame_files.is_empty() {
                        continue;
                    }

                    let mut frames = Vec::with_capacity(frame_files.len());
                    for file in frame_files {
                        let texture_id = load_texture_from_file(&file.path())?;
                        frames.push(texture_id);
                    }

                    info!(
                        "Loaded animation '{}' with {} frames",
                        dir_name,
                        frames.len()
                    );
                    animations.push(SpriteAnimation {
                        name: dir_name,
                        frames,
                    });
                }
            }

            if animations.is_empty() {
                let mut frame_files: Vec<_> = std::fs::read_dir(&sprites_dir)?
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("png"))
                            .unwrap_or(false)
                    })
                    .collect();
                frame_files.sort_by_key(|e| e.file_name());

                if !frame_files.is_empty() {
                    let mut frames = Vec::with_capacity(frame_files.len());
                    for file in frame_files {
                        let texture_id = load_texture_from_file(&file.path())?;
                        frames.push(texture_id);
                    }
                    info!("Loaded default animation with {} frames", frames.len());
                    animations.push(SpriteAnimation {
                        name: "default".to_string(),
                        frames,
                    });
                }
            }
        }

        if animations.is_empty() {
            return Err("No videos or sprite frames found to play.".into());
        }

        self.select_prob = vec![1.0; animations.len()];
        self.active_animation_index = 0;
        self.current_frame = 0;
        self.mode = PlaybackMode::Sprite { animations };
        Ok(())
    }

    pub fn update(&mut self) {
        match &mut self.mode {
            PlaybackMode::Video {
                animations,
                active_stream,
                frame_buf,
                texture_id,
            } => {
                if animations.is_empty() {
                    return;
                }

                let mut finished = false;
                if let Some(stream) = active_stream {
                    match stream.read_frame(frame_buf) {
                        Ok(true) => unsafe {
                            gl::BindTexture(gl::TEXTURE_2D, *texture_id);
                            gl::TexSubImage2D(
                                gl::TEXTURE_2D,
                                0,
                                0,
                                0,
                                self.width as i32,
                                self.height as i32,
                                gl::RGBA,
                                gl::UNSIGNED_BYTE,
                                frame_buf.as_ptr() as *const c_void,
                            );
                        },
                        Ok(false) => {
                            finished = true;
                        }
                        Err(e) => {
                            log::warn!("Video stream error: {}", e);
                            finished = true;
                        }
                    }
                } else {
                    finished = true;
                }

                if finished {
                    *active_stream = None;
                    self.active_animation_index =
                        weighted_random_index(&self.select_prob, &mut self.rng);
                    let next_anim = &animations[self.active_animation_index];
                    info!("Switching to video animation: '{}'", next_anim.name);

                    match VideoStream::new(
                        &next_anim.path,
                        self.width,
                        self.height,
                        self.black_threshold,
                    ) {
                        Ok(mut new_stream) => {
                            if let Ok(true) = new_stream.read_frame(frame_buf) {
                                unsafe {
                                    gl::BindTexture(gl::TEXTURE_2D, *texture_id);
                                    gl::TexSubImage2D(
                                        gl::TEXTURE_2D,
                                        0,
                                        0,
                                        0,
                                        self.width as i32,
                                        self.height as i32,
                                        gl::RGBA,
                                        gl::UNSIGNED_BYTE,
                                        frame_buf.as_ptr() as *const c_void,
                                    );
                                }
                            }
                            *active_stream = Some(new_stream);
                        }
                        Err(e) => {
                            log::error!("Failed to open video {}: {}", next_anim.path.display(), e);
                        }
                    }
                }
            }
            PlaybackMode::Sprite { animations } => {
                if animations.is_empty() {
                    return;
                }
                let current_anim = &animations[self.active_animation_index];
                self.current_frame = (self.current_frame + 1) % current_anim.frames.len();
                if self.current_frame == 0 {
                    self.active_animation_index =
                        weighted_random_index(&self.select_prob, &mut self.rng);
                }
            }
            PlaybackMode::None => {}
        }
    }

    pub fn render(&mut self) {
        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::UseProgram(self.shader_program);

            // Pass black threshold uniform to shader
            if self.u_black_threshold_loc >= 0 {
                gl::Uniform1f(self.u_black_threshold_loc, self.black_threshold);
            }

            match &self.mode {
                PlaybackMode::Video { texture_id, .. } => {
                    gl::BindTexture(gl::TEXTURE_2D, *texture_id);
                }
                PlaybackMode::Sprite { animations } => {
                    if !animations.is_empty() {
                        let current_anim = &animations[self.active_animation_index];
                        let texture_id = current_anim.frames[self.current_frame];
                        gl::BindTexture(gl::TEXTURE_2D, texture_id);
                    }
                }
                PlaybackMode::None => {}
            }

            gl::BindVertexArray(self.vao);
            gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
        }
        self.window.swap_buffers();
    }

    pub fn is_pixel_opaque(&self, x: f64, y: f64) -> bool {
        if x < 0.0 || y < 0.0 {
            return false;
        }
        let px = x as u32;
        let py = y as u32;
        if px >= self.width || py >= self.height {
            return false;
        }

        match &self.mode {
            PlaybackMode::Video { frame_buf, .. } => {
                let idx = ((py * self.width + px) * 4) as usize;
                if idx + 3 < frame_buf.len() {
                    // Alpha holds the background mask computed in read_frame.
                    return frame_buf[idx + 3] >= 25;
                }
                true
            }
            PlaybackMode::Sprite { .. } | PlaybackMode::None => true,
        }
    }
}
