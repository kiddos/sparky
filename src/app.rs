use std::error::Error;
use std::ffi::{CString, c_void};
use std::path::Path;

use glfw::Context;
use log::info;
use rand::rngs::ThreadRng;

use crate::brain::{BrainEvent, BrainHandle};
use crate::chat::ChatWindow;
use crate::shader::{FS_SRC, VS_SRC, create_shader};
use crate::util::{
    calculate_drag_offset, calculate_drag_target, get_global_cursor_pos, weighted_random_index,
};
use crate::video::{VideoAnimation, VideoStream, find_videos_dir, load_video_animations};

pub enum PlaybackMode {
    None,
    Video {
        animations: Vec<VideoAnimation>,
        active_stream: Option<VideoStream>,
        frame_buf: Vec<u8>,
        texture_id: u32,
    }
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
    pub rng: ThreadRng,
    pub is_dragging: bool,
    pub drag_start_pos: (f64, f64),
    pub drag_offset: (i32, i32),
    pub window_pos: (i32, i32),
    pub brain: Option<BrainHandle>,
    pub chat_window: Option<ChatWindow>,
    pub queued_animation: Option<usize>,
    pub current_dialogue: Option<String>,
    pub current_emotion: Option<String>,
}

impl App {
    pub fn new(window: glfw::PWindow, width: u32, height: u32, black_threshold: f32) -> Self {
        let window_pos = window.get_pos();
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
            rng: rand::rng(),
            is_dragging: false,
            drag_start_pos: (0.0, 0.0),
            drag_offset: (0, 0),
            window_pos,
            brain: None,
            chat_window: None,
            queued_animation: None,
            current_dialogue: None,
            current_emotion: None,
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
                    let prev_anim_name = animations.get(self.active_animation_index).map(|a| a.name.clone());
                    if let Some(name) = prev_anim_name {
                        if let Some(ref brain) = self.brain {
                            brain.send_event(BrainEvent::AnimationCompleted { name });
                        }
                    }

                    self.active_animation_index = if let Some(queued) = self.queued_animation.take() {
                        queued
                    } else {
                        weighted_random_index(&self.select_prob, &mut self.rng)
                    };
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
            PlaybackMode::None => {}
        }
    }

    pub fn set_brain(&mut self, brain: BrainHandle) {
        self.brain = Some(brain);
    }

    pub fn set_chat_window(&mut self, chat_window: ChatWindow) {
        self.chat_window = Some(chat_window);
    }

    pub fn set_window_pos(&mut self, x: i32, y: i32) {
        self.window.set_pos(x, y);
        self.window_pos = (x, y);
    }

    pub fn start_dragging(&mut self, click_x: f64, click_y: f64) {
        self.is_dragging = true;
        self.drag_start_pos = (click_x, click_y);
        let (wx, wy) = self.window.get_pos();
        self.window_pos = (wx, wy);

        if let Some((gx, gy)) = get_global_cursor_pos() {
            self.drag_offset = calculate_drag_offset((gx, gy), self.window_pos);
        } else {
            self.drag_offset = (click_x.round() as i32, click_y.round() as i32);
        }
    }

    pub fn stop_dragging(&mut self) {
        self.is_dragging = false;
    }

    pub fn update_drag(&mut self, screen_h: i32) -> Option<(f64, f64)> {
        if !self.is_dragging {
            return None;
        }

        let (target_x, target_y) = if let Some((gx, gy)) = get_global_cursor_pos() {
            calculate_drag_target((gx, gy), self.drag_offset)
        } else {
            let (cx, cy) = self.window.get_cursor_pos();
            let dx = cx - self.drag_start_pos.0;
            let dy = cy - self.drag_start_pos.1;
            (
                self.window_pos.0 + dx.round() as i32,
                self.window_pos.1 + dy.round() as i32,
            )
        };

        if (target_x, target_y) != self.window_pos {
            let dx = (target_x - self.window_pos.0) as f64;
            let dy = (target_y - self.window_pos.1) as f64;
            self.set_window_pos(target_x, target_y);
            self.update_chat_position(screen_h);
            Some((dx, dy))
        } else {
            None
        }
    }

    pub fn toggle_chat(&mut self, screen_h: i32) {
        let (wx, wy) = self.window_pos;
        if let Some(ref mut chat) = self.chat_window {
            chat.toggle_visibility(wx, wy, self.width, self.height, screen_h);
        }
    }

    pub fn set_chat_visible(&mut self, show: bool, screen_h: i32) {
        let (wx, wy) = self.window_pos;
        if let Some(ref mut chat) = self.chat_window {
            chat.set_visible(show, wx, wy, self.width, self.height, screen_h);
        }
    }

    pub fn update_chat_position(&self, screen_h: i32) {
        let (wx, wy) = self.window_pos;
        if let Some(ref chat) = self.chat_window {
            if chat.is_visible() {
                chat.update_position(wx, wy, self.width, self.height, screen_h);
            }
        }
    }

    pub fn animation_names(&self) -> Vec<String> {
        match &self.mode {
            PlaybackMode::Video { animations, .. } => {
                animations.iter().map(|a| a.name.clone()).collect()
            }
            PlaybackMode::None => Vec::new(),
        }
    }

    pub fn find_animation_index(&self, name: &str) -> Option<usize> {
        match &self.mode {
            PlaybackMode::Video { animations, .. } => {
                animations.iter().position(|a| a.name.eq_ignore_ascii_case(name))
            }
            PlaybackMode::None => None,
        }
    }

    pub fn update_select_prob_for(&mut self, target_idx: usize) {
        if target_idx < self.select_prob.len() {
            for (i, p) in self.select_prob.iter_mut().enumerate() {
                if i == target_idx {
                    *p = 6.0;
                } else {
                    *p = 1.0;
                }
            }
        }
    }

    pub fn poll_brain(&mut self) -> Option<(String, String)> {
        let mut decisions = Vec::new();
        if let Some(ref brain) = self.brain {
            while let Some(decision) = brain.try_recv_decision() {
                decisions.push(decision);
            }
        }

        let mut latest = None;
        let names = self.animation_names();
        for decision in decisions {
            let resolved = decision.resolve_animation_name(&names);
            info!(
                "[Sparky Brain] Gemma 4 transition: emotion='{}' (video: {:?}) | \"{}\"",
                decision.target_emotion, resolved, decision.dialogue
            );

            self.current_dialogue = Some(decision.dialogue.clone());
            self.current_emotion = Some(decision.target_emotion.clone());

            if let Some(ref anim_name) = resolved {
                if let Some(idx) = self.find_animation_index(anim_name) {
                    self.queued_animation = Some(idx);
                    self.update_select_prob_for(idx);
                }
            }

            if let Some(ref chat) = self.chat_window {
                chat.set_status(&format!("[{}]", decision.target_emotion));
                chat.add_sparky_message(&decision.target_emotion, &decision.dialogue);
            }

            latest = Some((decision.target_emotion, decision.dialogue));
        }
        latest
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
            PlaybackMode::None => true,
        }
    }
}
