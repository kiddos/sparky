use std::ffi::c_void;
use std::mem;

pub struct RenderQuad {
    vao: u32,
    vbo: u32,
    texture_id: u32,
}

impl RenderQuad {
    pub fn new() -> Self {
        let vertices: [f32; 16] = [
            // pos(x,y)   // tex(u,v)
            -1.0, -1.0,   0.0, 0.0,
             1.0, -1.0,   1.0, 0.0,
            -1.0,  1.0,   0.0, 1.0,
             1.0,  1.0,   1.0, 1.0,
        ];

        let mut vao = 0;
        let mut vbo = 0;
        let mut texture_id = 0;

        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);

            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * mem::size_of::<f32>()) as isize,
                vertices.as_ptr() as *const c_void,
                gl::STATIC_DRAW,
            );

            // Position attribute (location = 0)
            gl::VertexAttribPointer(
                0,
                2,
                gl::FLOAT,
                gl::FALSE,
                (4 * mem::size_of::<f32>()) as i32,
                std::ptr::null(),
            );
            gl::EnableVertexAttribArray(0);

            // TexCoord attribute (location = 1)
            gl::VertexAttribPointer(
                1,
                2,
                gl::FLOAT,
                gl::FALSE,
                (4 * mem::size_of::<f32>()) as i32,
                (2 * mem::size_of::<f32>()) as *const c_void,
            );
            gl::EnableVertexAttribArray(1);

            // Setup dynamic OpenGL texture
            gl::GenTextures(1, &mut texture_id);
            gl::BindTexture(gl::TEXTURE_2D, texture_id);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

            gl::BindVertexArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        }

        log::info!("Allocated textured render quad (VAO: {vao}, VBO: {vbo}, Texture ID: {texture_id})");

        Self {
            vao,
            vbo,
            texture_id,
        }
    }

    pub fn update_texture(&self, width: u32, height: u32, rgb_data: &[u8]) {
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, self.texture_id);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGB as i32,
                width as i32,
                height as i32,
                0,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                rgb_data.as_ptr() as *const c_void,
            );
        }
    }

    pub fn draw(&self) {
        unsafe {
            gl::BindVertexArray(self.vao);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, self.texture_id);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
            gl::BindVertexArray(0);
        }
    }
}

impl Drop for RenderQuad {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.texture_id);
            gl::DeleteBuffers(1, &self.vbo);
            gl::DeleteVertexArrays(1, &self.vao);
        }
    }
}
