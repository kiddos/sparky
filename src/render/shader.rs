use anyhow::{Context, Result, bail};
use std::ffi::CString;
use std::fs;
use std::ptr;

pub struct ShaderProgram {
    id: u32,
}

impl ShaderProgram {
    /// Compiles and links shaders directly from in-memory string slices
    pub fn from_source(vert_src: &str, frag_src: &str) -> Result<Self> {
        let vert_shader = Self::compile_shader(vert_src, gl::VERTEX_SHADER, "embedded:quad.vert")?;
        let frag_shader =
            Self::compile_shader(frag_src, gl::FRAGMENT_SHADER, "embedded:chroma.frag")?;

        let id = unsafe { gl::CreateProgram() };
        unsafe {
            gl::AttachShader(id, vert_shader);
            gl::AttachShader(id, frag_shader);
            gl::LinkProgram(id);

            let mut success = 0;
            gl::GetProgramiv(id, gl::LINK_STATUS, &mut success);
            if success == 0 {
                let mut len = 0;
                gl::GetProgramiv(id, gl::INFO_LOG_LENGTH, &mut len);
                let mut buffer = vec![0u8; len as usize];
                gl::GetProgramInfoLog(id, len, ptr::null_mut(), buffer.as_mut_ptr() as *mut i8);
                let error_msg = String::from_utf8_lossy(&buffer);
                gl::DeleteShader(vert_shader);
                gl::DeleteShader(frag_shader);
                gl::DeleteProgram(id);
                bail!("Failed to link shader program: {}", error_msg);
            }

            gl::DeleteShader(vert_shader);
            gl::DeleteShader(frag_shader);
        }

        log::info!("Shader program successfully linked from source (Program ID: {id})");
        Ok(Self { id })
    }

    pub fn new(vert_path: &str, frag_path: &str) -> Result<Self> {
        log::info!("Compiling shader program (vertex: '{vert_path}', fragment: '{frag_path}')");
        let vert_src = fs::read_to_string(vert_path)
            .with_context(|| format!("Failed to read vertex shader from {}", vert_path))?;
        let frag_src = fs::read_to_string(frag_path)
            .with_context(|| format!("Failed to read fragment shader from {}", frag_path))?;

        let vert_shader = Self::compile_shader(&vert_src, gl::VERTEX_SHADER, vert_path)?;
        let frag_shader = Self::compile_shader(&frag_src, gl::FRAGMENT_SHADER, frag_path)?;

        let id = unsafe { gl::CreateProgram() };
        unsafe {
            gl::AttachShader(id, vert_shader);
            gl::AttachShader(id, frag_shader);
            gl::LinkProgram(id);

            let mut success = 0;
            gl::GetProgramiv(id, gl::LINK_STATUS, &mut success);
            if success == 0 {
                let mut len = 0;
                gl::GetProgramiv(id, gl::INFO_LOG_LENGTH, &mut len);
                let mut buffer = vec![0u8; len as usize];
                gl::GetProgramInfoLog(id, len, ptr::null_mut(), buffer.as_mut_ptr() as *mut i8);
                let error_msg = String::from_utf8_lossy(&buffer);
                gl::DeleteShader(vert_shader);
                gl::DeleteShader(frag_shader);
                gl::DeleteProgram(id);
                bail!("Failed to link shader program: {}", error_msg);
            }

            gl::DeleteShader(vert_shader);
            gl::DeleteShader(frag_shader);
        }

        log::info!("Shader program successfully linked (Program ID: {id})");
        Ok(Self { id })
    }

    fn compile_shader(src: &str, shader_type: u32, path: &str) -> Result<u32> {
        let shader = unsafe { gl::CreateShader(shader_type) };
        let c_src = CString::new(src.as_bytes())?;
        unsafe {
            gl::ShaderSource(shader, 1, &c_src.as_ptr(), ptr::null());
            gl::CompileShader(shader);

            let mut success = 0;
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut len = 0;
                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
                let mut buffer = vec![0u8; len as usize];
                gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buffer.as_mut_ptr() as *mut i8);
                let error_msg = String::from_utf8_lossy(&buffer);
                gl::DeleteShader(shader);
                bail!("Shader compilation failed for {}: {}", path, error_msg);
            }
        }
        Ok(shader)
    }

    pub fn use_program(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }

    pub fn set_int(&self, name: &str, value: i32) {
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                let loc = gl::GetUniformLocation(self.id, c_name.as_ptr());
                if loc >= 0 {
                    gl::Uniform1i(loc, value);
                }
            }
        }
    }

    pub fn set_float(&self, name: &str, value: f32) {
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                let loc = gl::GetUniformLocation(self.id, c_name.as_ptr());
                if loc >= 0 {
                    gl::Uniform1f(loc, value);
                }
            }
        }
    }

    pub fn set_vec2(&self, name: &str, x: f32, y: f32) {
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                let loc = gl::GetUniformLocation(self.id, c_name.as_ptr());
                if loc >= 0 {
                    gl::Uniform2f(loc, x, y);
                }
            }
        }
    }

    pub fn set_vec3(&self, name: &str, x: f32, y: f32, z: f32) {
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                let loc = gl::GetUniformLocation(self.id, c_name.as_ptr());
                if loc >= 0 {
                    gl::Uniform3f(loc, x, y, z);
                }
            }
        }
    }
}

impl Drop for ShaderProgram {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.id);
        }
    }
}
