use std::ffi::CString;

pub const VS_SRC: &str = r#"
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main() {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
"#;

pub const FS_SRC: &str = r#"
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;

    uniform sampler2D ourTexture;
    uniform float uBlackThreshold;

    void main() {
        vec4 texColor = texture(ourTexture, TexCoord);

        // Discard transparent pixels. For video playback the alpha channel holds
        // a background mask computed on the CPU (flood fill of dark pixels that
        // are connected to the frame border), so dark parts *inside* Sparky
        // (pupils, outlines, shadows) keep alpha = 1 and are not keyed out.
        if (texColor.a < 0.1)
            discard;

        if (uBlackThreshold > 0.0) {
            // Only soften pixels that sit right next to the background, to hide
            // harsh compression borders without eating into dark interior detail.
            vec2 texel = 1.0 / vec2(textureSize(ourTexture, 0));
            float minNeighborA = min(
                min(texture(ourTexture, TexCoord + vec2(texel.x, 0.0)).a,
                    texture(ourTexture, TexCoord - vec2(texel.x, 0.0)).a),
                min(texture(ourTexture, TexCoord + vec2(0.0, texel.y)).a,
                    texture(ourTexture, TexCoord - vec2(0.0, texel.y)).a));

            float alphaFactor = 1.0;
            if (minNeighborA < 0.5) {
                float maxVal = max(texColor.r, max(texColor.g, texColor.b));
                float smoothRange = 0.08;
                alphaFactor = clamp((maxVal - uBlackThreshold) / smoothRange, 0.0, 1.0);
            }
            FragColor = vec4(texColor.rgb, texColor.a * alphaFactor);
        } else {
            FragColor = texColor;
        }
    }
"#;

pub unsafe fn create_shader(shader_type: gl::types::GLenum, source: &str) -> gl::types::GLuint {
    let shader = unsafe { gl::CreateShader(shader_type) };
    let c_str = CString::new(source.as_bytes()).unwrap();
    unsafe {
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);

        // Check shader compile status
        let mut success = gl::FALSE as gl::types::GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as gl::types::GLint {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);

            let error_message = if len > 0 {
                let mut buffer = vec![0u8; len as usize];
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
            log::error!("Shader compilation error: {}", error_message);
        }
    }

    shader
}
