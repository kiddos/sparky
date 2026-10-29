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
uniform vec3 uKeyColor = vec3(0.0, 1.0, 0.0); // Pure green (#00FF00)
uniform float uSimilarity = 0.35;             // How much green to cut (0.30 - 0.45)
uniform float uSmoothness = 0.20;             // Edge softness (0.10 - 0.25)
uniform float uSpill      = 0.20;             // Green spill removal strength

// Convert RGB to Chrominance (Cb, Cr) so brightness/shadows don't affect the key
vec2 RGBtoUV(vec3 rgb) {
    return vec2(
        rgb.r * -0.169 + rgb.g * -0.331 + rgb.b *  0.500 + 0.5,
        rgb.r *  0.500 + rgb.g * -0.419 + rgb.b * -0.081 + 0.5
    );
}

void main() {
    vec4 rgba = texture(ourTexture, TexCoord);

    // 1. Measure color distance in UV chrominance space
    float chromaDist = distance(RGBtoUV(rgba.rgb), RGBtoUV(uKeyColor));

    // 2. Compute smooth alpha mask with smoothstep for softer, anti-aliased edges
    float alpha = smoothstep(uSimilarity, uSimilarity + max(uSmoothness, 0.0001), chromaDist);

    if (alpha <= 0.001)
        discard;

    // 3. Green Spill Suppression on edge pixels
    float spillFactor = smoothstep(uSimilarity, uSimilarity + max(uSpill, 0.0001), chromaDist);
    
    // Suppress excess green by clamping G to the max of R and B on keyed edges
    vec3 despilledRGB = rgba.rgb;
    float maxRB = max(rgba.r, rgba.b);
    despilledRGB.g = mix(min(rgba.g, maxRB), rgba.g, spillFactor);

    // Optional: Unpremultiply out the subtracted green brightness on semi-transparent edges
    float greenRemoved = rgba.g - despilledRGB.g;
    despilledRGB.rb += greenRemoved * 0.35;

    FragColor = vec4(clamp(despilledRGB, 0.0, 1.0), alpha);
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
