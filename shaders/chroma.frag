#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D u_Texture;
uniform vec3 u_KeyColorRGB;   // Target green: vec3(0.0, 1.0, 0.0) or tuned
uniform float u_Threshold;    // 0.32
uniform float u_Softness;     // 0.12
uniform float u_Despill;      // 0.85
uniform vec3 u_EmotionTint;   // Emotional mood tint: vec3(1.0) by default, [0.8, 0.85, 1.05] for sad

vec2 rgbToCbCr(vec3 rgb) {
    float cb = -0.168736 * rgb.r - 0.331264 * rgb.g + 0.500000 * rgb.b + 0.5;
    float cr =  0.500000 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b + 0.5;
    return vec2(cb, cr);
}

void main() {
    vec4 texColor = texture(u_Texture, TexCoord);
    vec2 pixelCbCr = rgbToCbCr(texColor.rgb);
    vec2 keyCbCr = rgbToCbCr(u_KeyColorRGB);

    float dist = distance(pixelCbCr, keyCbCr);
    float alpha = smoothstep(u_Threshold, u_Threshold + u_Softness, dist);

    // Despill: Suppress green reflected bleed on fur
    float maxRB = max(texColor.r, texColor.b);
    vec3 cleanRGB = texColor.rgb;
    if (cleanRGB.g > maxRB) {
        cleanRGB.g = mix(cleanRGB.g, maxRB, u_Despill);
    }

    // Apply emotion tint (e.g. melancholic blue for sad, warm golden for smirk)
    vec3 tintedRGB = cleanRGB * u_EmotionTint;

    FragColor = vec4(tintedRGB, alpha * texColor.a);
}
