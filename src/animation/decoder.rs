use image::codecs::webp::WebPDecoder;
use image::{AnimationDecoder, DynamicImage};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct Frame {
    pub rgb_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub duration_ms: u32,
}

pub struct AnimationClip {
    #[allow(dead_code)]
    pub name: String,
    pub frames: Vec<Frame>,
}

impl AnimationClip {
    pub fn load_from_webp<P: AsRef<Path>>(name: &str, path: P) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let decoder = WebPDecoder::new(reader)?;

        let mut frames = Vec::new();
        for frame_res in decoder.into_frames() {
            let frame = frame_res?;
            let delay_ms = frame.delay().numer_denom_ms().0;
            let dyn_img = DynamicImage::ImageRgba8(frame.into_buffer());
            let rgb_img = dyn_img.to_rgb8();

            frames.push(Frame {
                rgb_data: rgb_img.into_raw(),
                width: dyn_img.width(),
                height: dyn_img.height(),
                duration_ms: if delay_ms == 0 { 41 } else { delay_ms }, // Default 24fps ~ 41ms
            });
        }

        log::info!("Loaded animation '{}' with {} frames", name, frames.len());
        Ok(Self {
            name: name.to_string(),
            frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_webp_animation() {
        if Path::new("assets/animations/sit.webp").exists() {
            let clip = AnimationClip::load_from_webp("sit", "assets/animations/sit.webp")
                .expect("Failed to load sit.webp");
            assert_eq!(clip.name, "sit");
            assert!(!clip.frames.is_empty());
            assert_eq!(clip.frames[0].width, 384);
            assert_eq!(clip.frames[0].height, 216);
            assert_eq!(clip.frames[0].rgb_data.len(), (384 * 216 * 3) as usize);
        }
    }
}
