use std::env;
use std::error::Error;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const DEFAULT_SPRITE_WIDTH: u32 = 128;
pub const DEFAULT_SPRITE_HEIGHT: u32 = 128;
pub const DEFAULT_ANIMATION_SPEED: Duration = Duration::from_millis(42);

#[derive(Clone)]
pub struct SpriteAnimation {
    #[allow(dead_code)]
    pub name: String,
    pub frames: Vec<u32>,
}

pub fn find_sprites_dir() -> PathBuf {
    // 1. Current working directory
    let cwd_sprites = PathBuf::from("sprites");
    if cwd_sprites.is_dir() {
        return cwd_sprites;
    }

    // 2. Relative to executable location
    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        let exe_sprites = parent.join("sprites");
        if exe_sprites.is_dir() {
            return exe_sprites;
        }
        let target_sprites = parent.join("../../sprites");
        if target_sprites.is_dir() {
            return target_sprites;
        }
    }

    // 3. Compile-time CARGO_MANIFEST_DIR fallback
    let manifest_sprites = Path::new(env!("CARGO_MANIFEST_DIR")).join("sprites");
    if manifest_sprites.is_dir() {
        return manifest_sprites;
    }

    cwd_sprites
}

pub fn load_texture_from_file(path: &Path) -> Result<u32, Box<dyn Error>> {
    let image = image::open(path)?.to_rgba8();
    let (img_width, img_height) = image.dimensions();
    let image_data = image.into_raw();

    let mut texture_id = 0;
    unsafe {
        gl::GenTextures(1, &mut texture_id);
        gl::BindTexture(gl::TEXTURE_2D, texture_id);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
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
