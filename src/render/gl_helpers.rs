#[allow(dead_code)]
pub fn check_gl_error(op: &str) {
    unsafe {
        let mut err = gl::GetError();
        while err != gl::NO_ERROR {
            log::error!("OpenGL error after {}: 0x{:X}", op, err);
            err = gl::GetError();
        }
    }
}
