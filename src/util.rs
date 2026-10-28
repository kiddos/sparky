use std::env;
use std::fs::File;

use log::LevelFilter;
use rand::Rng;
use rand::rngs::ThreadRng;
use simplelog::{CombinedLogger, Config as LogConfig, WriteLogger};

pub fn init_log() {
    let temp_dir = env::temp_dir();
    let log_file_path = temp_dir.join("sparky.log");
    if let Ok(log_file) = File::create(&log_file_path) {
        let _ = CombinedLogger::init(vec![WriteLogger::new(
            LevelFilter::Info,
            LogConfig::default(),
            log_file,
        )]);
    }
}

pub fn weighted_random_index(weights: &[f32], rng: &mut ThreadRng) -> usize {
    if weights.is_empty() {
        return 0;
    }

    let total_weight: f32 = weights.iter().sum();
    if total_weight.is_nan() || total_weight.is_infinite() || total_weight < f32::EPSILON {
        return 0;
    }

    let random_value = rng.random_range(0.0..total_weight);
    let mut cumulative_sum: f32 = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative_sum += weight;
        if random_value < cumulative_sum {
            return index;
        }
    }
    0
}

/// Calculates the offset between the global cursor and the window's top-left position.
pub fn calculate_drag_offset(global_cursor: (i32, i32), window_pos: (i32, i32)) -> (i32, i32) {
    (global_cursor.0 - window_pos.0, global_cursor.1 - window_pos.1)
}

/// Calculates the target window position given the current global cursor position
/// and the initial drag offset.
pub fn calculate_drag_target(global_cursor: (i32, i32), drag_offset: (i32, i32)) -> (i32, i32) {
    (global_cursor.0 - drag_offset.0, global_cursor.1 - drag_offset.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_random_index() {
        let mut rng = rand::rng();
        assert_eq!(weighted_random_index(&[], &mut rng), 0);
        assert_eq!(weighted_random_index(&[0.0, 0.0], &mut rng), 0);
        assert_eq!(weighted_random_index(&[10.0], &mut rng), 0);
        assert_eq!(weighted_random_index(&[0.0, 10.0], &mut rng), 1);
    }

    #[test]
    fn test_calculate_drag_offset_and_target() {
        let window_pos = (500, 300);
        let global_cursor = (550, 360);
        let offset = calculate_drag_offset(global_cursor, window_pos);
        assert_eq!(offset, (50, 60));

        // When cursor is at initial position, target equals initial window pos
        let target = calculate_drag_target(global_cursor, offset);
        assert_eq!(target, window_pos);

        // When cursor moves by +20, -15
        let moved_cursor = (570, 345);
        let new_target = calculate_drag_target(moved_cursor, offset);
        assert_eq!(new_target, (520, 285));
        assert_eq!(new_target.0 - window_pos.0, 20);
        assert_eq!(new_target.1 - window_pos.1, -15);
    }

    #[test]
    fn test_calculate_drag_multi_monitor_negative_coords() {
        let window_pos = (-1920, 100);
        let global_cursor = (-1800, 150);
        let offset = calculate_drag_offset(global_cursor, window_pos);
        assert_eq!(offset, (120, 50));

        let moved_cursor = (-1750, 200);
        let new_target = calculate_drag_target(moved_cursor, offset);
        assert_eq!(new_target, (-1870, 150));
    }
}

#[cfg(target_os = "linux")]
pub fn get_global_cursor_pos() -> Option<(i32, i32)> {
    if !gtk::is_initialized_main_thread() {
        return None;
    }
    use gtk::gdk::prelude::*;
    let display = gtk::gdk::Display::default()?;
    let seat = display.default_seat()?;
    let pointer = seat.pointer()?;
    let (_screen, x, y) = pointer.position();
    Some((x, y))
}

#[cfg(target_os = "windows")]
pub fn get_global_cursor_pos() -> Option<(i32, i32)> {
    #[repr(C)]
    struct POINT {
        x: i32,
        y: i32,
    }
    unsafe extern "system" {
        fn GetCursorPos(lpPoint: *mut POINT) -> i32;
    }
    let mut pt = POINT { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut pt) } != 0 {
        Some((pt.x, pt.y))
    } else {
        None
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn get_global_cursor_pos() -> Option<(i32, i32)> {
    None
}
