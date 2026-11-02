use std::time::{Duration, Instant};

pub struct FramePacer {
    #[allow(dead_code)]
    target_fps: f64,
    frame_duration: Duration,
    last_frame_instant: Instant,
    current_frame_index: usize,
}

impl FramePacer {
    pub fn new(fps: f64) -> Self {
        Self {
            target_fps: fps,
            frame_duration: Duration::from_secs_f64(1.0 / fps),
            last_frame_instant: Instant::now(),
            current_frame_index: 0,
        }
    }

    /// Returns the duration until the next frame must be displayed.
    pub fn time_until_next_frame(&self) -> Duration {
        let elapsed = self.last_frame_instant.elapsed();
        if elapsed >= self.frame_duration {
            Duration::ZERO
        } else {
            self.frame_duration - elapsed
        }
    }

    /// Advance to the next frame if the frame interval has elapsed.
    pub fn advance_frame(&mut self, total_frames: usize) -> bool {
        if total_frames == 0 {
            return false;
        }
        let elapsed = self.last_frame_instant.elapsed();
        if elapsed >= self.frame_duration {
            self.last_frame_instant = Instant::now();
            self.current_frame_index = (self.current_frame_index + 1) % total_frames;
            true
        } else {
            false
        }
    }

    pub fn current_index(&self) -> usize {
        self.current_frame_index
    }

    pub fn reset(&mut self) {
        self.current_frame_index = 0;
        self.last_frame_instant = Instant::now();
    }

    #[allow(dead_code)]
    pub fn target_fps(&self) -> f64 {
        self.target_fps
    }

    #[allow(dead_code)]
    pub fn set_target_fps(&mut self, fps: f64) {
        self.target_fps = fps;
        self.frame_duration = Duration::from_secs_f64(1.0 / fps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_pacer_pacing_and_wrapping() {
        let mut pacer = FramePacer::new(24.0);
        assert_eq!(pacer.current_index(), 0);

        // Simulate elapsed time
        pacer.last_frame_instant = Instant::now() - Duration::from_millis(100);
        let advanced = pacer.advance_frame(10);
        assert!(advanced);
        assert_eq!(pacer.current_index(), 1);

        pacer.reset();
        assert_eq!(pacer.current_index(), 0);
    }
}
