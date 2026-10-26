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
}
