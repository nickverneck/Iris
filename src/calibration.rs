use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalibrationProfile {
    // 3x3 Grid (9 points)
    pub top_left: Point,
    pub top_center: Point,
    pub top_right: Point,
    pub middle_left: Point,
    pub center: Point,
    pub middle_right: Point,
    pub bottom_left: Point,
    pub bottom_center: Point,
    pub bottom_right: Point,
}

impl CalibrationProfile {
    pub fn default() -> Self {
        // Fallback default values if no calibration exists
        CalibrationProfile {
            top_left: Point { x: 0.3, y: 0.3 },
            top_center: Point { x: 0.5, y: 0.3 },
            top_right: Point { x: 0.7, y: 0.3 },
            middle_left: Point { x: 0.3, y: 0.5 },
            center: Point { x: 0.5, y: 0.5 },
            middle_right: Point { x: 0.7, y: 0.5 },
            bottom_left: Point { x: 0.3, y: 0.7 },
            bottom_center: Point { x: 0.5, y: 0.7 },
            bottom_right: Point { x: 0.7, y: 0.7 },
        }
    }

    pub fn load(path: &str) -> Self {
        if Path::new(path).exists() {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(profile) = serde_json::from_str(&content) {
                    return profile;
                }
            }
        }
        Self::default()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    // Map raw iris coordinates to normalized screen coordinates (0.0 to 1.0)
    // using bilinear interpolation across a 3x3 grid.
    pub fn map(&self, raw: &Point) -> Point {
        // Helper for 1D linear interpolation
        fn lerp(a: f64, b: f64, t: f64) -> f64 {
            a + t * (b - a)
        }
        
        // Helper to get t-value (0-1) for where raw falls between two calibration points
        fn get_t(raw_val: f64, cal_low: f64, cal_high: f64) -> f64 {
            if (cal_high - cal_low).abs() < 0.001 {
                return 0.5;
            }
            ((raw_val - cal_low) / (cal_high - cal_low)).clamp(0.0, 1.0)
        }

        // Determine which row (top/middle/bottom) based on Y
        let (row_low, row_high, ty) = if raw.y < self.center.y {
            // Top half: between top row and middle row
            let top_avg_y = (self.top_left.y + self.top_center.y + self.top_right.y) / 3.0;
            let mid_avg_y = (self.middle_left.y + self.center.y + self.middle_right.y) / 3.0;
            (0.0, 0.5, get_t(raw.y, top_avg_y, mid_avg_y))
        } else {
            // Bottom half: between middle row and bottom row
            let mid_avg_y = (self.middle_left.y + self.center.y + self.middle_right.y) / 3.0;
            let bot_avg_y = (self.bottom_left.y + self.bottom_center.y + self.bottom_right.y) / 3.0;
            (0.5, 1.0, get_t(raw.y, mid_avg_y, bot_avg_y))
        };

        // Determine which column (left/center/right) based on X
        let (col_low, col_high, tx) = if raw.x < self.center.x {
            // Left half: between left column and center column
            let left_avg_x = (self.top_left.x + self.middle_left.x + self.bottom_left.x) / 3.0;
            let ctr_avg_x = (self.top_center.x + self.center.x + self.bottom_center.x) / 3.0;
            (0.0, 0.5, get_t(raw.x, left_avg_x, ctr_avg_x))
        } else {
            // Right half: between center column and right column
            let ctr_avg_x = (self.top_center.x + self.center.x + self.bottom_center.x) / 3.0;
            let right_avg_x = (self.top_right.x + self.middle_right.x + self.bottom_right.x) / 3.0;
            (0.5, 1.0, get_t(raw.x, ctr_avg_x, right_avg_x))
        };

        // Bilinear interpolation
        let dx = lerp(col_low, col_high, tx);
        let dy = lerp(row_low, row_high, ty);

        Point { x: dx.clamp(0.0, 1.0), y: dy.clamp(0.0, 1.0) }
    }
}
