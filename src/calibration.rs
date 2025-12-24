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
    pub center: Point,
    pub top_left: Point,
    pub top_right: Point,
    pub bottom_left: Point,
    pub bottom_right: Point,
}

impl CalibrationProfile {
    pub fn default() -> Self {
        // Fallback default values if no calibration exists
        CalibrationProfile {
            center: Point { x: 0.5, y: 0.5 },
            top_left: Point { x: 0.3, y: 0.3 },
            top_right: Point { x: 0.7, y: 0.3 },
            bottom_left: Point { x: 0.3, y: 0.7 },
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
    // using a quadrant-based approach relative to the center.
    pub fn map(&self, raw: &Point) -> Point {
        let dx;
        let dy;

        // Determine X position relative to center
        if raw.x < self.center.x {
            // Left Side: Map [top_left.x/bottom_left.x, center.x] -> [0.0, 0.5]
            // We use linear interpolation between the left bound and center.
            // But the left bound changes depending on Y. Let's simplify:
            // Normalize X based on whether it's left or right of center
            let left_bound = (self.top_left.x + self.bottom_left.x) / 2.0; // Simplify for X
            if raw.x < left_bound {
                dx = 0.0;
            } else {
                 dx = 0.5 * (raw.x - left_bound) / (self.center.x - left_bound);
            }
        } else {
             // Right Side: Map [center.x, right_bound] -> [0.5, 1.0]
             let right_bound = (self.top_right.x + self.bottom_right.x) / 2.0;
             if raw.x > right_bound {
                 dx = 1.0;
             } else {
                 dx = 0.5 + 0.5 * (raw.x - self.center.x) / (right_bound - self.center.x);
             }
        }

        // Determine Y position relative to center
        if raw.y < self.center.y {
             // Top Side (Raw Y is usually smaller at top? Check phyisology. 
             // In screen space 0 is top. In ratio: 
             // If looking UP, iris goes UP. Iris y coordinate depends on camera.
             // Usually Iris Y Decreases as you look up (closer to top of frame/eyelid)? 
             // Let's assume input raw follows screen convention (0 top, 1 bottom) broadly or we flip later.
             let top_bound = (self.top_left.y + self.top_right.y) / 2.0;
             if raw.y < top_bound {
                 dy = 0.0;
             } else {
                 dy = 0.5 * (raw.y - top_bound) / (self.center.y - top_bound);
             }
        } else {
             // Bottom Side
             let bottom_bound = (self.bottom_left.y + self.bottom_right.y) / 2.0;
             if raw.y > bottom_bound {
                 dy = 1.0;
             } else {
                 dy = 0.5 + 0.5 * (raw.y - self.center.y) / (bottom_bound - self.center.y);
             }
        }

        Point { x: dx.clamp(0.0, 1.0), y: dy.clamp(0.0, 1.0) }
    }
}
