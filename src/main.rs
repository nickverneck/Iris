use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use serde::Deserialize;
use mouse_rs::Mouse;
use display_info::DisplayInfo;

#[derive(Deserialize, Debug)]
struct GazePoint {
    x: f64,
    y: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Iris Eye Tracker...");

    // Get Screen Dimensions
    let displays = DisplayInfo::all()?;
    if displays.is_empty() {
        return Err("No displays found".into());
    }
    let main_display = &displays[0]; // Assume first is main for now
    let screen_width = main_display.width as f64;
    let screen_height = main_display.height as f64;
    
    println!("Screen Resolution: {}x{}", screen_width, screen_height);

    let mouse = Mouse::new();

    // Start Python Process
    let mut child = Command::new("./venv/bin/python")
        .arg("tracking/eye_tracker.py")
        .stdout(Stdio::piped())
        .spawn()?;

    println!("Vision process started. PID: {}", child.id());

    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let reader = BufReader::new(stdout);

    // Smoothing state
    let mut smooth_x = screen_width / 2.0;
    let mut smooth_y = screen_height / 2.0;
    let alpha = 0.2; // Smoothing factor (0.0 to 1.0). Lower = smoother but more lag.

    for line in reader.lines() {
        let line = line?;
        if let Ok(point) = serde_json::from_str::<GazePoint>(&line) {
            // Map 0-1 to Screen Coordinates
            let target_x = point.x * screen_width;
            let target_y = point.y * screen_height;

            // Apply Smoothing
            smooth_x = alpha * target_x + (1.0 - alpha) * smooth_x;
            smooth_y = alpha * target_y + (1.0 - alpha) * smooth_y;

            // Move Mouse
            // Mouse-rs coords might be i32
            let _ = mouse.move_to(smooth_x as i32, smooth_y as i32);
        } else {
             // Print invalid lines for debugging (maybe python debug output)
             // println!("Raw: {}", line);
        }
    }

    Ok(())
}
