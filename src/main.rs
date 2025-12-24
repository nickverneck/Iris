use std::io::{self, BufRead, BufReader, Write};
use std::process::{Command, Stdio, ChildStdout};
use std::{thread, time};
use serde::Deserialize;
use mouse_rs::Mouse;
use display_info::DisplayInfo;
use clap::Parser;

mod calibration;
use calibration::{CalibrationProfile, Point};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run calibration sequence
    #[arg(short, long)]
    calibrate: bool,
}

#[derive(Deserialize, Debug)]
struct GazePoint {
    x: f64,
    y: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let calibrations_file = "calibration.json";

    println!("Starting Iris Eye Tracker...");

    // Start Python Process
    let mut child = Command::new("./venv/bin/python")
        .arg("tracking/eye_tracker.py")
        .stdout(Stdio::piped())
        .spawn()?;

    println!("Vision process started. PID: {}", child.id());

    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let mut reader = BufReader::new(stdout);

    if args.calibrate {
        run_calibration_sequence(&mut reader, calibrations_file)?;
        println!("Calibration saved. Restart without --calibrate to run.");
        return Ok(());
    }

    // Load Calibration
    let profile = CalibrationProfile::load(calibrations_file);
    println!("Loaded Profile: {:?}", profile);

    // Get Screen Dimensions
    let displays = DisplayInfo::all()?;
    if displays.is_empty() {
        return Err("No displays found".into());
    }
    let main_display = &displays[0];
    let screen_width = main_display.width as f64;
    let screen_height = main_display.height as f64;
    println!("Screen Resolution: {}x{}", screen_width, screen_height);

    let mouse = Mouse::new();

    // Smoothing state
    let mut smooth_x = screen_width / 2.0;
    let mut smooth_y = screen_height / 2.0;
    let alpha = 0.15; // Slightly smoother

    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break; // EOF
        }

        if let Ok(raw_point) = serde_json::from_str::<GazePoint>(&line) {
             let raw = Point { x: raw_point.x, y: raw_point.y };
             
             // Map to Unit Square (0.0-1.0) using Calibration
             let mapped = profile.map(&raw);

             // Map to Screen
             let target_x = mapped.x * screen_width;
             let target_y = mapped.y * screen_height;

             // Apply Smoothing
             smooth_x = alpha * target_x + (1.0 - alpha) * smooth_x;
             smooth_y = alpha * target_y + (1.0 - alpha) * smooth_y;

             if let Err(e) = mouse.move_to(smooth_x as i32, smooth_y as i32) {
                 // eprintln!("Mouse error: {}", e); 
             }
        }
    }

    Ok(())
}

fn run_calibration_sequence(reader: &mut BufReader<ChildStdout>, path: &str) -> std::io::Result<()> {
    println!("\n=== 5-POINT CALIBRATION ===");
    println!("We will record 5 points: Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right.");
    println!("For each point, look at it steadily and press ENTER.");
    
    let center = collect_point(reader, "CENTER")?;
    let top_left = collect_point(reader, "TOP LEFT")?;
    let top_right = collect_point(reader, "TOP RIGHT")?;
    let bottom_left = collect_point(reader, "BOTTOM LEFT")?;
    let bottom_right = collect_point(reader, "BOTTOM RIGHT")?;

    let profile = CalibrationProfile {
        center,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    };

    profile.save(path)?;
    println!("\nCalibration saved to {}!", path);
    Ok(())
}

fn collect_point(reader: &mut BufReader<ChildStdout>, name: &str) -> std::io::Result<Point> {
    print!("\nLook at the [{}] and press ENTER...", name);
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    println!("Recording (keep looking)...");
    
    // Collect samples for 1 second (approx 30 frames)
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0;
    let start = time::Instant::now();
    let duration = time::Duration::from_millis(1000);

    let mut line = String::new();
    while start.elapsed() < duration {
        line.clear();
        if reader.read_line(&mut line)? > 0 {
             if let Ok(p) = serde_json::from_str::<GazePoint>(&line) {
                 sum_x += p.x;
                 sum_y += p.y;
                 count += 1;
             }
        }
    }

    if count == 0 {
        return Err(io::Error::new(io::ErrorKind::Other, "No data received from eye tracker"));
    }

    let avg_x = sum_x / count as f64;
    let avg_y = sum_y / count as f64;
    println!("Captured: {:.4}, {:.4}", avg_x, avg_y);

    Ok(Point { x: avg_x, y: avg_y })
}
