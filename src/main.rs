use std::io::{self, BufRead, BufReader, Write};
use std::process::{Command, Stdio, ChildStdout, ChildStdin};
use std::time;
use serde::Deserialize;
use mouse_rs::Mouse;
use display_info::DisplayInfo;
use clap::Parser;

mod smoothing;
use smoothing::OneEuroFilter;

mod ipc;
use ipc::Command as PythonCommand;

mod calibration;
use calibration::{CalibrationProfile, Point};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run calibration sequence
    #[arg(short, long)]
    calibrate: bool,
    
    /// Run diagnostics mode
    #[arg(short, long)]
    diagnose: bool,
}

#[derive(Deserialize, Debug)]
struct GazePoint {
    x: f64,
    y: f64,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let calibrations_file = "calibration.json";

    // Get Screen Dimensions First
    let displays = DisplayInfo::all()?;
    if displays.is_empty() {
        return Err("No displays found".into());
    }
    let main_display = &displays[0];
    let screen_width = main_display.width as f64;
    let screen_height = main_display.height as f64;
    
    println!("Starting Iris Eye Tracker...");
    println!("Screen Resolution: {}x{}", screen_width, screen_height);
    
    // ... spawn process ...
    let mut child = Command::new("./venv/bin/python")
        .arg("tracking/eye_tracker.py")
        .stdout(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn()?;

    println!("Vision process started. PID: {}", child.id());

    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let mut stdin = child.stdin.take().ok_or("Failed to capture stdin")?;
    let mut reader = BufReader::new(stdout);
    
    // Send Init
    let init_cmd = PythonCommand::Init { 
        width: screen_width as u32, 
        height: screen_height as u32 
    };
    writeln!(stdin, "{}", serde_json::to_string(&init_cmd)?)?;

    if args.calibrate {
        run_calibration_sequence(&mut reader, &mut stdin, calibrations_file)?;
        println!("Calibration saved. Restart without --calibrate to run.");
        return Ok(());
    }
    
    // Smoothing state (One Euro Filter)
    // MinCutoff: Lower = More smoothing when stationary (0.01 is very steady)
    // Beta: Lower = Less sensitivity to speed (0.002 reduces jitter spikes)
    // Load Calibration
    let profile = CalibrationProfile::load(calibrations_file);
    
    if args.diagnose {
        run_diagnostics(&mut reader, &mut stdin, &profile, screen_width, screen_height)?;
        println!("Diagnostics complete. Data saved to diagnostics.csv");
        return Ok(());
    }

    // ... (rest of main loop for tracking)
    let mouse = Mouse::new();
    let mut filter_x = OneEuroFilter::new(0.01, 0.002);
    let mut filter_y = OneEuroFilter::new(0.01, 0.002);
    
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 { break; }

        if let Ok(raw_point) = serde_json::from_str::<GazePoint>(&line) {
             let raw = Point { x: raw_point.x, y: raw_point.y };
             let mapped = profile.map(&raw);
             let target_x = mapped.x * screen_width;
             let target_y = mapped.y * screen_height;
             let smooth_x = filter_x.filter(target_x);
             let smooth_y = filter_y.filter(target_y);
             if let Err(_e) = mouse.move_to(smooth_x as i32, smooth_y as i32) {}
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct TriggerMessage {
    #[serde(rename = "type")]
    msg_type: String,
}

fn run_diagnostics(
    reader: &mut BufReader<ChildStdout>, 
    writer: &mut ChildStdin, 
    profile: &CalibrationProfile,
    screen_w: f64,
    screen_h: f64
) -> std::io::Result<()> {
    use std::fs::File;
    
    println!("\n=== DIAGNOSTICS MODE ===");
    println!("We will record data for 3 seconds at 5 points.");
    println!("Look at the RED DOT and press SPACE.");

    let points = vec![
        ("CENTER", 0.5, 0.5),
        ("TOP LEFT", 0.05, 0.05),
        ("TOP RIGHT", 0.95, 0.05),
        ("BOTTOM LEFT", 0.05, 0.95),
        ("BOTTOM RIGHT", 0.95, 0.95),
    ];

    let mut file = File::create("diagnostics.csv")?;
    writeln!(file, "TargetName,TargetX,TargetY,RawX,RawY,MappedX,MappedY")?;
    
    // Give Python a moment to initialize the window
    std::thread::sleep(std::time::Duration::from_secs(2));

    for (name, x, y) in points {
        // Send command
        let cmd = PythonCommand::CalibrationPoint { x, y };
        writeln!(writer, "{}", serde_json::to_string(&cmd)?)?;
        
        println!("\nWaiting for user trigger at {}...", name);
        wait_for_trigger(reader)?;

        println!("Recording for 3 seconds...");
        let start = time::Instant::now();
        let duration = time::Duration::from_millis(3000);
        let mut line = String::new();
        
        while start.elapsed() < duration {
            line.clear();
            if reader.read_line(&mut line)? > 0 {
                 if let Ok(p) = serde_json::from_str::<GazePoint>(&line) {
                     let raw = Point { x: p.x, y: p.y };
                     let mapped = profile.map(&raw);
                     
                     writeln!(file, "{},{},{},{:.6},{:.6},{:.6},{:.6}", 
                        name, x * screen_w, y * screen_h, 
                        raw.x, raw.y, 
                        mapped.x * screen_w, mapped.y * screen_h
                     )?;
                 }
            }
        }
    }

    let cmd = PythonCommand::CalibrationEnd;
    writeln!(writer, "{}", serde_json::to_string(&cmd)?)?;
    Ok(())
}

fn run_calibration_sequence(reader: &mut BufReader<ChildStdout>, writer: &mut ChildStdin, path: &str) -> std::io::Result<()> {
    println!("\n=== 9-POINT CALIBRATION ===");
    println!("The tracker window will go fullscreen.");
    println!("Look at the RED DOT and press SPACE or ENTER (on the tracking window, not terminal).");

    // 3x3 Grid: 9 points
    let points = vec![
        ("CENTER",        0.5,  0.5),
        ("TOP LEFT",      0.05, 0.05),
        ("TOP CENTER",    0.5,  0.05),
        ("TOP RIGHT",     0.95, 0.05),
        ("MIDDLE LEFT",   0.05, 0.5),
        ("MIDDLE RIGHT",  0.95, 0.5),
        ("BOTTOM LEFT",   0.05, 0.95),
        ("BOTTOM CENTER", 0.5,  0.95),
        ("BOTTOM RIGHT",  0.95, 0.95),
    ];

    let mut results = Vec::new();

    // Give Python a moment to initialize the window
    std::thread::sleep(std::time::Duration::from_secs(2));
    
    // READY step: Show a dot to ensure display is initialized
    // This is dismissed without recording data
    let ready_cmd = PythonCommand::CalibrationPoint { x: 0.5, y: 0.5 };
    writeln!(writer, "{}", serde_json::to_string(&ready_cmd)?)?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!("\nPress SPACE when READY...");
    wait_for_trigger(reader)?;

    for (name, x, y) in points {
        // Send command
        let cmd = PythonCommand::CalibrationPoint { x, y };
        writeln!(writer, "{}", serde_json::to_string(&cmd)?)?;
        
        // Give Python time to draw the dot
        std::thread::sleep(std::time::Duration::from_millis(500));
        
        println!("\nWaiting for user trigger at {}...", name);
        wait_for_trigger(reader)?;

        println!("Recording...");
        let pt = collect_point(reader)?;
        results.push(pt);
    }

    let cmd = PythonCommand::CalibrationEnd;
    writeln!(writer, "{}", serde_json::to_string(&cmd)?)?;

    let profile = CalibrationProfile {
        center: results[0].clone(),
        top_left: results[1].clone(),
        top_center: results[2].clone(),
        top_right: results[3].clone(),
        middle_left: results[4].clone(),
        middle_right: results[5].clone(),
        bottom_left: results[6].clone(),
        bottom_center: results[7].clone(),
        bottom_right: results[8].clone(),
    };

    profile.save(path)?;
    println!("\nCalibration saved into {}!", path);
    Ok(())
}

fn wait_for_trigger(reader: &mut BufReader<ChildStdout>) -> std::io::Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }
        // Check if line is trigger
        if let Ok(msg) = serde_json::from_str::<TriggerMessage>(&line) {
            if msg.msg_type == "trigger" {
                return Ok(());
            }
        }
    }
}

fn collect_point(reader: &mut BufReader<ChildStdout>) -> std::io::Result<Point> {
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
        return Err(io::Error::new(io::ErrorKind::Other, "No data received"));
    }
    Ok(Point { x: sum_x / count as f64, y: sum_y / count as f64 })
}
