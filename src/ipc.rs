use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum Command {
    #[serde(rename = "init")]
    Init { width: u32, height: u32 },
    #[serde(rename = "calibration_point")]
    CalibrationPoint { x: f64, y: f64 },
    #[serde(rename = "calibration_end")]
    CalibrationEnd,
}
