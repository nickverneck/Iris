use std::time::Instant;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct LowPassFilter {
    y: f64,
    a: f64,
    s: f64,
    initialized: bool,
}

impl LowPassFilter {
    pub fn new() -> LowPassFilter {
        LowPassFilter { y: 0.0, a: 0.0, s: 0.0, initialized: false }
    }

    pub fn filter(&mut self, value: f64, alpha: f64) -> f64 {
        if !self.initialized {
            self.s = value;
            self.y = value;
            self.initialized = true;
        } else {
            self.a = alpha;
            self.s = alpha * value + (1.0 - alpha) * self.s;
            self.y = self.s;
        }
        self.y
    }
    
    pub fn last_val(&self) -> f64 {
        self.y
    }
}

pub struct OneEuroFilter {
    freq: f64,
    mincutoff: f64,
    beta: f64,
    dcutoff: f64,
    x_filter: LowPassFilter,
    dx_filter: LowPassFilter,
    last_time: Option<Instant>,
}

impl OneEuroFilter {
    pub fn new(mincutoff: f64, beta: f64) -> OneEuroFilter {
        OneEuroFilter {
            freq: 30.0, // Default assumption, updated dynamically
            mincutoff,
            beta,
            dcutoff: 1.0,
            x_filter: LowPassFilter::new(),
            dx_filter: LowPassFilter::new(),
            last_time: None,
        }
    }

    fn alpha(&self, cutoff: f64) -> f64 {
        let te = 1.0 / self.freq;
        let tau = 1.0 / (2.0 * PI * cutoff);
        1.0 / (1.0 + tau / te)
    }

    pub fn filter(&mut self, value: f64) -> f64 {
        let now = Instant::now();
        
        // Update frequency based on time delta
        if let Some(last) = self.last_time {
            let dt = now.duration_since(last).as_secs_f64();
            if dt > 0.0 {
                self.freq = 1.0 / dt;
            }
        }
        self.last_time = Some(now);

        // Estimate derivative (speed)
        let dx = (value - self.x_filter.last_val()) * self.freq;
        let edx = self.dx_filter.filter(dx, self.alpha(self.dcutoff));

        // Use speed to adapt cutoff frequency
        // High speed -> High cutoff (less lag)
        // Low speed -> Low cutoff (more smoothing)
        let cutoff = self.mincutoff + self.beta * edx.abs();
        
        self.x_filter.filter(value, self.alpha(cutoff))
    }
}
