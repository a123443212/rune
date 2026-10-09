pub fn clipped_relu(x: f32) -> f32 {
    if x < 0.0 {
        return 0.0;
    }
    if x > 1.0 {
        return 1.0;
    }
    x
}

pub fn hard_sigmoid(s: f32) -> f32 {
    let t = 0.2_f32 * s + 0.5_f32;
    if t < 0.0 {
        return 0.0;
    }
    if t > 1.0 {
        return 1.0;
    }
    t
}

pub fn screlu(s: f32) -> f32 {
    let c = clipped_relu(s);
    c * c
}

pub fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn relu_inplace(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

pub fn clamp_delta(v: f32) -> f32 {
    if v < -0.25 {
        return -0.25;
    }
    if v > 0.25 {
        return 0.25;
    }
    v
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    Clip,
    HardSigmoid,
    Screlu,
}

impl Gate {
    pub fn from_str(s: &str) -> Option<Gate> {
        match s {
            "clip" => Some(Gate::Clip),
            "hard_sigmoid" => Some(Gate::HardSigmoid),
            "screlu" => Some(Gate::Screlu),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Gate::Clip => "clip",
            Gate::HardSigmoid => "hard_sigmoid",
            Gate::Screlu => "screlu",
        }
    }

    pub fn apply(self, x: f32) -> f32 {
        match self {
            Gate::Clip => clipped_relu(x),
            Gate::HardSigmoid => hard_sigmoid(x),
            Gate::Screlu => screlu(x),
        }
    }
}

pub fn tokens_clip(acc: &[f32], tok: &mut [f32]) {
    for i in 0..acc.len() {
        tok[i] = clipped_relu(acc[i]);
    }
}

pub fn tokens_dequant_clip(acc: &[i32], scale: f32, tok: &mut [f32]) {
    for i in 0..acc.len() {
        tok[i] = clipped_relu(acc[i] as f32 * scale);
    }
}
