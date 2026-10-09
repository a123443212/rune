use std::fmt;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    BadFen(String),
    BadUci(String),
    UnsupportedArch(String),
    UnsupportedQuant(String),
    GameMismatch(String),
    FeatureMismatch(String),
    TensorMissing(String),
    Shape(String),
    InvalidState(String),
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::BadFen(s) => write!(f, "bad fen: {}", s),
            RuntimeError::BadUci(s) => write!(f, "bad uci: {}", s),
            RuntimeError::UnsupportedArch(s) => write!(f, "unsupported arch: {}", s),
            RuntimeError::UnsupportedQuant(s) => write!(f, "unsupported quant: {}", s),
            RuntimeError::GameMismatch(s) => write!(f, "game mismatch: {}", s),
            RuntimeError::FeatureMismatch(s) => write!(f, "feature mismatch: {}", s),
            RuntimeError::TensorMissing(s) => write!(f, "missing tensor: {}", s),
            RuntimeError::Shape(s) => write!(f, "shape: {}", s),
            RuntimeError::InvalidState(s) => write!(f, "invalid state: {}", s),
        }
    }
}
impl std::error::Error for RuntimeError {}
pub type Result<T> = std::result::Result<T, RuntimeError>;
