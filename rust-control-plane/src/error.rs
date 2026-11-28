use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Config(String),
    NotReady,
    ServerBusy,
    StaleGeneration,
    DuplicateWorker,
}

impl Error {
    pub fn status(&self) -> u16 {
        match self {
            Self::ServerBusy => 429,
            _ => 503,
        }
    }
    pub fn code(&self) -> &'static str {
        match self {
            Self::ServerBusy => "server_busy",
            _ => "not_ready",
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
