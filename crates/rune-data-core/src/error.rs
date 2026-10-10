// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    BadFen(String),
    BadPgn(String),
    BadFormat(String),
    ChecksumMismatch { want: u64, got: u64 },
    UnsupportedVersion { want: u32, got: u32 },
    Truncated,
    Io(String),
    BadConfig(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadFen(s) => write!(f, "bad FEN: {s}"),
            Error::BadPgn(s) => write!(f, "bad PGN: {s}"),
            Error::BadFormat(s) => write!(f, "bad format: {s}"),
            Error::ChecksumMismatch { want, got } => {
                write!(f, "checksum mismatch: want {want:016x} got {got:016x}")
            }
            Error::UnsupportedVersion { want, got } => {
                write!(f, "unsupported version: want {want} got {got}")
            }
            Error::Truncated => write!(f, "truncated input"),
            Error::Io(s) => write!(f, "I/O error: {s}"),
            Error::BadConfig(s) => write!(f, "bad config: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
