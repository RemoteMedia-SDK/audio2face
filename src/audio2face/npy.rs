//! Minimal NumPy `.npy` v1/v2 reader.
//!
//! Rust port of [`external/.../Utils/IO/NpyReader.cs`](../../../../../../external/handcrafted-persona-engine/src/PersonaEngine/PersonaEngine.Lib/Utils/IO/NpyReader.cs).
//!
//! Supports float32 (`<f4`) and int32 (`<i4`) C-contiguous (row-major)
//! arrays — that's what the persona-engine bundle's NPZs ship and is
//! what every consumer in this codebase needs. Other dtypes / endianness
//! are rejected with a clear error.
//!
//! Reads from any `Read`er — the [`super::npz::NpzArchive`] wraps
//! `zip::read::ZipFile` streams here.
//!
//! Format reference: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>

use std::io::Read;

/// `\x93NUMPY` — fixed magic-bytes prefix on every `.npy` file.
const MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];

/// Loaded f32 data + its row-major shape.
#[derive(Debug)]
pub struct NpyF32 {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Loaded i32 data + its row-major shape.
#[derive(Debug)]
pub struct NpyI32 {
    pub data: Vec<i32>,
    pub shape: Vec<usize>,
}

/// Read a `.npy` stream as float32. Errors on non-`<f4` dtype or
/// truncated payloads.
pub fn read_f32<R: Read>(mut reader: R, source: &str) -> Result<NpyF32, NpyError> {
    let (dtype, shape) = read_header(&mut reader, source)?;
    if dtype != "<f4" && dtype != "f4" {
        return Err(NpyError::WrongDtype {
            expected: "<f4",
            actual: dtype,
            context: source.to_string(),
        });
    }
    let total = shape_total(&shape);
    let mut data = vec![0.0f32; total];
    let bytes = bytemuck::cast_slice_mut(&mut data[..]);
    reader
        .read_exact(bytes)
        .map_err(|e| NpyError::Io(e, source.to_string()))?;
    Ok(NpyF32 { data, shape })
}

/// Read a `.npy` stream as int32.
pub fn read_i32<R: Read>(mut reader: R, source: &str) -> Result<NpyI32, NpyError> {
    let (dtype, shape) = read_header(&mut reader, source)?;
    if dtype != "<i4" && dtype != "i4" {
        return Err(NpyError::WrongDtype {
            expected: "<i4",
            actual: dtype,
            context: source.to_string(),
        });
    }
    let total = shape_total(&shape);
    let mut data = vec![0i32; total];
    let bytes = bytemuck::cast_slice_mut(&mut data[..]);
    reader
        .read_exact(bytes)
        .map_err(|e| NpyError::Io(e, source.to_string()))?;
    Ok(NpyI32 { data, shape })
}

/// Errors surfaced by the NPY reader.
#[derive(Debug, thiserror::Error)]
pub enum NpyError {
    #[error("not a valid .npy file (bad magic) in {0}")]
    BadMagic(String),

    #[error("unsupported .npy version {version} in {context}")]
    UnsupportedVersion { version: u8, context: String },

    #[error("wrong dtype in {context}: expected {expected}, got {actual}")]
    WrongDtype {
        expected: &'static str,
        actual: String,
        context: String,
    },

    #[error("malformed header in {context}: {message}")]
    BadHeader { context: String, message: String },

    #[error("invalid shape in {context}: {message}")]
    BadShape { context: String, message: String },

    #[error("io error reading {1}: {0}")]
    Io(#[source] std::io::Error, String),
}

/// Compute total element count from a shape vector. Empty shape → 1
/// (matches NumPy zero-d scalar semantics).
fn shape_total(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>().max(1)
}

/// Parse the `.npy` preamble + header, returning `(dtype, shape)`.
fn read_header<R: Read>(reader: &mut R, source: &str) -> Result<(String, Vec<usize>), NpyError> {
    // Preamble: 10 bytes for v1, 12 for v2. We always read 10 first.
    let mut preamble = [0u8; 10];
    reader
        .read_exact(&mut preamble)
        .map_err(|e| NpyError::Io(e, source.to_string()))?;
    if preamble[..6] != MAGIC {
        return Err(NpyError::BadMagic(source.to_string()));
    }
    let major = preamble[6];
    let header_len = match major {
        1 => u16::from_le_bytes([preamble[8], preamble[9]]) as usize,
        2 => {
            let mut tail = [0u8; 2];
            reader
                .read_exact(&mut tail)
                .map_err(|e| NpyError::Io(e, source.to_string()))?;
            u32::from_le_bytes([preamble[8], preamble[9], tail[0], tail[1]]) as usize
        }
        v => {
            return Err(NpyError::UnsupportedVersion {
                version: v,
                context: source.to_string(),
            })
        }
    };
    let mut header_buf = vec![0u8; header_len];
    reader
        .read_exact(&mut header_buf)
        .map_err(|e| NpyError::Io(e, source.to_string()))?;
    let header = std::str::from_utf8(&header_buf).map_err(|_| NpyError::BadHeader {
        context: source.to_string(),
        message: "header bytes are not valid UTF-8".into(),
    })?;
    parse_header(header, source)
}

/// Extract `dtype` and `shape` from a numpy header dict-string like
/// `{'descr': '<f4', 'fortran_order': False, 'shape': (768, 1024), }`.
fn parse_header(header: &str, source: &str) -> Result<(String, Vec<usize>), NpyError> {
    let dtype = extract_string_value(header, "'descr':").ok_or_else(|| NpyError::BadHeader {
        context: source.to_string(),
        message: "missing 'descr' key".into(),
    })?;
    let shape_str = extract_tuple_value(header, "'shape':").ok_or_else(|| NpyError::BadHeader {
        context: source.to_string(),
        message: "missing 'shape' key".into(),
    })?;
    let inner = shape_str.trim_matches(|c: char| c == '(' || c == ')' || c.is_whitespace());
    let shape: Vec<usize> = if inner.is_empty() {
        vec![1]
    } else {
        inner
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>().map_err(|_| NpyError::BadShape {
                    context: source.to_string(),
                    message: format!("non-integer dim '{s}'"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok((dtype, shape))
}

/// Extract a quoted-string value following `key` in the header.
fn extract_string_value(header: &str, key: &str) -> Option<String> {
    let idx = header.find(key)?;
    let rest = &header[idx + key.len()..];
    let rest = rest.trim_start();
    if !rest.starts_with('\'') {
        return None;
    }
    let after_quote = &rest[1..];
    let end = after_quote.find('\'')?;
    Some(after_quote[..end].to_string())
}

/// Extract a tuple `(…, …)` value following `key`. Returns the raw
/// substring including parens.
fn extract_tuple_value(header: &str, key: &str) -> Option<String> {
    let idx = header.find(key)?;
    let rest = &header[idx + key.len()..];
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return None;
    }
    let end = rest.find(')')?;
    Some(rest[..=end].to_string())
}
