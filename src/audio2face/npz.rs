//! Minimal `.npz` archive reader.
//!
//! `.npz` is just a ZIP file containing one `.npy` per saved array.
//! NumPy's `np.savez(path, name=array)` produces entries like
//! `name.npy`. The persona-engine bundle's NPZs follow that
//! convention (`neutral.npy`, `frontalMask.npy`, `eyeBlinkLeft.npy`,
//! …), so we just open the ZIP and dispatch each entry through
//! [`super::npy`].
//!
//! No streaming abstraction here on purpose — the archives are
//! ≤ 15 MB compressed and we read each entry once at startup. Flat
//! API beats lifetime gymnastics.

use super::npy::{self, NpyError, NpyF32, NpyI32};
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

/// Errors surfaced by the NPZ archive layer.
#[derive(Debug, thiserror::Error)]
pub enum NpzError {
    #[error("io error opening {1}: {0}")]
    Io(#[source] std::io::Error, String),

    #[error("zip error in {1}: {0}")]
    Zip(#[source] zip::result::ZipError, String),

    #[error("npy error in {1}: {0}")]
    Npy(#[source] NpyError, String),

    #[error("entry '{0}' not found in NPZ {1}")]
    EntryMissing(String, String),
}

/// Open + dispatch over an `.npz` archive.
pub struct NpzArchive<R: Read + Seek> {
    archive: zip::ZipArchive<R>,
    /// Source path / id, surfaced in errors for diagnostics.
    source: String,
}

impl NpzArchive<File> {
    /// Open `.npz` from a filesystem path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, NpzError> {
        let path = path.as_ref();
        let source = path.display().to_string();
        let file = File::open(path).map_err(|e| NpzError::Io(e, source.clone()))?;
        let archive = zip::ZipArchive::new(file).map_err(|e| NpzError::Zip(e, source.clone()))?;
        Ok(Self { archive, source })
    }
}

impl<R: Read + Seek> NpzArchive<R> {
    /// Build from any `Read + Seek` source (e.g. an in-memory cursor
    /// for tests).
    pub fn from_reader(reader: R, source_name: impl Into<String>) -> Result<Self, NpzError> {
        let source = source_name.into();
        let archive = zip::ZipArchive::new(reader).map_err(|e| NpzError::Zip(e, source.clone()))?;
        Ok(Self { archive, source })
    }

    /// True iff the archive has an entry by the exact name.
    pub fn has_entry(&mut self, name: &str) -> bool {
        self.archive.by_name(name).is_ok()
    }

    /// Read an `.npy` entry as float32. The entry name should include
    /// the `.npy` suffix (NumPy's `savez` always writes that suffix).
    pub fn read_f32(&mut self, name: &str) -> Result<NpyF32, NpzError> {
        let mut entry = self.archive.by_name(name).map_err(|e| match e {
            zip::result::ZipError::FileNotFound => {
                NpzError::EntryMissing(name.to_string(), self.source.clone())
            }
            other => NpzError::Zip(other, self.source.clone()),
        })?;
        npy::read_f32(&mut entry, name).map_err(|e| NpzError::Npy(e, name.to_string()))
    }

    /// Read an `.npy` entry as int32.
    pub fn read_i32(&mut self, name: &str) -> Result<NpyI32, NpzError> {
        let mut entry = self.archive.by_name(name).map_err(|e| match e {
            zip::result::ZipError::FileNotFound => {
                NpzError::EntryMissing(name.to_string(), self.source.clone())
            }
            other => NpzError::Zip(other, self.source.clone()),
        })?;
        npy::read_i32(&mut entry, name).map_err(|e| NpzError::Npy(e, name.to_string()))
    }

    /// List every entry name in the archive (for diagnostics +
    /// fixture introspection).
    pub fn entry_names(&self) -> Vec<String> {
        self.archive.file_names().map(|s| s.to_string()).collect()
    }
}
