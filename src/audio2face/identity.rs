//! Audio2Face bundle identities (Claire, James, Mark) + filename
//! resolver for the persona-engine bundle layout.
//!
//! The bundle ships three identities side-by-side under one directory.
//! Filenames embed the identity name as a suffix, e.g.
//! `bs_skin_Claire.npz`, `model_data_James.npz`. This module owns the
//! convention so the inference + loader code can refer to identities
//! symbolically.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Per-identity blendshape rig packaged in the persona-engine bundle.
/// Each identity has its own NPZ files + JSON configs; the ONNX
/// network is shared across identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Audio2FaceIdentity {
    /// Default identity in the persona-engine. Use this if you don't
    /// have a preference — the rest of the bundle's tuning targets it.
    Claire,
    James,
    Mark,
}

impl Audio2FaceIdentity {
    /// Index in the model's identity-onehot input vector (3 slots,
    /// matches `num_identities` in `network_info.json`).
    pub fn one_hot_index(self) -> usize {
        match self {
            Audio2FaceIdentity::Claire => 0,
            Audio2FaceIdentity::James => 1,
            Audio2FaceIdentity::Mark => 2,
        }
    }

    /// Lowercase suffix used in filenames within the bundle.
    /// (`Claire` → `"Claire"` — bundle uses PascalCase verbatim.)
    pub fn suffix(self) -> &'static str {
        match self {
            Audio2FaceIdentity::Claire => "Claire",
            Audio2FaceIdentity::James => "James",
            Audio2FaceIdentity::Mark => "Mark",
        }
    }
}

impl Default for Audio2FaceIdentity {
    fn default() -> Self {
        Self::Claire
    }
}

/// Resolves the canonical filenames inside a persona-engine
/// Audio2Face bundle directory.
///
/// The bundle's directory layout is flat: every identity's NPZ + JSON
/// sits next to the shared `network.onnx`. This resolver doesn't
/// validate that the files exist — that's the loader's job — it just
/// constructs the paths so error messages can pinpoint exactly which
/// file is missing.
#[derive(Debug, Clone)]
pub struct BundlePaths {
    pub root: PathBuf,
    pub identity: Audio2FaceIdentity,
}

impl BundlePaths {
    /// Build a resolver rooted at `root` for the given identity.
    pub fn new(root: impl AsRef<Path>, identity: Audio2FaceIdentity) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            identity,
        }
    }

    /// `network.onnx` — shared across identities.
    pub fn network_onnx(&self) -> PathBuf {
        self.root.join("network.onnx")
    }

    /// `network_info.json` — shared.
    pub fn network_info(&self) -> PathBuf {
        self.root.join("network_info.json")
    }

    /// `bs_skin_<Identity>.npz` — per-identity blendshape vertex deltas.
    pub fn bs_skin_npz(&self) -> PathBuf {
        self.root
            .join(format!("bs_skin_{}.npz", self.identity.suffix()))
    }

    /// `bs_skin_config_<Identity>.json` — per-identity solver tuning.
    pub fn bs_skin_config(&self) -> PathBuf {
        self.root
            .join(format!("bs_skin_config_{}.json", self.identity.suffix()))
    }

    /// `model_data_<Identity>.npz` — per-identity neutral skin + eye/lip
    /// pose deltas + saccade.
    pub fn model_data_npz(&self) -> PathBuf {
        self.root
            .join(format!("model_data_{}.npz", self.identity.suffix()))
    }

    /// `model_config_<Identity>.json` — per-identity model metadata
    /// (smoothing, strength offsets — used by the renderer, not the
    /// LipSync node directly, but we resolve it for completeness).
    pub fn model_config(&self) -> PathBuf {
        self.root
            .join(format!("model_config_{}.json", self.identity.suffix()))
    }
}
