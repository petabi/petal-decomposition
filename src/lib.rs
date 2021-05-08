//! petal-decomposition provides matrix decomposition algorithms including PCA
//! (principal component analysis) and ICA (independent component analysis).

mod ica;
mod linalg;
mod pca;

pub use ica::{FastIca, FastIcaBuilder};
pub use pca::{Pca, PcaBuilder, RandomizedPca, RandomizedPcaBuilder};
use thiserror::Error;

/// The error type for PCA operations.
#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("invalid matrix: {0}")]
    InvalidInput(String),
    #[error("linear algerba operation failed: {0}")]
    LinalgError(String),
}
