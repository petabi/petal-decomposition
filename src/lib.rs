mod ica;
mod linalg;
mod pca;

pub use ica::FastIca;
use ndarray_linalg::error::LinalgError;
pub use pca::{Pca, PcaBuilder, RandomizedPca, RandomizedPcaBuilder};
use thiserror::Error;

/// The error type for PCA operations.
#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("invalid matrix size")]
    InvalidInput,
    #[error("linear algerba operation failed")]
    LinalgError(#[from] LinalgError),
}
