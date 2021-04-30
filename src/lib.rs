mod ica;
mod linalg;
mod pca;
mod scale;

pub use ica::FastIca;
use ndarray_linalg::error::LinalgError;
pub use pca::{Pca, RandomizedPca};
pub use scale::{MeanCentered, Scale};
use thiserror::Error;

/// The error type for PCA operations.
#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("invalid matrix size")]
    InvalidInput,
    #[error("linear algerba operation failed")]
    LinalgError(#[from] LinalgError),
}
