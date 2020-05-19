mod pca;

use ndarray_linalg::error::LinalgError;
pub use pca::Pca;
use thiserror::Error;

/// The error type for PCA operations.
#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("both dimensions of input matrix ({}, {}) should be larger than or equal to the number of components, {}", .n_rows, .n_cols, .n_components)]
    InvalidInput {
        n_components: usize,
        n_rows: usize,
        n_cols: usize,
    },
    #[error("linear algerba operation failed")]
    LinalgError(#[from] LinalgError),
}
