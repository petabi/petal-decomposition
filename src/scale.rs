use crate::DecompositionError;
use ndarray::{Array1, Array2, Axis, Data};
use ndarray::{ArrayBase, Ix2};
use ndarray_linalg::Scalar;

/// Trait for implementing scalings on [`Array2`] data.
pub trait Scale<A, S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    /// Scale an input dataset.
    fn scale(&self, input: &ArrayBase<S, Ix2>) -> Array2<A>;

    /// Reverse the scaling.
    fn inverse_scale(&self, input: &ArrayBase<S, Ix2>) -> Array2<A>;
}

/// Implements a mean centered scaling.
#[derive(Debug)]
pub struct MeanCentered<A>
where
    A: Scalar,
{
    /// The computed means for [`Axis`] 0 of the input data.
    pub means: Array1<A>,
}

impl<A> MeanCentered<A>
where
    A: Scalar,
{
    /// Compute the [`Axis`] 0 means on input data and create a [`MeanCentered`].
    ///
    /// # Errors
    ///
    /// * `DecompositionError::InvalidInput` if the number of rows in the input is 0
    ///
    /// # Examples
    ///
    /// ```
    /// use petal_decomposition::{Scale, MeanCentered};
    /// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [2_f64, 2_f64]]);
    /// let mean_scale = MeanCentered::new(&x).unwrap();
    /// assert_eq!(mean_scale.means[0], 1.);
    /// assert_eq!(mean_scale.means[1], 1.);
    ///
    /// let scaled_x = mean_scale.scale(&x);
    /// assert_eq!(scaled_x[(0, 0)], -1.);
    /// assert_eq!(scaled_x[(1, 0)], 0.);
    /// assert_eq!(scaled_x[(2, 0)], 1.);
    /// ```
    pub fn new<S>(input: &ArrayBase<S, Ix2>) -> Result<Self, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        let means = if let Some(means) = input.mean_axis(Axis(0)) {
            means
        } else {
            return Err(DecompositionError::InvalidInput);
        };
        Ok(Self { means })
    }
}

impl<A, S> Scale<A, S> for MeanCentered<A>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    /// Perform mean centered scaling on the input data.
    fn scale(&self, input: &ArrayBase<S, Ix2>) -> Array2<A> {
        input - &self.means
    }

    /// Perform mean centered scaling on the input data.
    fn inverse_scale(&self, input: &ArrayBase<S, Ix2>) -> Array2<A> {
        input + &self.means
    }
}

#[cfg(test)]
mod test {
    use crate::DecompositionError;

    use super::{MeanCentered, Scale};
    use approx::assert_relative_eq;
    use ndarray::Array2;
    #[test]
    fn mean_centered_scale() {
        let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [2_f64, 2_f64]]);
        let mean_scale = MeanCentered::new(&x).unwrap();
        assert_relative_eq!(mean_scale.means[0], 1.);
        assert_relative_eq!(mean_scale.means[1], 1.);

        let scaled_x = mean_scale.scale(&x);
        assert_relative_eq!(scaled_x[(0, 0)], -1.);
        assert_relative_eq!(scaled_x[(1, 0)], 0.);
        assert_relative_eq!(scaled_x[(2, 0)], 1.);

        assert_relative_eq!(x, mean_scale.inverse_scale(&scaled_x));
    }

    #[test]
    fn mean_centered_scale_zeros() {
        let x = Array2::<f32>::zeros((0, 5));
        match MeanCentered::new(&x) {
            Ok(_) => assert!(false, "Mean on zero size matrix should error."),
            Err(DecompositionError::InvalidInput) => assert!(true),
            Err(e) => assert!(false, "Unexpected error type {:?} for mean", e),
        };
    }
}
