use crate::DecompositionError;
use itertools::izip;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix2, OwnedRepr, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar, SVD};
use num::traits::real::Real;

/// Principal component analysis.
///
/// This reduces the dimensionality of the input data using Singular Value
/// Decomposition (SVD). The data is centered for each feature before applying
/// SVD.
///
/// # Examples
///
/// ```
/// use float_cmp::approx_eq;
/// use ndarray::arr2;
/// use petal_decomposition::Pca;
///
/// let x = arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [2_f64, 2_f64]]);
/// let y = Pca::new(1).fit_transform(&x).unwrap();  // [-2_f64.sqrt(), 0_f64, 2_f64.sqrt()]
/// assert!(approx_eq!(f64, (y[(0, 0)] - y[(2, 0)]).abs(), 2_f64.sqrt() * 2.));
/// ```
pub struct Pca<A>
where
    A: Scalar,
{
    components: Array2<A>,
    n_samples: usize,
    means: Array1<A>,
    singular: Array1<A::Real>,
}

impl<A> Pca<A>
where
    A: Scalar + Lapack,
    A::Real: ScalarOperand,
{
    /// Creates a PCA model with the given number of components.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Pca {
            components: Array2::<A>::zeros((n_components, 0)),
            n_samples: 0,
            means: Array1::<A>::zeros(0),
            singular: Array1::<A::Real>::zeros(0),
        }
    }

    /// Returns the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.components.nrows()
    }

    /// Returns sigular values.
    #[must_use]
    pub fn singular_values(&self) -> &Array1<A::Real> {
        &self.singular
    }

    /// Returns the ratio of explained variance for each component.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> Array1<A::Real> {
        let mut variance: Array1<A::Real> = &self.singular * &self.singular;
        let total_variance = variance.sum();
        variance /= total_variance;
        variance
    }

    /// Fits the model with `input`.
    ///
    /// # Errors
    ///
    /// * `DecompositionError::InvalidInput` if any of the dimensions of `input`
    ///   is less than the number of components.
    /// * `DecompositionError::LinalgError` if the underlying Singular Vector
    ///   Decomposition routine fails.
    pub fn fit<S>(&mut self, input: &ArrayBase<S, Ix2>) -> Result<(), DecompositionError>
    where
        S: Data<Elem = A>,
    {
        self.inner_fit(input)?;
        Ok(())
    }

    /// Applies dimensionality reduction to `input`.
    ///
    /// # Errors
    ///
    /// * `DecompositionError::InvalidInput` if the number of features in
    ///   `input` does not match that of the training data.
    pub fn transform<S>(&self, input: &ArrayBase<S, Ix2>) -> Result<Array2<A>, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        if input.ncols() != self.means.len() {
            return Err(DecompositionError::InvalidInput);
        }
        let x = input - &self.means;
        Ok(x.dot(&self.components.t()))
    }

    /// Fits the model with `input` and apply the dimensionality reduction on
    /// `input`.
    ///
    /// This is equivalent to calling both [`fit`] and [`transform`] for the
    /// same input.
    ///
    /// [`fit`]: #method.fit
    /// [`transform`]: #method.transform
    ///
    /// # Errors
    ///
    /// Returns `DecompositionError::LinalgError` if the underlying Singular
    /// Vector Decomposition routine fails.
    pub fn fit_transform<S>(
        &mut self,
        input: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<A>, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        let u = self.inner_fit(input)?;
        Ok(unsafe {
            let mut y: ArrayBase<OwnedRepr<A>, Ix2> =
                ArrayBase::uninitialized((input.nrows(), self.n_components()));
            for (y_row, u_row) in y
                .lanes_mut(Axis(1))
                .into_iter()
                .zip(u.slice(s![.., 0..self.n_components()]).lanes(Axis(1)))
            {
                for (y_v, u_v, sigma_v) in izip!(y_row.into_iter(), u_row, &self.singular) {
                    *y_v = *u_v * A::from_real(*sigma_v);
                }
            }
            y
        })
    }

    /// Fits the model with `input`.
    ///
    /// # Errors
    ///
    /// * `DecompositionError::InvalidInput` if any of the dimensions of `input`
    ///   is less than the number of components.
    /// * `DecompositionError::LinalgError` if the underlying Singular Vector
    ///   Decomposition routine fails.
    fn inner_fit<S>(&mut self, input: &ArrayBase<S, Ix2>) -> Result<Array2<A>, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        if input.shape().iter().any(|v| *v < self.n_components()) {
            return Err(DecompositionError::InvalidInput);
        }

        let means = if let Some(means) = input.mean_axis(Axis(0)) {
            means
        } else {
            return Ok(Array2::<A>::zeros((0, input.ncols())));
        };
        let x = input - &means;
        let (u, sigma, vt) = x.svd(true, true)?;
        let mut u = u.expect("`svd` should return `u`");
        let mut vt = vt.expect("`svd` should return `vt`");
        svd_flip(&mut u, &mut vt);
        self.components = vt.slice(s![0..self.n_components(), ..]).into_owned();
        self.n_samples = input.nrows();
        self.means = means;
        self.singular = sigma;

        Ok(u)
    }
}

/// Makes `SVD`'s output deterministic using the columns of `u` as the basis for
/// sign flipping.
fn svd_flip<A>(u: &mut Array2<A>, v: &mut Array2<A>)
where
    A: Scalar,
{
    for (u_col, v_row) in u.lanes_mut(Axis(0)).into_iter().zip(v.lanes_mut(Axis(1))) {
        let mut u_col_iter = u_col.iter();
        let e = if let Some(e) = u_col_iter.next() {
            *e
        } else {
            continue;
        };
        let mut absmax = e.abs();
        let mut signum = e.re().signum();
        for e in u_col_iter {
            let abs = e.abs();
            if abs <= absmax {
                continue;
            }
            absmax = abs;
            signum = if e.re() == A::zero().re() {
                e.im().signum()
            } else {
                e.re().signum()
            };
        }
        if signum < A::zero().re() {
            let signum = A::from_real(signum);
            for e in u_col {
                *e *= signum;
            }
            for e in v_row {
                *e *= signum;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use float_cmp::approx_eq;
    use ndarray::{arr2, Array2};

    #[test]
    fn pca_zero_component() {
        let mut pca = super::Pca::new(0);

        let x = Array2::<f32>::zeros((0, 5));
        let y = pca.fit_transform(&x).unwrap();
        assert_eq!(y.nrows(), 0);
        assert_eq!(y.ncols(), 0);

        let x = arr2(&[[0_f32, 0_f32], [3_f32, 4_f32], [6_f32, 8_f32]]);
        let y = pca.fit_transform(&x).unwrap();
        assert_eq!(y.nrows(), 3);
        assert_eq!(y.ncols(), 0);
    }

    #[test]
    fn pca_single_sample() {
        let mut pca = super::Pca::new(1);
        let x = arr2(&[[1_f32, 1_f32]]);
        let y = pca.fit_transform(&x).unwrap();
        assert_eq!(y, arr2(&[[0.0]]));
    }

    #[test]
    fn pca() {
        let x = arr2(&[[0_f64, 0_f64], [3_f64, 4_f64], [6_f64, 8_f64]]);
        let mut pca = super::Pca::new(1);
        let y = pca.fit_transform(&x).unwrap();
        assert!(approx_eq!(f64, (y[(0, 0)] - y[(2, 0)]).abs(), 10.));
        assert!(approx_eq!(f64, y[(1, 0)], 0.));

        let mut pca = super::Pca::new(1);
        assert!(pca.fit(&x).is_ok());
        let y = pca.transform(&x).unwrap();
        assert!(approx_eq!(f64, (y[(0, 0)] - y[(2, 0)]).abs(), 10.));
        assert!(approx_eq!(f64, y[(1, 0)], 0.));
    }

    #[test]
    fn pca_explained_variance_ratio() {
        let x = arr2(&[
            [-1_f64, -1_f64],
            [-2_f64, -1_f64],
            [-3_f64, -2_f64],
            [1_f64, 1_f64],
            [2_f64, 1_f64],
            [3_f64, 2_f64],
        ]);
        let mut pca = super::Pca::new(2);
        assert!(pca.fit(&x).is_ok());
        let ratio = pca.explained_variance_ratio();
        assert!(ratio.get(0).unwrap() > &0.99244);
        assert!(ratio.get(1).unwrap() < &0.00756);
    }

    #[test]
    fn svd_flip() {
        let mut u = arr2(&[[2., -1., 3.], [-1., -3., 2.]]);
        let mut v = arr2(&[[1., 1.], [-2., 2.], [3., -3.]]);
        super::svd_flip(&mut u, &mut v);
        assert_eq!(u, arr2(&[[2., 1., 3.], [-1., 3., 2.]]));
        assert_eq!(v, arr2(&[[1., 1.], [2., -2.], [3., -3.]]));
    }
}
