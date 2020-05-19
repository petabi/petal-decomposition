use crate::DecompositionError;
use itertools::izip;
use ndarray::{s, Array2, ArrayBase, Axis, Data, Ix2, OwnedRepr};
use ndarray_linalg::{Lapack, Scalar, SVD};
use num_traits::real::Real;

/// Principal component analysis.
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
pub struct Pca {
    n_components: usize,
}

impl Pca {
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Pca { n_components }
    }

    /// Applies dimensionality reduction on `input`.
    ///
    /// # Errors
    ///
    /// Returns `LinalgError` if the underlying Singular Vector Decomposition
    /// routine fails.
    pub fn fit_transform<A, S>(
        &mut self,
        input: &ArrayBase<S, Ix2>,
    ) -> Result<ArrayBase<OwnedRepr<A>, Ix2>, DecompositionError>
    where
        A: Scalar + Lapack,
        S: Data<Elem = A>,
    {
        if input.shape().iter().any(|v| *v < self.n_components) {
            return Err(DecompositionError::InvalidInput {
                n_components: self.n_components,
                n_rows: input.shape()[0],
                n_cols: input.shape()[1],
            });
        }

        let x = unsafe {
            let mut x: ArrayBase<OwnedRepr<A>, Ix2> = ArrayBase::uninitialized(input.dim());
            for (input_col, x_col) in input.lanes(Axis(0)).into_iter().zip(x.lanes_mut(Axis(0))) {
                if let Some(mean) = input_col.mean() {
                    for (iv, xv) in input_col.into_iter().zip(x_col) {
                        *xv = *iv - mean;
                    }
                }
            }
            x
        };
        let (u, sigma, _vt) = x.svd(true, false)?;
        let mut u = u.expect("`svd` should return `u`");
        svd_flip(&mut u);
        Ok(unsafe {
            let mut y: ArrayBase<OwnedRepr<A>, Ix2> =
                ArrayBase::uninitialized((input.nrows(), self.n_components));
            for (y_row, u_row) in y
                .lanes_mut(Axis(1))
                .into_iter()
                .zip(u.slice(s![.., 0..self.n_components]).lanes(Axis(1)))
            {
                for (y_v, u_v, sigma_v) in izip!(y_row.into_iter(), u_row, &sigma) {
                    *y_v = *u_v * A::from_real(*sigma_v);
                }
            }
            y
        })
    }
}

/// Makes `SVD`'s output deterministic using the columns of `u` as the basis for
/// sign flipping.
fn svd_flip<A>(u: &mut Array2<A>)
where
    A: Scalar,
{
    for u_col in u.lanes_mut(Axis(0)) {
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
            for e in u_col {
                *e *= A::from_real(signum);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use float_cmp::approx_eq;
    use ndarray::arr2;

    #[test]
    fn pca() {
        let x = arr2(&[[0_f64, 0_f64], [3_f64, 4_f64], [6_f64, 8_f64]]);
        let mut pca = super::Pca::new(1);
        let y = pca.fit_transform(&x).unwrap();
        assert!(approx_eq!(f64, (y[(0, 0)] - y[(2, 0)]).abs(), 10.));
        assert!(approx_eq!(f64, y[(1, 0)], 0.));
    }

    #[test]
    fn svd_flip() {
        let mut u = arr2(&[[2., -1., 3.], [-1., -3., 2.]]);
        super::svd_flip(&mut u);
        assert_eq!(u, arr2(&[[2., 1., 3.], [-1., 3., 2.]]));
    }
}
