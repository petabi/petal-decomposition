use crate::DecompositionError;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix2};
use ndarray_linalg::{Eigh, Lapack, Scalar, SVD, UPLO};
use num_traits::FromPrimitive;
use rand::Rng;
use rand_distr::StandardNormal;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp;
use std::iter::FromIterator;

/// Independent component analysis using the [FastICA] algorithm.
///
/// [FastICA]: https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
///
/// # Examples
///
/// ```
/// use petal_decomposition::FastIca;
///
/// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [1_f64, -1_f64]]);
/// let mut ica = FastIca::new(rand::thread_rng());
/// let y = ica.fit_transform(&x).unwrap();
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        bound = "A: Serialize, for<'a> A: Deserialize<'a>, R: Serialize, for<'a> R: Deserialize<'a>"
    )
)]
#[allow(clippy::module_name_repetitions)]
pub struct FastIca<A, R>
where
    A: Scalar,
    R: Rng,
{
    rng: R,
    components: Array2<A>,
    means: Array1<A>,
    n_iter: usize,
}

impl<A, R> FastIca<A, R>
where
    A: Scalar + Lapack,
    R: Rng,
{
    /// Creates an ICA model.
    #[must_use]
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            components: Array2::<A>::zeros((0, 0)),
            means: Array1::<A>::zeros(0),
            n_iter: 0,
        }
    }

    /// Fits the model with `input`.
    ///
    /// # Errors
    ///
    /// * `DecompositionError::LinalgError` if the underlying Singular Vector
    ///   Decomposition routine fails.
    pub fn fit<S>(&mut self, input: &ArrayBase<S, Ix2>) -> Result<(), DecompositionError>
    where
        S: Data<Elem = A>,
    {
        self.inner_fit(input)?;
        Ok(())
    }

    /// Applies ICA to `input`.
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

    /// Fits the model with `input` and apply ICA on `input`.
    ///
    /// This is equivalent to calling both [`fit`] and [`transform`] for the
    /// same input, but more efficient.
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
        let x = self.inner_fit(input)?;
        Ok(self.components.dot(&x).t().to_owned())
    }

    fn inner_fit<S>(&mut self, input: &ArrayBase<S, Ix2>) -> Result<Array2<A>, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        let n_components = cmp::min(input.nrows(), input.ncols());
        let means = if let Some(means) = input.mean_axis(Axis(0)) {
            means
        } else {
            return Ok(Array2::<A>::zeros((0, input.ncols())));
        };
        let n_features = input.nrows();
        let x = unsafe {
            let mut x: Array2<A> = ArrayBase::uninitialized((input.ncols(), input.nrows()));
            for (mut x_row, (input_col, col_mean)) in x
                .lanes_mut(Axis(1))
                .into_iter()
                .zip(input.lanes(Axis(0)).into_iter().zip(means.iter()))
            {
                for (x_elem, input_elem) in x_row.iter_mut().zip(input_col.iter()) {
                    *x_elem = *input_elem - *col_mean;
                }
            }
            x
        };
        let (u, sigma, _) = x.svd(true, false)?;
        let u = u.expect("`svd` should return `u`");
        let k = unsafe {
            let mut x: Array2<A> = ArrayBase::uninitialized((n_components, u.ncols()));
            for ((u_col, sigma_elem), mut x_row) in u
                .lanes(Axis(0))
                .into_iter()
                .zip(sigma.into_iter())
                .take(n_components)
                .zip(x.lanes_mut(Axis(1)).into_iter())
            {
                let d = A::from_real(*sigma_elem);
                for (u_elem, x_elem) in u_col.iter().take(n_components).zip(x_row.iter_mut()) {
                    *x_elem = *u_elem / d;
                }
            }
            x
        };
        let mut x1 = k.dot(&x);
        let n_features_sqrt = A::from_usize(n_features).expect("approximation").sqrt();
        for x_elem in x1.iter_mut() {
            *x_elem *= n_features_sqrt;
        }

        let w_init = Array2::<A>::from_shape_fn((n_components, n_components), |_| {
            let r = A::Real::from_f64(self.rng.sample(StandardNormal))
                .expect("float to float conversion never fails");
            A::from_real(r)
        });

        let (w, n_iter) = ica_par(&x1, A::from_f64(1e-4).expect("float").re(), 200, &w_init);
        self.components = w.dot(&k);
        self.means = means;
        self.n_iter = n_iter;
        Ok(x)
    }
}

fn ica_par<A, S>(
    input: &ArrayBase<S, Ix2>,
    tol: A::Real,
    max_iter: usize,
    w_init: &ArrayBase<S, Ix2>,
) -> (Array2<A>, usize)
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    let mut w = symmetric_decorrelation(w_init);
    let p_inv = A::one() / A::from_usize(input.ncols()).expect("approximation");
    for i in 0..max_iter {
        let (gwtx, g_wtx) = logcosh(w.dot(input));
        let mut gwtx_dot = gwtx.dot(&input.t());
        for (mut dot_row, (&g_wtx_elem, w_row)) in gwtx_dot
            .lanes_mut(Axis(1))
            .into_iter()
            .zip(g_wtx.iter().zip(w.lanes(Axis(1)).into_iter()))
        {
            for (dot_elem, w_elem) in dot_row.iter_mut().zip(w_row.iter()) {
                *dot_elem = *dot_elem * p_inv - g_wtx_elem * *w_elem;
            }
        }
        let w1 = symmetric_decorrelation(&gwtx_dot);
        let mut lim = A::zero().re();
        for d in w1
            .lanes(Axis(1))
            .into_iter()
            .zip(w.lanes(Axis(0)).into_iter())
            .map(|(w1_row, w_col)| (w1_row.dot(&w_col).abs() - A::one().re()).abs())
        {
            if d > lim {
                lim = d;
            }
        }
        if lim < tol {
            return (w1, i + 1);
        }
        w = w1;
    }
    (w, max_iter)
}

fn symmetric_decorrelation<A, S>(input: &ArrayBase<S, Ix2>) -> Array2<A>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    let (e, mut v) = input.dot(&input.t()).eigh(UPLO::Lower).unwrap();
    let v_t = v.t().to_owned();
    let e_sqrt_inv = Array::from_iter(
        e.iter()
            .map(|r| Scalar::from_real(A::one().re() / r.sqrt())),
    );
    for mut row in v.lanes_mut(Axis(1)) {
        for (v_e, s_e) in row.iter_mut().zip(e_sqrt_inv.iter()) {
            *v_e *= *s_e;
        }
    }
    v.dot(&v_t).dot(input)
}

fn logcosh<A, S>(mut input: ArrayBase<S, Ix2>) -> (ArrayBase<S, Ix2>, Array1<A>)
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    for elem in input.iter_mut() {
        *elem = elem.tanh();
    }
    let ncols = A::from_usize(input.ncols()).expect("approximation");
    let g_x = Array::from_iter(
        input
            .lanes(Axis(1))
            .into_iter()
            .map(|row| row.iter().map(|&elem| A::one() - elem * elem).sum::<A>() / ncols),
    );
    (input, g_x)
}

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::arr2;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    const RNG_SEED: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    #[test]
    fn fast_ica_fit_transform() {
        let rng = Pcg32::from_seed(RNG_SEED);
        let x = arr2(&[[0., 0.], [1., 1.], [1., -1.]]);
        let mut ica = super::FastIca::new(rng);
        assert!(ica.fit(&x).is_ok());
        assert_eq!(ica.n_iter, 1);
        let result_fit = ica.transform(&x).unwrap();

        let rng = Pcg32::from_seed(RNG_SEED);
        let mut ica = super::FastIca::new(rng);
        let result_fit_transform = ica.fit_transform(&x).unwrap();
        assert_eq!(ica.n_iter, 1);

        assert_eq!(result_fit, result_fit_transform);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn fast_ica_serialize() {
        use approx::AbsDiffEq;
        let rng = Pcg32::from_seed(RNG_SEED);
        let x = arr2(&[[0., 0.], [1., 1.], [1., -1.]]);
        let mut ica = super::FastIca::new(rng);
        assert!(ica.fit(&x).is_ok());
        let serialized = serde_json::to_string(&ica).unwrap();
        let deserialized: super::FastIca<f64, Pcg32> = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized.components.abs_diff_eq(&ica.components, 1e-12));
        assert!(deserialized.means.abs_diff_eq(&ica.means, 1e12));
    }

    #[test]
    fn ica_par_single_iter() {
        let x = arr2(&[[-0.5, 0.5], [-0.3, 0.3]]);
        let w = arr2(&[[1., 2.], [3., 4.]]);
        let (y, n) = super::ica_par(&x, 0.5, 1, &w);
        assert_abs_diff_eq!(y[(0, 0)], 0.51449576, epsilon = 1e-8);
        assert_abs_diff_eq!(y[(0, 1)], -0.85749293, epsilon = 1e-8);
        assert_abs_diff_eq!(y[(1, 0)], -0.85749293, epsilon = 1e-8);
        assert_abs_diff_eq!(y[(1, 1)], -0.51449576, epsilon = 1e-8);
        assert_eq!(n, 1);
    }

    #[test]
    fn ica_par_multi_iter() {
        let x = arr2(&[[1., -1.], [0., 0.]]);
        let w = arr2(&[[1., 2.], [3., 4.]]);
        let (y, n) = super::ica_par(&x, 1e-4, 200, &w);
        assert_abs_diff_eq!(y[(0, 0)], -0.00172682, epsilon = 1e-8);
        assert_abs_diff_eq!(y[(0, 1)], 0.99999851, epsilon = 1e-8);
        assert_abs_diff_eq!(y[(1, 0)], 0.99999851, epsilon = 1e-8);
        assert_abs_diff_eq!(y[(1, 1)], 0.00172682, epsilon = 1e-8);
        assert_eq!(n, 6);
    }

    #[test]
    fn logcosh() {
        let x = arr2(&[[1., 2.], [3., 4.]]);
        let (x, y) = super::logcosh(x);
        assert_relative_eq!(x[(0, 0)], 0.76159416, max_relative = 1e-8);
        assert_relative_eq!(x[(0, 1)], 0.96402758, max_relative = 1e-8);
        assert_relative_eq!(x[(1, 0)], 0.99505475, max_relative = 1e-8);
        assert_relative_eq!(x[(1, 1)], 0.99932930, max_relative = 1e-8);
        assert_relative_eq!(y[0], 0.24531258, max_relative = 1e-6);
        assert_relative_eq!(y[1], 0.00560349, max_relative = 1e-6);
    }

    #[test]
    fn symmetric_decorrelation() {
        let x = arr2(&[[33., 24.], [48., 57.]]);
        let w = super::symmetric_decorrelation(&x);
        assert_relative_eq!(w[(0, 0)], 0.96623494, max_relative = 1e-8);
        assert_relative_eq!(w[(0, 1)], -0.25766265, max_relative = 1e-8);
        assert_relative_eq!(w[(1, 0)], 0.25766265, max_relative = 1e-8);
        assert_relative_eq!(w[(1, 1)], 0.96623494, max_relative = 1e-8);
    }
}
