use crate::{
    linalg::{self, eigh, svd, Lapack},
    DecompositionError,
};
use lair::{Real, Scalar};
use ndarray::{Array1, Array2, ArrayBase, AssignElem, Axis, Data, DataMut, Ix2};
use num_traits::{Float, FromPrimitive};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
#[cfg(target_pointer_width = "32")]
use rand_pcg::Lcg64Xsh32 as Pcg;
#[cfg(not(target_pointer_width = "32"))]
use rand_pcg::Mcg128Xsl64 as Pcg;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp;

/// Independent component analysis using the [FastICA] algorithm.
///
/// [FastICA]: https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
///
/// # Examples
///
/// ```
/// use petal_decomposition::FastIcaBuilder;
///
/// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [1_f64, -1_f64]]);
/// let mut ica = FastIcaBuilder::new().build();
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
pub struct FastIca<A, R = Pcg>
where
    A: Scalar,
    R: Rng,
{
    rng: R,
    components: Array2<A>,
    means: Array1<A>,
    n_iter: usize,
}

impl<A> FastIca<A, Pcg>
where
    A: Scalar,
{
    /// Creates an ICA model with a random seed.
    ///
    /// It uses a PCG random number generator (the XSL 128/64 (MCG) variant on a
    /// 64-bit CPU and the XSH RR 64/32 (LCG) variant on a 32-bit CPU),
    /// initialized with a randomly-generated seed.
    #[must_use]
    pub fn new() -> Self {
        let seed: u128 = rand::thread_rng().gen();
        Self::with_seed(seed)
    }

    /// Creates an ICA model with the given seed for random number generation.
    ///
    /// It uses a PCG random number generator (the XSL 128/64 (MCG) variant on a
    /// 64-bit CPU and the XSH RR 64/32 (LCG) variant on a 32-bit CPU). Use
    /// [`with_rng`] for a different random number generator.
    ///
    /// [`with_rng`]: #method.with_rng
    #[must_use]
    pub fn with_seed(seed: u128) -> Self {
        let rng = Pcg::from_seed(seed.to_be_bytes());
        Self::with_rng(rng)
    }
}

impl<A, R> FastIca<A, R>
where
    A: Scalar,
    R: Rng,
{
    /// Creates an ICA model with the given random number generator.
    #[must_use]
    pub fn with_rng(rng: R) -> Self {
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
    /// * [`DecompositionError::InvalidInput`] if the layout of `input` is
    ///   incompatible with LAPACK.
    /// * [`DecompositionError::LinalgError`] if the underlying Singular Vector
    ///   Decomposition routine fails.
    pub fn fit<S>(&mut self, input: &ArrayBase<S, Ix2>) -> Result<(), DecompositionError>
    where
        A: Lapack,
        S: Data<Elem = A>,
    {
        self.inner_fit(input)?;
        Ok(())
    }

    /// Applies ICA to `input`.
    ///
    /// # Errors
    ///
    /// * [`DecompositionError::InvalidInput`] if the number of features in
    ///   `input` does not match that of the training data.
    pub fn transform<S>(&self, input: &ArrayBase<S, Ix2>) -> Result<Array2<A>, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        if input.ncols() != self.means.len() {
            return Err(DecompositionError::InvalidInput(
                "too many columns".to_string(),
            ));
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
    /// * [`DecompositionError::InvalidInput`] if the layout of `input` is
    ///   incompatible with LAPACK.
    /// * [`DecompositionError::LinalgError`] if the underlying Singular Vector
    ///   Decomposition routine fails.
    pub fn fit_transform<S>(
        &mut self,
        input: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<A>, DecompositionError>
    where
        A: Lapack,
        S: Data<Elem = A>,
    {
        let x = self.inner_fit(input)?;
        Ok(self.components.dot(&x).t().to_owned())
    }

    /// Fits the model with `input`.
    ///
    /// # Errors
    ///
    /// * [`DecompositionError::InvalidInput`] if the layout of `input` is
    ///   incompatible with LAPACK.
    /// * [`DecompositionError::LinalgError`] if the underlying Singular Vector
    ///   Decomposition routine fails.
    fn inner_fit<S>(&mut self, input: &ArrayBase<S, Ix2>) -> Result<Array2<A>, linalg::Error>
    where
        A: Lapack,
        A::Real: Float,
        S: Data<Elem = A>,
    {
        let n_components = cmp::min(input.nrows(), input.ncols());
        let Some(means) = input.mean_axis(Axis(0)) else {
            return Ok(Array2::<A>::zeros((0, input.ncols())));
        };
        let n_features = input.nrows();
        let mut x = Array2::<A>::uninit((input.ncols(), input.nrows()));
        for (mut x_row, (input_col, col_mean)) in x
            .lanes_mut(Axis(1))
            .into_iter()
            .zip(input.lanes(Axis(0)).into_iter().zip(means.iter()))
        {
            for (x_elem, input_elem) in x_row.iter_mut().zip(input_col.iter()) {
                x_elem.assign_elem(*input_elem - *col_mean);
            }
        }
        let x = unsafe { x.assume_init() };
        let (u, sigma, _) = svd(&mut x.clone(), false)?;
        let mut k = Array2::<A>::uninit((n_components, u.ncols()));
        for ((u_col, sigma_elem), mut k_row) in u
            .lanes(Axis(0))
            .into_iter()
            .zip(sigma.into_iter())
            .take(n_components)
            .zip(k.lanes_mut(Axis(1)).into_iter())
        {
            let d = sigma_elem.into();
            for (u_elem, k_elem) in u_col.iter().take(n_components).zip(k_row.iter_mut()) {
                k_elem.assign_elem(*u_elem / d);
            }
        }
        let k = unsafe { k.assume_init() };
        let mut x1 = k.dot(&x);
        let n_features_sqrt = A::from_usize(n_features).expect("approximation").sqrt();
        for x_elem in &mut x1 {
            *x_elem *= n_features_sqrt;
        }

        let w_init = Array2::<A>::from_shape_fn((n_components, n_components), |_| {
            let r = A::Real::from_f64(self.rng.sample(StandardNormal))
                .expect("float to float conversion never fails");
            r.into()
        });

        let (w, n_iter) = ica_par(&x1, A::from_f64(1e-4).expect("float").re(), 200, &w_init);
        self.components = w.dot(&k);
        self.means = means;
        self.n_iter = n_iter;
        Ok(x)
    }
}

impl<A> Default for FastIca<A, Pcg>
where
    A: Scalar,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`FastIca`].
///
/// # Examples
///
/// ```
/// use petal_decomposition::FastIcaBuilder;
///
/// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64]]);
/// let mut ica = FastIcaBuilder::new().build();
/// ica.fit(&x);
/// ```
pub struct FastIcaBuilder<R> {
    rng: R,
}

impl FastIcaBuilder<Pcg> {
    /// Sets the number of components for PCA.
    ///
    /// It uses a PCG random number generator (the XSL 128/64 (MCG) variant on a
    /// 64-bit CPU and the XSH RR 64/32 (LCG) variant on a 32-bit CPU),
    /// initialized with a randomly-generated seed.
    #[must_use]
    pub fn new() -> Self {
        let seed: u128 = rand::thread_rng().gen();
        Self {
            rng: Pcg::from_seed(seed.to_be_bytes()),
        }
    }

    /// Initialized the PCG random number genernator with the given seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use petal_decomposition::FastIcaBuilder;
    ///
    /// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64]]);
    /// let mut ica = FastIcaBuilder::new().seed(1234567891011121314).build();
    /// ica.fit(&x);
    /// ```
    #[must_use]
    pub fn seed(mut self, seed: u128) -> Self {
        self.rng = Pcg::from_seed(seed.to_be_bytes());
        self
    }
}

impl<R: Rng> FastIcaBuilder<R> {
    /// Sets the random number generator for FastICA.
    ///
    /// # Examples
    ///
    /// ```
    /// use petal_decomposition::FastIcaBuilder;
    /// use rand_pcg::Pcg64;
    ///
    /// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64]]);
    /// let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    /// let mut ica = FastIcaBuilder::with_rng(rng).build();
    /// ica.fit(&x);
    /// ```
    #[must_use]
    pub fn with_rng(rng: R) -> Self {
        Self { rng }
    }

    /// Creates an instance of [`FastIca`].
    pub fn build<A: Scalar>(self) -> FastIca<A, R> {
        FastIca {
            rng: self.rng,
            components: Array2::<A>::zeros((0, 0)),
            means: Array1::<A>::zeros(0),
            n_iter: 0,
        }
    }
}

impl Default for FastIcaBuilder<Pcg> {
    fn default() -> Self {
        let seed: u128 = rand::thread_rng().gen();
        Self {
            rng: Pcg::from_seed(seed.to_be_bytes()),
        }
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
    A::Real: Real,
    S: Data<Elem = A>,
{
    let (e, mut v) = eigh(input.dot(&input.t())).unwrap();
    let v_t = v.t().to_owned();
    let e_sqrt_inv: Array1<A> = e
        .iter()
        .map(|r| (A::one().re() / r.sqrt()).into())
        .collect();
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
    for elem in &mut input {
        *elem = elem.tanh();
    }
    let ncols = A::from_usize(input.ncols()).expect("approximation");
    let g_x: Array1<A> = input
        .lanes(Axis(1))
        .into_iter()
        .map(|row| row.iter().map(|&elem| A::one() - elem * elem).sum::<A>() / ncols)
        .collect();
    (input, g_x)
}

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::arr2;

    const RNG_SEED: u128 = 1234567891011121314;

    #[test]
    fn fast_ica_fit_transform() {
        let x = arr2(&[[0., 0.], [1., 1.], [1., -1.]]);
        let mut ica = super::FastIca::with_seed(RNG_SEED);
        assert!(ica.fit(&x).is_ok());
        assert_eq!(ica.n_iter, 1);
        let result_fit = ica.transform(&x).unwrap();

        let mut ica = super::FastIca::with_seed(RNG_SEED);
        let result_fit_transform = ica.fit_transform(&x).unwrap();
        assert_eq!(ica.n_iter, 1);

        assert_eq!(result_fit, result_fit_transform);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn fast_ica_serialize() {
        let x = arr2(&[[0., 0.], [1., 1.], [1., -1.]]);
        let mut ica = super::FastIca::new();
        assert!(ica.fit(&x).is_ok());
        let serialized = serde_json::to_string(&ica).unwrap();
        let deserialized: super::FastIca<f64> = serde_json::from_str(&serialized).unwrap();
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
