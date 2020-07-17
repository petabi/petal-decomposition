use crate::linalg::{lu_pl, Lapack};
use crate::DecompositionError;
use itertools::izip;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix2, OwnedRepr, ScalarOperand};
use ndarray_linalg::{error::LinalgError, QRInto, Scalar, UVTFlag, SVD, SVDDC};
use num_traits::{real::Real, FromPrimitive};
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::StandardNormal;
#[cfg(target_pointer_width = "32")]
use rand_pcg::Lcg64Xsh32 as Pcg;
#[cfg(not(target_pointer_width = "32"))]
use rand_pcg::Mcg128Xsl64 as Pcg;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp;

/// Principal component analysis.
///
/// This reduces the dimensionality of the input data using Singular Value
/// Decomposition (SVD). The data is centered for each feature before applying
/// SVD.
///
/// # Examples
///
/// ```
/// use petal_decomposition::Pca;
///
/// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [2_f64, 2_f64]]);
/// let y = Pca::new(1).fit_transform(&x).unwrap();  // [-2_f64.sqrt(), 0_f64, 2_f64.sqrt()]
/// assert!((y[(0, 0)].abs() - 2_f64.sqrt()).abs() < 1e-8);
/// assert!(y[(1, 0)].abs() < 1e-8);
/// assert!((y[(2, 0)].abs() - 2_f64.sqrt()).abs() < 1e-8);
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound = "A: Serialize, for<'a> A: Deserialize<'a>")
)]
pub struct Pca<A>
where
    A: Scalar,
{
    components: Array2<A>,
    n_samples: usize,
    means: Array1<A>,
    total_variance: A::Real,
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
        Self {
            components: Array2::<A>::zeros((n_components, 0)),
            n_samples: 0,
            means: Array1::<A>::zeros(0),
            total_variance: A::zero().re(),
            singular: Array1::<A::Real>::zeros(0),
        }
    }

    /// Returns the principal axes in feature space.
    #[inline]
    pub fn components(&self) -> &Array2<A> {
        &self.components
    }

    /// Returns the per-feature empirical mean.
    #[inline]
    pub fn mean(&self) -> &Array1<A> {
        &self.means
    }

    /// Returns the number of components.
    #[inline]
    pub fn n_components(&self) -> usize {
        self.components.nrows()
    }

    /// Returns sigular values.
    #[inline]
    pub fn singular_values(&self) -> &Array1<A::Real> {
        &self.singular
    }

    /// Returns the ratio of explained variance for each component.
    pub fn explained_variance_ratio(&self) -> Array1<A::Real> {
        let mut variance: Array1<A::Real> = &self.singular * &self.singular;
        variance /= self.total_variance;
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

    /// Transforms data back to its original space.
    ///
    /// # Errors
    ///
    /// Returns `DecompositionError::InvalidInput` if the number of rows of
    /// `input` is different from that of the training data, or the number of
    /// columns of `input` is different from the number of components.
    pub fn inverse_transform<S>(
        &self,
        input: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<A>, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        debug_assert_eq!(self.components.ncols(), self.means.len());
        if input.ncols() != self.components.nrows() {
            return Err(DecompositionError::InvalidInput);
        }
        Ok(input.dot(&self.components) + &self.means)
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
        self.total_variance = sigma.dot(&sigma);
        self.components = vt.slice(s![0..self.n_components(), ..]).into_owned();
        self.n_samples = input.nrows();
        self.means = means;
        self.singular = sigma.slice(s![0..self.n_components()]).into_owned();

        Ok(u)
    }
}

/// Principal component analysis using randomized singular value decomposition.
///
/// This uses randomized SVD (singular value decomposition) proposed by Halko et
/// al. [1] to reduce the dimensionality of the input data. The data is centered
/// for each feature before applying randomized SVD.
///
/// # Examples
///
/// ```
/// use petal_decomposition::RandomizedPca;
///
/// let x = ndarray::arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [2_f64, 2_f64]]);
/// let mut pca = RandomizedPca::new(1);
/// let y = pca.fit_transform(&x).unwrap();  // [-2_f64.sqrt(), 0_f64, 2_f64.sqrt()]
/// assert!((y[(0, 0)].abs() - 2_f64.sqrt()).abs() < 1e-8);
/// assert!(y[(1, 0)].abs() < 1e-8);
/// assert!((y[(2, 0)].abs() - 2_f64.sqrt()).abs() < 1e-8);
/// ```
///
/// # References
///
/// 1. N. Halko, P. G. Martinsson, and J. A. Tropp. Finding Structure with
///    Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
///    Decompositions. _SIAM Review,_ 53(2), 217–288, 2011.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        bound = "A: Serialize, for<'a> A: Deserialize<'a>, R: Serialize, for<'a> R: Deserialize<'a>"
    )
)]
#[allow(clippy::module_name_repetitions)]
pub struct RandomizedPca<A, R>
where
    A: Scalar,
    R: Rng,
{
    rng: R,
    components: Array2<A>,
    n_samples: usize,
    means: Array1<A>,
    total_variance: A::Real,
    singular: Array1<A::Real>,
}

impl<A> RandomizedPca<A, Pcg>
where
    A: Scalar + Lapack,
    A::Real: ScalarOperand,
{
    /// Creates a PCA model based on randomized SVD.
    ///
    /// The random matrix for randomized SVD is created from a PCG random number
    /// generator (the XSL 128/64 (MCG) variant on a 64-bit CPU and the XSH RR
    /// 64/32 (LCG) variant on a 32-bit CPU), initialized with a
    /// randomly-generated seed.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        let seed: u128 = rand::thread_rng().gen();
        Self::with_seed(n_components, seed)
    }

    /// Creates a PCA model based on randomized SVD, with a PCG random number
    /// generator initialized with the given seed.
    ///
    /// It uses a PCG random number generator (the XSL 128/64 (MCG) variant on a
    /// 64-bit CPU and the XSH RR 64/32 (LCG) variant on a 32-bit CPU). Use
    /// [`with_rng`] for a different random number generator.
    ///
    /// [`with_rng`]: #method.with_rng
    #[must_use]
    pub fn with_seed(n_components: usize, seed: u128) -> Self {
        let rng = Pcg::from_seed(seed.to_be_bytes());
        Self::with_rng(n_components, rng)
    }
}

impl<A, R> RandomizedPca<A, R>
where
    A: Scalar + Lapack,
    A::Real: ScalarOperand,
    R: Rng,
{
    /// Creates a PCA model with the given number of components and random
    /// number generator. The random number generator is used to create a random
    /// matrix for randomized SVD.
    #[must_use]
    pub fn with_rng(n_components: usize, rng: R) -> Self {
        Self {
            rng,
            components: Array2::<A>::zeros((n_components, 0)),
            n_samples: 0,
            means: Array1::<A>::zeros(0),
            total_variance: A::zero().re(),
            singular: Array1::<A::Real>::zeros(0),
        }
    }

    /// Returns the principal axes in feature space.
    #[inline]
    pub fn components(&self) -> &Array2<A> {
        &self.components
    }

    /// Returns the per-feature empirical mean.
    #[inline]
    pub fn mean(&self) -> &Array1<A> {
        &self.means
    }

    /// Returns the number of components.
    #[inline]
    pub fn n_components(&self) -> usize {
        self.components.nrows()
    }

    /// Returns sigular values.
    #[inline]
    pub fn singular_values(&self) -> &Array1<A::Real> {
        &self.singular
    }

    /// Returns the ratio of explained variance for each component.
    pub fn explained_variance_ratio(&self) -> Array1<A::Real> {
        let mut variance: Array1<A::Real> = &self.singular * &self.singular;
        variance /= self.total_variance;
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

    /// Transforms data back to its original space.
    ///
    /// # Errors
    ///
    /// Returns `DecompositionError::InvalidInput` if the number of rows of
    /// `input` is different from that of the training data, or the number of
    /// columns of `input` is different from the number of components.
    pub fn inverse_transform<S>(
        &self,
        input: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<A>, DecompositionError>
    where
        S: Data<Elem = A>,
    {
        debug_assert_eq!(self.components.ncols(), self.means.len());
        if input.ncols() != self.components.nrows() {
            return Err(DecompositionError::InvalidInput);
        }
        Ok(input.dot(&self.components) + &self.means)
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
        let (u, sigma, vt) = randomized_svd(&x, self.n_components(), &mut self.rng)?;
        self.total_variance = x.iter().fold(A::zero().re(), |var, &e| var + e.square());
        self.components = vt.slice(s![0..self.n_components(), ..]).into_owned();
        self.n_samples = input.nrows();
        self.means = means;
        self.singular = sigma.slice(s![0..self.n_components()]).into_owned();

        Ok(u)
    }
}

type Svd<A> = (Array2<A>, Array1<<A as Scalar>::Real>, Array2<A>);

/// Computes a truncated randomized SVD
fn randomized_svd<A, S, R>(
    input: &ArrayBase<S, Ix2>,
    n_components: usize,
    rng: &mut R,
) -> Result<Svd<A>, LinalgError>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    R: RngCore,
{
    let n_random = n_components + 10; // oversample by 10
    let q = randomized_range_finder(input, n_random, 7, rng)?;
    let b = q.t().dot(input);
    let (u, sigma, vt) = b.svddc(UVTFlag::Some)?;
    let u = u.expect("`svddc` should return `u`");
    let mut vt = vt.expect("`svddc` should return `vt`");
    let mut u = q.dot(&u);
    svd_flip(&mut u, &mut vt);
    Ok((u, sigma, vt))
}

/// Computes an orthonormal matrix whose range approximates the range of `input`.
fn randomized_range_finder<A, S, R>(
    input: &ArrayBase<S, Ix2>,
    size: usize,
    n_iter: usize,
    rng: &mut R,
) -> Result<Array2<A>, LinalgError>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    R: RngCore,
{
    let mut q = ArrayBase::from_shape_fn((input.ncols(), size), |_| {
        let r = A::Real::from_f64(rng.sample(StandardNormal))
            .expect("float to float conversion never fails");
        A::from_real(r)
    });
    let mut pl = q.view();
    q = input.dot(&pl);
    for _ in 0..n_iter {
        lu_pl(&mut q)?;
        pl = q.slice(s![.., 0..cmp::min(q.nrows(), q.ncols())]);
        q = input.t().dot(&pl);
        lu_pl(&mut q)?;
        pl = q.slice(s![.., 0..cmp::min(q.nrows(), q.ncols())]);
        q = input.dot(&pl);
    }
    let (q, _) = q.qr_into()?;
    Ok(q)
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
    use approx::{assert_abs_diff_eq, assert_relative_eq, AbsDiffEq};
    use ndarray::{arr2, Array2};
    use rand::Rng;
    use rand_distr::StandardNormal;
    use rand_pcg::Pcg64Mcg;

    const RNG_SEED: u128 = 1234567891011121314;

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
        assert_eq!(pca.n_components(), 1);

        let y = pca.fit_transform(&x).unwrap();
        assert_abs_diff_eq!(y[(0, 0)].abs(), 5., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(1, 0)], 0., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(2, 0)].abs(), 5., epsilon = 1e-10);
        let z = pca.inverse_transform(&y).expect("valid input");
        assert!(z.abs_diff_eq(&x, 1e-10));

        let mut pca = super::Pca::new(1);
        assert!(pca.fit(&x).is_ok());
        assert_eq!(pca.n_components(), 1);
        assert!(pca.components().abs_diff_eq(&arr2(&[[-0.6, -0.8]]), 1e-10));
        let y = pca.transform(&x).unwrap();
        assert_abs_diff_eq!(y[(0, 0)].abs(), 5., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(1, 0)], 0., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(2, 0)].abs(), 5., epsilon = 1e-10);
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
    #[cfg(feature = "serde")]
    fn pca_serialize() {
        use approx::AbsDiffEq;
        let mut pca = super::Pca::new(1);
        let x = arr2(&[[1_f32, 1_f32]]);
        assert!(pca.fit(&x).is_ok());
        let serialized = serde_json::to_string(&pca).unwrap();
        let deserialized: super::Pca<f32> = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized
            .components()
            .abs_diff_eq(pca.components(), 1e-12));
        assert!(deserialized.mean().abs_diff_eq(pca.mean(), 1e12));
    }

    #[test]
    fn randomized_pca() {
        let x = arr2(&[[0_f64, 0_f64], [3_f64, 4_f64], [6_f64, 8_f64]]);
        let mut pca = super::RandomizedPca::with_seed(1, RNG_SEED);
        assert_eq!(pca.n_components(), 1);

        let res = pca.fit(&x);
        assert!(res.is_ok());
        assert_eq!(pca.n_components(), 1);
        let y = pca.transform(&x).unwrap();
        assert_abs_diff_eq!(y[(0, 0)].abs(), 5., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(1, 0)], 0., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(2, 0)].abs(), 5., epsilon = 1e-10);
        let z = pca.inverse_transform(&y).expect("valid input");
        assert!(z.abs_diff_eq(&x, 1e-10));

        let mut pca = super::RandomizedPca::with_rng(1, rand::thread_rng());
        let y = pca.fit_transform(&x).unwrap();
        assert_abs_diff_eq!(y[(0, 0)].abs(), 5., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(1, 0)], 0., epsilon = 1e-10);
        assert_abs_diff_eq!(y[(2, 0)].abs(), 5., epsilon = 1e-10);
    }

    #[test]
    fn randomized_pca_explained_variance_ratio() {
        let x = arr2(&[
            [-1_f64, -1_f64],
            [-2_f64, -1_f64],
            [-3_f64, -2_f64],
            [1_f64, 1_f64],
            [2_f64, 1_f64],
            [3_f64, 2_f64],
        ]);
        let mut pca = super::RandomizedPca::with_rng(2, rand::thread_rng());
        assert!(pca.fit(&x).is_ok());
        let ratio = pca.explained_variance_ratio();
        assert!(ratio.get(0).unwrap() > &0.99244);
        assert!(ratio.get(1).unwrap() < &0.00756);
    }

    #[test]
    fn randomized_pca_explained_variance_equivalence() {
        let mut rng = Pcg64Mcg::new(RNG_SEED);
        let x = Array2::from_shape_fn((100, 80), |_| rng.sample::<f64, _>(StandardNormal));

        let mut pca = super::Pca::new(2);
        let mut pca_rand = super::RandomizedPca::with_rng(2, rng);

        assert!(pca.fit(&x).is_ok());
        assert!(pca_rand.fit(&x).is_ok());

        for (a, b) in pca
            .explained_variance_ratio()
            .iter()
            .zip(pca_rand.explained_variance_ratio().iter())
        {
            assert_relative_eq!(a, b, max_relative = 0.05);
        }
    }

    #[test]
    fn randomized_pca_singular_values_consistency() {
        let mut rng = Pcg64Mcg::new(RNG_SEED);
        let x = Array2::from_shape_fn((100, 80), |_| rng.sample::<f64, _>(StandardNormal));

        let mut pca = super::Pca::new(2);
        let mut pca_rand = super::RandomizedPca::with_rng(2, rng);

        assert!(pca.fit(&x).is_ok());
        assert!(pca_rand.fit(&x).is_ok());

        for (a, b) in pca
            .singular_values()
            .iter()
            .zip(pca_rand.singular_values().iter())
        {
            assert_relative_eq!(a, b, max_relative = 0.05);
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn randomized_pca_serialize() {
        use approx::AbsDiffEq;
        let mut pca = super::RandomizedPca::with_seed(1, RNG_SEED);
        let x = arr2(&[[1_f32, 1_f32]]);
        assert!(pca.fit(&x).is_ok());
        let serialized = serde_json::to_string(&pca).unwrap();
        let deserialized: super::Pca<f32> = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized
            .components()
            .abs_diff_eq(pca.components(), 1e-12));
        assert!(deserialized.mean().abs_diff_eq(pca.mean(), 1e12));
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
