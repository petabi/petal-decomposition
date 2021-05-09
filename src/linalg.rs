mod lapack;

use crate::linalg::lapack::Lapack;
use crate::DecompositionError;
use cauchy::Scalar;
use ndarray::{s, Array1, Array2, ArrayBase, DataMut, Ix2};
use num_complex::{Complex32, Complex64};
use std::cmp;
use std::convert::TryFrom;

type Pivot = Vec<i32>;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("invalid layout: {0}")]
    InvalidLayout(#[from] LayoutError),
    #[error("{0}")]
    OperationFailure(String),
}

impl From<Error> for DecompositionError {
    fn from(e: Error) -> Self {
        match e {
            Error::InvalidLayout(e) => DecompositionError::InvalidInput(e.to_string()),
            Error::OperationFailure(reason) => DecompositionError::LinalgError(reason),
        }
    }
}

/// Computes P * L after LU decomposition.
///
/// # Panics
///
/// * Any dimension of `m` is greater than `i32::MAX`
/// * `m`'s memory layout is not contiguous or not in the standard layout.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub(crate) fn lu_pl<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<(), LayoutError>
where
    A: Scalar + LuPiv,
    S: DataMut<Elem = A>,
{
    assert!(a.is_standard_layout());
    let nrows = i32::try_from(a.nrows())
        .map_err(|_| LayoutError::TooManyRows(format!("{} > {}", a.nrows(), i32::MAX)))?;
    let ncols = i32::try_from(a.ncols())
        .map_err(|_| LayoutError::TooManyColumns(format!("{} > {}", a.ncols(), i32::MAX)))?;
    let a_ptr = a
        .as_slice_memory_order_mut()
        .ok_or(LayoutError::NotContiguous)?;
    let mut pivots = unsafe { A::lupiv(nrows, ncols, a_ptr) };
    if pivots.len() < a.nrows() {
        pivots.extend(pivots.len() as i32 + 1..=a.nrows() as i32);
    }
    for i in (0..pivots.len()).rev() {
        pivots[i] -= 1;
        let target = pivots[i] as usize;
        if i == target {
            continue;
        }
        pivots[i] = pivots[target];
        pivots[target] = i as i32;
    }

    let mut pl = a.slice_mut(s![.., 0..cmp::min(a.nrows(), a.ncols())]);
    let mut dst = 0;
    let mut i = dst;
    loop {
        let src = pivots[dst] as usize;
        for k in 0..cmp::min(src, pl.ncols()) {
            pl[[dst, k]] = pl[[src, k]];
        }
        if src < pl.ncols() {
            pl[[dst, src]] = A::one();
        }
        for k in src + 1..pl.ncols() {
            pl[[dst, k]] = A::zero();
        }
        pivots[dst] = pivots.len() as i32; // completed
        if pivots[src] == pivots.len() as i32 {
            dst = i + 1;
            while dst < pivots.len() && pivots[dst] == pivots.len() as i32 {
                dst += 1;
            }
            if dst == pivots.len() {
                break;
            }
            i = dst;
        } else {
            dst = src;
        }
    }
    Ok(())
}

pub trait LuPiv: Lapack + Sized {
    unsafe fn lupiv(m: i32, n: i32, a: &mut [Self]) -> Pivot;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path) => {
        impl LuPiv for $scalar {
            unsafe fn lupiv(m: i32, n: i32, a: &mut [Self]) -> Pivot {
                let k = ::std::cmp::min(m, n);
                let mut ipiv = vec![0; k as usize];
                let info = $getrf(lapacke::Layout::RowMajor, m, n, a, n, &mut ipiv);
                if info >= 0 {
                    ipiv
                } else {
                    unreachable!("valid getrf parameters")
                }
            }
        }
    };
}

impl_solve!(f64, lapacke::dgetrf);
impl_solve!(f32, lapacke::sgetrf);
impl_solve!(Complex64, lapacke::zgetrf);
impl_solve!(Complex32, lapacke::cgetrf);

#[derive(Debug, thiserror::Error)]
pub(crate) enum LayoutError {
    #[error("memory not contiguous")]
    NotContiguous,
    #[error("too many columns: {0}")]
    TooManyColumns(String),
    #[error("too many rows: {0}")]
    TooManyRows(String),
}

pub(crate) fn eigh<A, S>(mut a: ArrayBase<S, Ix2>) -> Result<(Array1<A::Real>, Array2<A>), Error>
where
    A: Lapack,
    S: DataMut<Elem = A>,
{
    if !a.is_square() {
        let reason = if a.nrows() > a.ncols() {
            LayoutError::TooManyRows("more rows than columns".to_string())
        } else {
            LayoutError::TooManyColumns("more columns than rows".to_string())
        };
        return Err(Error::InvalidLayout(reason));
    }
    let n = i32::try_from(a.nrows())
        .map_err(|_| LayoutError::TooManyRows(format!("{} > {}", a.nrows(), i32::MAX)))?;
    let a_ptr = a
        .as_slice_memory_order_mut()
        .ok_or(LayoutError::NotContiguous)?;
    let ev = unsafe { A::heev(b'V', b'L', n, a_ptr, n) }
        .map_err(|_| Error::OperationFailure("cannot compute eigenvalues".to_string()))?;
    Ok((ArrayBase::from(ev), a.to_owned()))
}

pub(crate) type SvdOutput<A> = (Array2<A>, Array1<<A as Scalar>::Real>, Option<Array2<A>>);

/// Calls gesvd.
///
/// # Panics
///
/// Panics if `a`'s memory layout is not contiguous or not in the standard
/// layout.
pub(crate) fn svd<A, S>(a: &mut ArrayBase<S, Ix2>, calc_vt: bool) -> Result<SvdOutput<A>, Error>
where
    A: Lapack,
    S: DataMut<Elem = A>,
{
    assert!(a.is_standard_layout());
    let nrows = i32::try_from(a.nrows())
        .map_err(|_| LayoutError::TooManyRows(format!("{} > {}", a.nrows(), i32::MAX)))?;
    let ncols = i32::try_from(a.ncols())
        .map_err(|_| LayoutError::TooManyColumns(format!("{} > {}", a.ncols(), i32::MAX)))?;
    let a_ptr = a
        .as_slice_memory_order_mut()
        .ok_or(LayoutError::NotContiguous)?;
    let output = unsafe { A::gesvd(if calc_vt { b'A' } else { b'N' }, nrows, ncols, a_ptr) }
        .map_err(|_| Error::OperationFailure("did not converge".to_string()))?;
    let u = ArrayBase::from_shape_vec((a.nrows(), a.nrows()), output.1).expect("valid shape");
    let sigma = ArrayBase::from(output.0);
    let vt = output
        .2
        .map(|vt| ArrayBase::from_shape_vec((a.ncols(), a.ncols()), vt).expect("valid shape"));
    Ok((u, sigma, vt))
}

pub(crate) type SvddcOutput<A> = (Array2<A>, Array1<<A as Scalar>::Real>, Array2<A>);

/// Calls gesdd.
///
/// # Panics
///
/// Panics if `a`'s memory layout is not contiguous or not in the standard
/// layout.
pub(crate) fn svddc<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<SvddcOutput<A>, Error>
where
    A: Lapack,
    S: DataMut<Elem = A>,
{
    assert!(a.is_standard_layout());
    let nrows = i32::try_from(a.nrows())
        .map_err(|_| LayoutError::TooManyRows(format!("{} > {}", a.nrows(), i32::MAX)))?;
    let ncols = i32::try_from(a.ncols())
        .map_err(|_| LayoutError::TooManyColumns(format!("{} > {}", a.ncols(), i32::MAX)))?;
    let a_ptr = a
        .as_slice_memory_order_mut()
        .ok_or(LayoutError::NotContiguous)?;
    let output = unsafe { A::gesdd(nrows, ncols, a_ptr) }
        .map_err(|_| Error::OperationFailure("did not converge".to_string()))?;
    let k = cmp::min(a.nrows(), a.ncols());
    let u = ArrayBase::from_shape_vec((a.nrows(), k), output.1).expect("valid shape");
    let sigma = ArrayBase::from(output.0);
    let vt = ArrayBase::from_shape_vec((k, a.ncols()), output.2.expect("`vt` requested"))
        .expect("valid shape");
    Ok((u, sigma, vt))
}

/// # Panics
///
/// Panics if `a` is not in the standard layout.
pub(crate) fn qr<A, S>(mut a: ArrayBase<S, Ix2>) -> Result<Array2<A>, LayoutError>
where
    A: Lapack,
    S: DataMut<Elem = A>,
{
    assert!(a.is_standard_layout());
    let nrows = i32::try_from(a.nrows())
        .map_err(|_| LayoutError::TooManyRows(format!("{} > {}", a.nrows(), i32::MAX)))?;
    let ncols = i32::try_from(a.ncols())
        .map_err(|_| LayoutError::TooManyColumns(format!("{} > {}", a.ncols(), i32::MAX)))?;
    let a_ptr = a
        .as_slice_memory_order_mut()
        .ok_or(LayoutError::NotContiguous)?;
    let tau = unsafe { A::gelqf(ncols, nrows, a_ptr, ncols) }.expect("valid lapack parameters");
    let k = cmp::min(nrows, ncols);
    unsafe { A::unglq(k, nrows, k, a_ptr, ncols, &tau) }.expect("valid lapack parameters");
    let q = a
        .slice(s![.., ..usize::try_from(k).expect("valid usize")])
        .to_owned();
    Ok(q)
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, s};
    use std::cmp;

    #[test]
    fn lu_pl_identity_l() {
        let p = [
            [0_f64, 1_f64, 0_f64],
            [0_f64, 0_f64, 1_f64],
            [1_f64, 0_f64, 0_f64],
        ];
        let mut m = arr2(&p);
        super::lu_pl(&mut m).unwrap();
        let pl = m.slice(s![.., 0..cmp::min(m.nrows(), m.ncols())]);
        assert_eq!(pl, arr2(&p));
    }

    #[test]
    fn lu_pl_square() {
        let mut m = arr2(&[
            [0_f64, 1_f64, 2_f64],
            [1_f64, 2_f64, 3_f64],
            [2_f64, 3_f64, 4_f64],
        ]);
        super::lu_pl(&mut m).unwrap();
        let pl = m.slice(s![.., 0..cmp::min(m.nrows(), m.ncols())]);
        assert_eq!(pl, arr2(&[[0., 1., 0.], [0.5, 0.5, 1.], [1., 0., 0.]]));
    }

    #[test]
    fn lu_pl_wide_u() {
        let mut m = arr2(&[[0_f64, 1_f64, 2_f64], [1_f64, 2_f64, 3_f64]]);
        super::lu_pl(&mut m).unwrap();
        let pl = m.slice(s![.., 0..cmp::min(m.nrows(), m.ncols())]);
        assert_eq!(pl, arr2(&[[0., 1.], [1., 0.]]));
    }

    #[test]
    fn lu_pl_tall_l() {
        let mut m = arr2(&[[0_f64, 1_f64], [1_f64, 2_f64], [2_f64, 3_f64]]);
        super::lu_pl(&mut m).unwrap();
        let pl = m.slice(s![.., 0..cmp::min(m.nrows(), m.ncols())]);
        assert_eq!(pl, arr2(&[[0., 1.], [0.5, 0.5], [1., 0.]]));
    }

    #[test]
    fn lu_pl_singular() {
        let mut m = arr2(&[[0_f64, 0_f64], [3_f64, 4_f64], [6_f64, 8_f64]]);
        super::lu_pl(&mut m).unwrap();
        let pl = m.slice(s![.., 0..cmp::min(m.nrows(), m.ncols())]);
        assert_eq!(pl, arr2(&[[0., 0.], [0.5, 1.], [1., 0.]]));
    }
}
