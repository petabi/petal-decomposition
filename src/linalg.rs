mod lapack;

pub(crate) use crate::linalg::lapack::Lapack;
use crate::DecompositionError;
use lair::Scalar;
use ndarray::{s, Array1, Array2, ArrayBase, DataMut, Ix2};
use std::cmp;
use std::convert::TryFrom;

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
