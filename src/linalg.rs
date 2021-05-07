use lax::error::Error as LaxError;
use lax::{layout::MatrixLayout, UVTFlag, SVDDC_};
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix2, ShapeBuilder, ShapeError};
use ndarray_linalg::{c32, c64, Scalar};
use std::cmp;
use std::convert::TryFrom;
use std::num::TryFromIntError;

type Pivot = Vec<i32>;

/// Computes P * L after LU decomposition.
///
/// # Panics
///
/// * Any dimension of `m` is greater than `i32::MAX`
/// * `m`'s memory layout is not contiguous
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub(crate) fn lu_pl<A, S>(m: &mut ArrayBase<S, Ix2>) -> Result<(), LaxError>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    let layout = lax_layout(m).unwrap();
    let a = m.as_slice_memory_order_mut().expect("contiguous");
    let mut pivots = unsafe { A::lupiv(layout, a) }?;
    if pivots.len() < m.nrows() {
        pivots.extend(pivots.len() as i32 + 1..=m.nrows() as i32);
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

    let mut pl = m.slice_mut(s![.., 0..cmp::min(m.nrows(), m.ncols())]);
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

pub trait Lapack: ndarray_linalg::Lapack + Sized {
    unsafe fn lupiv(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot, LaxError>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path) => {
        impl Lapack for $scalar {
            unsafe fn lupiv(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot, LaxError> {
                let (row, col) = l.size();
                let k = ::std::cmp::min(row, col);
                let mut ipiv = vec![0; k as usize];
                let layout = match l {
                    MatrixLayout::C { .. } => lapacke::Layout::RowMajor,
                    MatrixLayout::F { .. } => lapacke::Layout::ColumnMajor,
                };
                let info = $getrf(layout, row, col, a, l.lda(), &mut ipiv);
                if info >= 0 {
                    Ok(ipiv)
                } else {
                    Err(LaxError::LapackInvalidValue { return_code: info })
                }
            }
        }
    };
}

impl_solve!(f64, lapacke::dgetrf);
impl_solve!(f32, lapacke::sgetrf);
impl_solve!(c64, lapacke::zgetrf);
impl_solve!(c32, lapacke::cgetrf);

#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("memory not contiguous")]
    NotContiguous,
    #[error("too many columns: {0}")]
    TooManyColumns(TryFromIntError),
    #[error("too many rows: {0}")]
    TooManyRows(TryFromIntError),
}

pub(crate) fn lax_layout<A, S>(a: &ArrayBase<S, Ix2>) -> Result<MatrixLayout, LayoutError>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let nrows = i32::try_from(a.nrows()).map_err(LayoutError::TooManyRows)?;
    let ncols = i32::try_from(a.ncols()).map_err(LayoutError::TooManyColumns)?;
    if nrows as isize == a.stride_of(Axis(1)) {
        Ok(MatrixLayout::F {
            col: ncols,
            lda: nrows,
        })
    } else if ncols as isize == a.stride_of(Axis(0)) {
        Ok(MatrixLayout::C {
            row: nrows,
            lda: ncols,
        })
    } else {
        Err(LayoutError::NotContiguous)
    }
}

pub(crate) type SvdOutput<A> = (Array2<A>, Array1<<A as Scalar>::Real>, Array2<A>);

/// Calls gesdd.
///
/// # Panics
///
/// Panics if `a`'s memory layout is not contiguous.
pub(crate) fn svddc<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<SvdOutput<A>, LaxError>
where
    A: SVDDC_,
    S: DataMut<Elem = A>,
{
    let l = lax_layout(a).unwrap();
    let nrows = i32::try_from(a.nrows()).expect("doesn't exceed i32::MAX");
    let ncols = i32::try_from(a.ncols()).expect("doesn't exceed i32::MAX");
    let k = cmp::min(nrows, ncols);
    let output = A::svddc(
        l,
        UVTFlag::Some,
        a.as_slice_memory_order_mut().expect("contiguous"),
    )?;
    let u =
        vec_into_array(l.resized(nrows, k), output.u.expect("`u` requested")).expect("valid shape");
    let sigma = ArrayBase::from(output.s);
    let vt = vec_into_array(l.resized(k, ncols), output.vt.expect("vt` requested"))
        .expect("valid shape");
    Ok((u, sigma, vt))
}

pub fn vec_into_array<A>(l: MatrixLayout, a: Vec<A>) -> Result<Array2<A>, ShapeError> {
    match l {
        MatrixLayout::C { row, lda } => Ok(ArrayBase::from_shape_vec(
            (
                usize::try_from(row).expect("positive"),
                usize::try_from(lda).expect("positive"),
            ),
            a,
        )?),
        MatrixLayout::F { col, lda } => Ok(ArrayBase::from_shape_vec(
            (
                usize::try_from(lda).expect("positive"),
                usize::try_from(col).expect("positive"),
            )
                .f(),
            a,
        )?),
    }
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
