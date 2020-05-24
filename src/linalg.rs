use ndarray::{s, ArrayBase, DataMut, Ix2};
use ndarray_linalg::{c32, c64, error::LinalgError, AllocatedArray, MatrixLayout, Pivot, Scalar};
use std::cmp;

/// Computes P * L after LU decomposition.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub(crate) fn lu_pl<A, S>(m: &mut ArrayBase<S, Ix2>) -> Result<(), LinalgError>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    let mut pivots = unsafe { A::lupiv(m.layout().unwrap(), m.as_slice_mut().unwrap()) }?;
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
    unsafe fn lupiv(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot, LinalgError>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path) => {
        impl Lapack for $scalar {
            unsafe fn lupiv(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot, LinalgError> {
                let (row, col) = l.size();
                let k = ::std::cmp::min(row, col);
                let mut ipiv = vec![0; k as usize];
                let info = $getrf(l.lapacke_layout(), row, col, a, l.lda(), &mut ipiv);
                if info >= 0 {
                    Ok(ipiv)
                } else {
                    Err(LinalgError::Lapack { return_code: info })
                }
            }
        }
    };
}

impl_solve!(f64, lapacke::dgetrf);
impl_solve!(f32, lapacke::sgetrf);
impl_solve!(c64, lapacke::zgetrf);
impl_solve!(c32, lapacke::cgetrf);

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
