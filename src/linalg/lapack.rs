use cauchy::Scalar;
use num_complex::{Complex32, Complex64};
use num_traits::{ToPrimitive, Zero};

pub trait Lapack: Scalar {
    unsafe fn heev(
        jobz: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        ld_a: i32,
    ) -> Result<Vec<Self::Real>, i32>;
}

macro_rules! impl_lapack {
    (@real, $scalar:ty, $heev:path) => {
        impl_lapack!(@body, $scalar, $heev, );
    };
    (@complex, $scalar:ty, $heev:path) => {
        impl_lapack!(@body, $scalar, $heev, rwork);
    };
    (@body, $scalar:ty, $heev:path, $($rwork_ident:ident),*) => {
        impl Lapack for $scalar {
            unsafe fn heev(
                jobz: u8,
                uplo: u8,
                n: i32,
                mut a: &mut [Self],
                ld_a: i32,
            ) -> Result<Vec<Self::Real>, i32> {
                let mut eigs = vec_uninit(n as usize);

                $(
                let mut $rwork_ident = vec_uninit(3 * n as usize - 2 as usize);
                )*

                let mut info = 0;
                let mut work_size = [Self::zero()];
                    $heev(
                        jobz,
                        uplo,
                        n,
                        &mut a,
                        ld_a,
                        &mut eigs,
                        &mut work_size,
                        -1,
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                if info != 0 {
                    return Err(info);
                }

                let lwork = work_size[0].to_usize().expect("valid integer");
                let mut work = vec_uninit(lwork);
                    $heev(
                        jobz,
                        uplo,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut work,
                        lwork as i32,
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                if info != 0 {
                    return Err(info);
                }

                Ok(eigs)
            }
        }
    };
}

impl_lapack!(@real, f32, lapack::ssyev);
impl_lapack!(@real, f64, lapack::dsyev);
impl_lapack!(@complex, Complex32, lapack::cheev);
impl_lapack!(@complex, Complex64, lapack::zheev);

unsafe fn vec_uninit<T: Sized>(n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    v.set_len(n);
    v
}
