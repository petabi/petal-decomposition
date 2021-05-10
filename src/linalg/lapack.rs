use lair::Scalar;
use num_complex::{Complex32, Complex64};
use num_traits::{ToPrimitive, Zero};
use std::cmp;

pub(super) type SvdOutput<A> = (Vec<<A as Scalar>::Real>, Vec<A>, Option<Vec<A>>);

pub trait Lapack: Scalar {
    unsafe fn gelqf(m: i32, n: i32, a: &mut [Self], ld_a: i32) -> Result<Vec<Self>, i32>;

    unsafe fn gesdd(m: i32, n: i32, a: &mut [Self]) -> Result<SvdOutput<Self>, i32>;

    unsafe fn gesvd(jobvt: u8, m: i32, n: i32, a: &mut [Self]) -> Result<SvdOutput<Self>, i32>;

    unsafe fn heev(
        jobz: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        ld_a: i32,
    ) -> Result<Vec<Self::Real>, i32>;

    unsafe fn unglq(
        m: i32,
        n: i32,
        k: i32,
        a: &mut [Self],
        ld_a: i32,
        tau: &[Self],
    ) -> Result<(), i32>;
}

macro_rules! impl_lapack {
    (@real, $scalar:ty, $heev:path, $gelqf:path, $gesdd:path, $gesvd:path, $unglq:path) => {
        impl_lapack!(@body, $scalar, $heev, $gelqf, $gesdd, $gesvd, $unglq, );
    };
    (@complex, $scalar:ty, $heev:path, $gelqf:path, $gesdd:path, $gesvd:path, $unglq:path) => {
        impl_lapack!(@body, $scalar, $heev, $gelqf, $gesdd, $gesvd, $unglq, rwork);
    };
    (@body, $scalar:ty, $heev:path, $gelqf:path, $gesdd:path, $gesvd:path, $unglq:path, $($rwork_ident:ident),*) => {
        impl Lapack for $scalar {
            unsafe fn gelqf(m: i32, n: i32, mut a: &mut [Self], ld_a: i32) -> Result<Vec<Self>, i32> {
                let k = cmp::min(m, n);
                let mut tau = vec_uninit(k as usize);

                let mut info = 0;
                let mut work_size = [Self::zero()];
                $gelqf(m, n, &mut a, ld_a, &mut tau, &mut work_size, -1, &mut info);
                if info != 0 {
                    return Err(info);
                }

                let lwork = work_size[0].to_usize().expect("valid integer");
                let mut work = vec_uninit(lwork);
                $gelqf(m, n, &mut a, ld_a, &mut tau, &mut work, lwork as i32, &mut info);
                if info != 0 {
                    return Err(info);
                }

                Ok(tau)
            }

            unsafe fn gesdd(m: i32, n: i32, mut a: &mut [Self]) -> Result<SvdOutput<Self>, i32> {
                let k = cmp::min(m, n);
                let mut s = vec_uninit(k as usize);
                let mut u = vec_uninit((n * k) as usize);
                let mut vt = vec_uninit((k * m) as usize);

                $(
                let mut $rwork_ident = {
                    let max_dim = cmp::max(m, n) as usize;
                    let min_dim = k as usize;
                    let lwork = cmp::max(5 * min_dim * min_dim + 5 * max_dim + min_dim, 2 * max_dim * min_dim + 2 * min_dim * min_dim + min_dim);
                    vec_uninit(lwork as usize)
                };
                )*

                let mut info = 0;
                let mut iwork = vec_uninit(8 * k as usize);
                let mut work_size = [Self::zero()];
                $gesdd(b'S', n, m, &mut a, n, &mut s, &mut u, n, &mut vt, k, &mut work_size, -1, $(&mut $rwork_ident,)* &mut iwork, &mut info);
                if info != 0 {
                    return Err(info);
                }

                let lwork = work_size[0].to_usize().expect("valid integer");
                let mut work = vec_uninit(lwork);
                $gesdd(b'S', n, m, &mut a, n, &mut s, &mut u, n, &mut vt, k, &mut work, lwork as i32, $(&mut $rwork_ident,)* &mut iwork, &mut info);
                if info != 0 {
                    return Err(info);
                }

                Ok((s, vt, Some(u)))
            }

            unsafe fn gesvd(jobvt: u8, m: i32, n: i32, mut a: &mut [Self]) -> Result<SvdOutput<Self>, i32> {
                let k = cmp::min(m, n);
                let mut s = vec_uninit(k as usize);
                let mut u = if jobvt == b'A' {
                    vec_uninit((n * n) as usize)
                } else {
                    Vec::new()
                };
                let mut vt = vec_uninit((m * m) as usize);

                $(
                let mut $rwork_ident = vec_uninit(5 * k as usize);
                )*

                let mut info = 0;
                let mut work_size = [Self::zero()];
                $gesvd(jobvt, b'A', n, m, &mut a, n, &mut s, &mut u, n, &mut vt, m, &mut work_size, -1, $(&mut $rwork_ident,)* &mut info);
                if info != 0 {
                    return Err(info);
                }

                let lwork = work_size[0].to_usize().expect("valid integer");
                let mut work = vec_uninit(lwork);
                $gesvd(jobvt, b'A', n, m, &mut a, n, &mut s, &mut u, n, &mut vt, m, &mut work, lwork as i32, $(&mut $rwork_ident,)* &mut info);
                if info != 0 {
                    return Err(info);
                }

                Ok((s, vt, if jobvt == b'A' { Some(u) } else { None }))
            }

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

            unsafe fn unglq(m: i32, n: i32, k: i32, mut a: &mut [Self], ld_a: i32, tau: &[Self]) -> Result<(), i32> {
                let mut info = 0;
                let mut work_size = [Self::zero()];
                $unglq(m, n, k, &mut a, ld_a, &tau, &mut work_size, -1, &mut info);
                if info != 0 {
                    return Err(info);
                }

                let lwork = work_size[0].to_usize().expect("valid integer");
                let mut work = vec_uninit(lwork);
                $unglq(m, n, k, &mut a, ld_a, &tau, &mut work, lwork as i32, &mut info);
                if info != 0 {
                    return Err(info);
                }

                Ok(())
            }
        }
    };
}

impl_lapack!(@real, f32, lapack::ssyev, lapack::sgelqf, lapack::sgesdd, lapack::sgesvd, lapack::sorglq);
impl_lapack!(@real, f64, lapack::dsyev, lapack::dgelqf, lapack::dgesdd, lapack::dgesvd, lapack::dorglq);
impl_lapack!(@complex, Complex32, lapack::cheev, lapack::cgelqf, lapack::cgesdd, lapack::cgesvd, lapack::cunglq);
impl_lapack!(@complex, Complex64, lapack::zheev, lapack::zgelqf, lapack::zgesdd, lapack::zgesvd, lapack::zunglq);

unsafe fn vec_uninit<T: Sized>(n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    v.set_len(n);
    v
}
