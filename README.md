# petal-decomposition

petal-decomposition provides matrix decomposition algorithms including PCA
(principal component analysis) and ICA (independent component analysis).

[![crates.io](https://img.shields.io/crates/v/petal-decomposition)](https://crates.io/crates/petal-decomposition)
[![Documentation](https://docs.rs/petal-decomposition/badge.svg)](https://docs.rs/petal-decomposition)
[![Coverage Status](https://codecov.io/gh/petabi/petal-decomposition/branch/master/graphs/badge.svg)](https://codecov.io/gh/petabi/petal-decomposition)

## Requirements

* Rust ≥ 1.46
* BLAS/LAPACK backend (OpenBLAS, Netlib, or Intel MKL)

## Features

* PCA with exact, full SVD (singular value decomposition)
* PCA with randomized, truncated SVD
* [FastICA](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf)

## Crate Features

* Use one of `intel-mkl-static`, `intel-mkl-system`, `netlib-static`, `netlib-system`,
  `openblas-static`, and `openblas-system` to select a BLAS/LAPACK
  backend.
  See [ndarray-linalg's documentation][ndarray-linalg-features] for details.
* `serialization` enables serialization/deserialization using [serde](https://crates.io/crates/serde).

## Examples

The following example shows how to apply PCA to an array of three samples, and
obtain singular values as well as how much variance each component explains.

```rust
use ndarray::arr2;
use petal_decomposition::Pca;

let x = arr2(&[[0_f64, 0_f64], [1_f64, 1_f64], [2_f64, 2_f64]]);
let mut pca = Pca::new(2);               // Keep two dimensions.
pca.fit(&x).unwrap();

let s = pca.singular_values();           // [2_f64, 0_f64]
let v = pca.explained_variance_ratio();  // [1_f64, 0_f64]
let y = pca.transform(&x).unwrap();      // [-2_f64.sqrt(), 0_f64, 2_f64.sqrt()]
```

## License

Copyright 2020-2021 Petabi, Inc.

Licensed under [Apache License, Version 2.0][apache-license] (the "License");
you may not use this crate except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See [LICENSE](LICENSE) for
the specific language governing permissions and limitations under the License.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0
license][apache-license], shall be licensed as above, without any additional
terms or conditions.

[apache-license]: http://www.apache.org/licenses/LICENSE-2.0
[ndarray-linalg-features]: https://github.com/rust-ndarray/ndarray-linalg#backend-features
