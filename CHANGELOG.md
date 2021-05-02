# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Requires Rust 1.46 or later.
- Updated ndarray to 0.14.

## [0.4.1] - 2020-07-17

### Added

- `Pca::components()` and `RandomizedPca::components()`.
- `Pca::mean()` and `RandomizedPca::mean()`.
- `Pca::inverse_transform()` and `RandomizedPca::inverse_transform()`.

## [0.4.0] - 2020-06-18

### Added

- Serialization/deserialization using serde.
- `FastIca::with_seed` and `RandomizedPca::with_seed` to easily create a model
  with a reproducible behavior.
- `FastIca::with_rng` and `RandomizedPca::with_rng` to replace `new` in earlier
  versions.

### Changed

- `FastIca::new` and `RandomizedPca::new` no longer requires a random number
  generator; they uses a PCG random number genrator by default. Use `with_rng`
  instead for a different random number generator.

## [0.3.0] - 2020-06-05

### Added

- `FastIca`, an independent component analysis (ICA) algorithm.

## [0.2.0] - 2020-05-24

### Added

- `RandomizedPca`, a randomized, truncated PCA.

### Fixed

- `Pca::explained_variance_ratio` returns the ratios for the principal
  components only.

## [0.1.1] - 2020-05-21

### Added

- cargo features to select the LAPACK backend: intel-mkl, netlib, or openblas.

## [0.1.0] - 2020-05-20

### Added

- Principal component analysis (PCA).

[Unreleased]: https://github.com/petabi/petal-decomposition/compare/0.4.1...master
[0.4.1]: https://github.com/petabi/petal-decomposition/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/petabi/petal-decomposition/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/petabi/petal-decomposition/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/petabi/petal-decomposition/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/petabi/petal-decomposition/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/petabi/petal-decomposition/tree/0.1.0
