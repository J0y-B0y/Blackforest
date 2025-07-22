# BlackForest

> **A production-grade, parallelized Random Forest implementation in pure Rust.**

BlackForest is a modular, high-performance machine learning library implementing Random Forests and Decision Trees from first principles. Designed for systems-level experimentation, education, and custom ML infrastructure, it delivers accuracy, parallelism, and portability — without external ML dependencies.

---

## Features

- Pure Rust implementation — no Python bindings or wrappers
- Recursive, depth-limited Decision Tree learning
- Random Forest ensemble with bootstrap sampling
- Multithreaded training via [Rayon](https://docs.rs/rayon)
- Model serialization with [Bincode](https://docs.rs/bincode)
- Benchmark-ready using [Criterion](https://docs.rs/criterion)
- Custom CSV dataset loader with type-safe parsing
- Predict, evaluate, save, and reload with ease

---

## Installation

Clone this repository or add it to your workspace manually:

```bash
git clone https://github.com/J0y-B0y/Blackforest.git
cd Blackforest
```

---

## Project Structure

```plaintext
src/
├── lib.rs            # Library entry point
├── bin/cli.rs        # CLI demo: load, train, evaluate, save
├── tree/             # Decision tree internals
│   ├── mod.rs
│   ├── split.rs
│   └── metrics.rs
├── forest/           # Random forest module
│   ├── mod.rs
│   └── data.rs
└── utils/            # Dataset loading & record struct
    ├── mod.rs
    ├── record.rs
    └── dataset.rs

benches/              # Criterion benchmarking suite
datasets/             # Sample dataset: iris.csv
```

---

## Example: Run the CLI

```bash
cargo run --bin cli
```

This command:

- Loads the Iris dataset
- Trains a decision tree and a random forest
- Prints accuracy and predictions
- Saves the model to `model.bin`
- Reloads the model and verifies consistency

---

## Algorithms Implemented

### Decision Tree

- Binary splits using Gini impurity
- Recursive tree building with max depth and min samples
- Majority voting at leaf nodes

### Random Forest

- Parallel training using bootstrapped samples
- Tree bagging with feature randomness
- Final prediction via majority vote aggregation
- Persistent model serialization

---

## Benchmarks

Run benchmarks using [Criterion](https://github.com/bheisler/criterion.rs):

```bash
cargo bench
```

Benchmarks measure:

- `DecisionTree::fit`
- `RandomForest::fit`
- Execution time over Iris dataset

---

## License

Licensed under the [MIT License](LICENSE).  
© 2025 Divjot Singh.

---

## Author

**Divjot Singh**  
Co-Founder @ [Waefa](https://linkedin.com/in/company/waefa)  
[LinkedIn](https://www.linkedin.com/in/divjotsingh) · [GitHub](https://github.com/J0y-B0y)

---

## Contributions & Acknowledgments

This project is maintained by Divjot Singh for educational, research, and production-use cases.  
Contributions are welcome via PRs or issues.

---

## Roadmap

- [ ] Add support for entropy-based splits (Information Gain)
- [ ] Implement OOB (out-of-bag) error estimation
- [ ] Extend CLI with configurable arguments
- [ ] Crates.io release
- [ ] Add unit tests & CI pipeline

---

## Inspired by

- [scikit-learn](https://scikit-learn.org/)
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)

---
