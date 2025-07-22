use blackforest::tree::DecisionTree;
use blackforest::forest::RandomForest;
use blackforest::utils::dataset::load_dataset;
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_decision_tree(c: &mut Criterion) {
    let records = load_dataset("datasets/iris.csv").expect("Failed to load");

    c.bench_function("decision_tree_fit", |b| {
        b.iter(|| {
            let mut tree = DecisionTree::new(4, 2);
            tree.fit(&records);
        });
    });
}

fn bench_random_forest(c: &mut Criterion) {
    let records = load_dataset("datasets/iris.csv").expect("Failed to load");

    c.bench_function("random_forest_fit", |b| {
        b.iter(|| {
            let mut forest = RandomForest::new(10, 4, 2);
            forest.fit(&records);
        });
    });
}

criterion_group!(benches, bench_decision_tree, bench_random_forest);
criterion_main!(benches);

