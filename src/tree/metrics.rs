use std::collections::HashMap;
use crate::utils::record::Record;

/// Computes the Gini impurity of a dataset slice.
pub fn gini_impurity(data: &[Record]) -> f64 {
    let mut label_counts = HashMap::new();
    let total = data.len() as f64;

    for record in data {
        *label_counts.entry(record.label).or_insert(0) += 1;
    }

    let impurity = label_counts
        .values()
        .map(|&count| {
            let p = count as f64 / total;
            p * p
        })
        .sum::<f64>();

    1.0 - impurity
}
