use crate::tree::metrics::gini_impurity;
use crate::tree::Split;
use crate::utils::record::Record;

/// Finds the best split point across all features in the dataset.
pub fn find_best_split(data: &[Record]) -> Option<(Split, f64)> {
    if data.is_empty() {
        return None;
    }

    let num_features = data[0].features.len();
    let mut best_split: Option<(Split, f64)> = None;

    for feature_idx in 0..num_features {
        // Collect all values for this feature
        let mut values: Vec<f64> = data.iter().map(|r| r.features[feature_idx]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Try midpoints between sorted values
        for i in 1..values.len() {
            let threshold = (values[i - 1] + values[i]) / 2.0;

            let (left, right): (Vec<_>, Vec<_>) = data
                .iter()
                .cloned()
                .partition(|r| r.features[feature_idx] <= threshold);

            if left.is_empty() || right.is_empty() {
                continue;
            }

            let gini_left = gini_impurity(&left);
            let gini_right = gini_impurity(&right);

            let weighted_gini = (left.len() as f64 * gini_left + right.len() as f64 * gini_right)
                / data.len() as f64;

            if best_split.is_none() || weighted_gini < best_split.as_ref().unwrap().1 {
                best_split = Some((
                    Split {
                        feature_index: feature_idx,
                        threshold,
                    },
                    weighted_gini,
                ));
            }
        }
    }

    best_split
}
