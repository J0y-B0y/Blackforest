// module declarations first
pub mod metrics;
pub mod split;

// use statements next
use crate::tree::split::find_best_split;
use crate::utils::record::Record;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    pub feature_index: usize,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Leaf { class: usize },
    Decision {
        split: Split,
        left: Box<Node>,
        right: Box<Node>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionTree {
    pub root: Option<Node>,
    pub max_depth: usize,
    pub min_samples_split: usize,
}

// decision tree methods
impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
        }
    }

    pub fn fit(&mut self, data: &[Record]) {
        self.root = Some(Self::build_tree(
            data,
            self.max_depth,
            self.min_samples_split,
        ));
    }
    pub fn predict(&self, record: &Record) -> Option<usize> {
        match &self.root {
            Some(node) => Some(Self::traverse(node, record)),
            None => None,
        }
    }

    fn traverse(node: &Node, record: &Record) -> usize {
        match node {
            Node::Leaf { class } => *class,
            Node::Decision { split, left, right } => {
                if record.features[split.feature_index] <= split.threshold {
                    Self::traverse(left, record)
                } else {
                    Self::traverse(right, record)
                }
            }
        }
    }

    fn build_tree(data: &[Record], depth: usize, min_samples: usize) -> Node {
        if is_pure(data) || depth == 0 || data.len() < min_samples {
            let class = majority_class(data);
            return Node::Leaf { class };
        }

        match find_best_split(data) {
            Some((split, _)) => {
                let (left_data, right_data): (Vec<_>, Vec<_>) = data
                    .iter()
                    .cloned()
                    .partition(|r| r.features[split.feature_index] <= split.threshold);

                let left_node = Self::build_tree(&left_data, depth - 1, min_samples);
                let right_node = Self::build_tree(&right_data, depth - 1, min_samples);

                Node::Decision {
                    split,
                    left: Box::new(left_node),
                    right: Box::new(right_node),
                }
            }
            None => {
                let class = majority_class(data);
                Node::Leaf { class }
            }
        }
    }
}

// helpers
fn is_pure(data: &[Record]) -> bool {
    if data.is_empty() {
        return true;
    }
    let first = data[0].label;
    data.iter().all(|r| r.label == first)
}

fn majority_class(data: &[Record]) -> usize {
    use std::collections::HashMap;

    let mut counts = HashMap::new();
    for record in data {
        *counts.entry(record.label).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label)
        .unwrap_or(0)
}
