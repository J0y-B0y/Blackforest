pub mod data;
use rayon::prelude::*;
use crate::tree::DecisionTree;
use crate::utils::record::Record;
use self::data::bootstrap_sample;
use std::fs::File;
use serde::{Serialize, Deserialize};
use std::io::{BufReader, BufWriter};

#[derive(Debug, Serialize, Deserialize)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
}

impl RandomForest {
    pub fn new(n_trees: usize, max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth,
            min_samples_split,
        }
    }
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<bincode::ErrorKind>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self, Box<bincode::ErrorKind>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let forest = bincode::deserialize_from(reader)?;
        Ok(forest)
    }

    pub fn fit(&mut self, data: &[Record]) {
    self.trees = (0..self.n_trees)
        .into_par_iter()
        .map(|_| {
            let sample = bootstrap_sample(data);
            let mut tree = DecisionTree::new(self.max_depth, self.min_samples_split);
            tree.fit(&sample);
            tree
        })
        .collect();
}

    pub fn predict(&self, record: &Record) -> Option<usize> {
    use std::collections::HashMap;

    let mut votes = HashMap::new();

    for tree in &self.trees {
        if let Some(label) = tree.predict(record) {
            *votes.entry(label).or_insert(0) += 1;
        }
    }

    votes
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label)
}

}