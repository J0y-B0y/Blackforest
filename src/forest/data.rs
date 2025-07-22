use rand::prelude::*;
use crate::utils::record::Record;

/// Returns a new Vec<Record> sampled *with replacement* from the original data.
pub fn bootstrap_sample(data: &[Record]) -> Vec<Record> {
    let mut rng = thread_rng();
    let n = data.len();
    let mut sample = Vec::with_capacity(n);

    for _ in 0..n {
        let idx = rng.gen_range(0..n);
        sample.push(data[idx].clone());
    }

    sample
}
