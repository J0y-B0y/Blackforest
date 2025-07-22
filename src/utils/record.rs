use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Record {
    pub features: Vec<f64>,
    pub label: usize,
}
