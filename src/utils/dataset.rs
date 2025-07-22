use crate::utils::record::Record;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::io::BufReader;

/// Load a dataset where each row ends with a label column.
/// All columns before the label must be numeric features.
pub fn load_dataset<P: AsRef<Path>>(path: P) -> Result<Vec<Record>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(BufReader::new(file));

    let mut dataset = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let mut row = record.iter().map(|s| s.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>()?;

        if row.is_empty() {
            continue;
        }

        let label = row.pop().ok_or("Missing label column")? as usize;
        dataset.push(Record {
            features: row,
            label,
        });
    }

    Ok(dataset)
}
