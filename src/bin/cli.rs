use blackforest::utils::dataset::load_dataset;
use blackforest::tree::metrics::gini_impurity;
use blackforest::tree::DecisionTree;
use blackforest::forest::data::bootstrap_sample;
use blackforest::forest::RandomForest;

fn main() {
    let path = "datasets/iris.csv";

    match load_dataset(path) {
        Ok(records) => {
            println!("✅ Loaded {} records.\n", records.len());

            for (i, rec) in records.iter().enumerate() {
                println!("Record {}: features = {:?}, label = {}", i + 1, rec.features, rec.label);
            }

            let gini = gini_impurity(&records);
            println!("\n📊 Gini Impurity of full dataset: {:.4}", gini);

            // Train/test split (80/20)
            let split_index = (records.len() as f64 * 0.8).round() as usize;
            let (train_data, test_data) = records.split_at(split_index);

            // Train Decision Tree
            let mut tree = DecisionTree::new(3, 2);
            tree.fit(train_data);
            println!("\n🌳 Decision Tree trained.\n");

            for (i, rec) in test_data.iter().enumerate() {
                let predicted = tree.predict(rec).unwrap_or(999);
                println!(
                    "🔍 DT Test Record {}: actual = {}, predicted = {} {}",
                    i + 1,
                    rec.label,
                    predicted,
                    if predicted == rec.label { "✅" } else { "❌" }
                );
            }

            // Bootstrap sample test
            let sample = bootstrap_sample(train_data);
            println!("\n📦 Bootstrapped Sample:");
            for (i, rec) in sample.iter().take(5).enumerate() {
                println!("Sample {}: label = {}", i + 1, rec.label);
            }

            // Train Random Forest
            let mut forest = RandomForest::new(5, 3, 2);
            forest.fit(train_data);
            println!("\n🌲 Random Forest trained on {} records.\n", train_data.len());

            // Evaluate Random Forest
            let mut correct = 0;
            for (i, rec) in test_data.iter().enumerate() {
                let prediction = forest.predict(rec).unwrap_or(999);
                if prediction == rec.label {
                    correct += 1;
                }
                println!(
                    "🧠 RF Prediction {}: actual = {}, predicted = {} {}",
                    i + 1,
                    rec.label,
                    prediction,
                    if prediction == rec.label { "✅" } else { "❌" }
                );
            }

            let accuracy = correct as f64 / test_data.len() as f64 * 100.0;
            println!("\n📈 Random Forest Accuracy on Test Set: {:.2}%", accuracy);

            // Save and Reload
            let save_path = "model.bin";
            forest.save_to_file(save_path).expect("❌ Failed to save model");
            println!("💾 Forest saved to '{}'", save_path);

            let loaded_forest =
                RandomForest::load_from_file(save_path).expect("❌ Failed to load model");
            println!("📂 Forest reloaded from disk.");

            for (i, rec) in test_data.iter().enumerate() {
                let prediction = loaded_forest.predict(rec).unwrap_or(999);
                println!(
                    "🔁 Reloaded RF Prediction {}: actual = {}, predicted = {} {}",
                    i + 1,
                    rec.label,
                    prediction,
                    if prediction == rec.label { "✅" } else { "❌" }
                );
            }
        }

        Err(e) => {
            eprintln!("❌ Failed to load dataset: {}", e);
        }
    }
}
