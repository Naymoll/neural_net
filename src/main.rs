mod neural_net;

use crate::neural_net::NeuralNet;
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::str::FromStr;

const CSV_DATA: usize = 785;

fn train(data: Vec<f64>, path_to_neural: &OsString, count: usize) {
    let input_nodes = 784;
    let hidden_nodes = 800;
    let output_nodes = 10;
    let training_rate = 0.1;

    let mut net = NeuralNet::new(input_nodes, hidden_nodes, output_nodes, training_rate);

    let epochs: u8 = 5;
    for _epoch in 0..epochs {
        println!("Epoch: {}", _epoch + 1);

        for number in 0..count {
            let index = number * CSV_DATA;
            let inputs: Vec<f64> = data[index + 1..index + 785].to_vec();

            let mut targets = vec![0.1; output_nodes];
            let answer = data[index];
            targets[answer as usize] = 0.99;

            net.train(inputs, targets);
        }
    }

    let serialized = serde_json::to_string(&net).unwrap();
    let mut file = File::create(path_to_neural).unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}

fn test(data: Vec<f64>, path_to_neural: &OsString, count: usize) {
    let mut json_data = String::new();

    let mut file = File::open(path_to_neural).unwrap();
    file.read_to_string(&mut json_data).unwrap();

    let mut net: NeuralNet = serde_json::from_str(&json_data).unwrap();

    let mut ok: usize = 0;
    for number in 0..count {
        let answer_index = number * CSV_DATA;
        let inputs: Vec<f64> = data[answer_index + 1..answer_index + 785].to_vec();

        let result = net.query(inputs);

        let answer = data[answer_index];
        let max = result
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();

        if answer as usize == max.0 {
            ok += 1;
        }
    }

    println!("{}", ok as f64 / count as f64 * 100.0);
}

fn main() {
    let args: Vec<OsString> = std::env::args_os().collect();

    let is_train = args[1] == "-t";
    let path_to_csv = &args[2];
    let path_to_width = &args[3];

    let mut csv = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path_to_csv)
        .unwrap();

    let count = match is_train {
        true => 60000,
        false => 10000,
    };

    let mut data = Vec::with_capacity(count * CSV_DATA);
    for record in csv.records() {
        for s in record.unwrap().iter() {
            let value = f64::from_str(s).unwrap();

            data.push(value / 255.0 * 0.99 + 0.01);
        }
    }

    match is_train {
        true => train(data, &path_to_width, count),
        false => test(data, &path_to_width, count),
    }
}
