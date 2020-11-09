use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NeuralNet {
    input_nodes: usize,
    hidden_nodes: usize,
    output_nodes: usize,
    training_rate: f64,
    wih: ndarray::Array2<f64>,
    who: ndarray::Array2<f64>,
}

impl NeuralNet {
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        training_rate: f64,
    ) -> NeuralNet {
        let rng = rand::thread_rng();

        let normal = Normal::new(0.0, (hidden_nodes as f64).powf(-0.5)).unwrap();
        let wih = ndarray::Array2::from_shape_vec(
            (hidden_nodes, input_nodes),
            normal
                .sample_iter(rng)
                .take(hidden_nodes * input_nodes)
                .collect(),
        )
        .unwrap();

        let normal = Normal::new(0.0, (output_nodes as f64).powf(-0.5)).unwrap();
        let who = ndarray::Array2::from_shape_vec(
            (output_nodes, hidden_nodes),
            normal
                .sample_iter(rng)
                .take(output_nodes * hidden_nodes)
                .collect(),
        )
        .unwrap();

        NeuralNet {
            input_nodes,
            hidden_nodes,
            output_nodes,
            training_rate,
            wih,
            who,
        }
    }

    pub fn query(&mut self, values: Vec<f64>) -> Vec<f64> {
        let input = ndarray::Array2::from_shape_vec((values.len(), 1), values).unwrap();

        let closure: fn(f64) -> f64 = |x| 1.0 / (1.0 + f64::exp(-x));
        let hidden_inputs = ndarray::Array2::dot(&self.wih, &input);
        let hidden_outputs = hidden_inputs.map(|x| closure(*x));

        let final_inputs = ndarray::Array2::dot(&self.who, &hidden_outputs);
        let final_outputs = final_inputs.map(|x| closure(*x));

        final_outputs.into_raw_vec()
    }

    pub fn train(&mut self, input_values: Vec<f64>, target_values: Vec<f64>) {
        let inputs =
            ndarray::Array2::from_shape_vec((input_values.len(), 1), input_values).unwrap();
        let targets =
            ndarray::Array2::from_shape_vec((target_values.len(), 1), target_values).unwrap();

        let closure: fn(f64) -> f64 = |x| 1.0 / (1.0 + f64::exp(-x));
        let hidden_inputs = self.wih.dot(&inputs);
        let hidden_outputs = hidden_inputs.map(|x| closure(*x));

        let final_inputs = self.who.dot(&hidden_outputs);
        let final_outputs = final_inputs.map(|x| closure(*x));

        let outputs_errors = &targets - &final_outputs;
        let hidden_errors = self.who.clone().reversed_axes().dot(&outputs_errors);

        let closure: fn(f64) -> f64 = |x| 1.0 - x;
        let buff = &outputs_errors * &final_outputs * &final_outputs.map(|x| closure(*x));
        self.who += &(&buff.dot(&hidden_outputs.clone().reversed_axes()) * self.training_rate);

        let buff = &hidden_errors * &hidden_outputs * &hidden_outputs.map(|x| closure(*x));
        self.wih += &(&buff.dot(&inputs.reversed_axes()) * self.training_rate);
    }
}
