use ndarray::{Array, Array1, Array2};

use crate::nn_functions::{ActivationFunction, CostFunction};

struct Layer {
    size: usize,
    weights_in: Array2<f64>,
    inputs: Array1<f64>,
    outputs: Array1<f64>,
    errors: Vec<f64>,
    activation_function: ActivationFunction,
}

struct Row {
    inputs: Vec<f64>,
    outputs: Vec<f64>,
}

type Dataset = Vec<Row>;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    training_dataset: Dataset,
    testing_dataset: Dataset,
}

fn get_input_layer(layers: &Vec<Layer>) -> &Layer {
    &layers[0]
}

fn get_output_layer(layers: &Vec<Layer>) -> &Layer {
    layers.last().unwrap()
}

fn get_input_layer_mut(layers: &mut Vec<Layer>) -> &mut Layer {
    &mut layers[0]
}

fn get_output_layer_mut(layers: &mut Vec<Layer>) -> &mut Layer {
    layers.last_mut().unwrap()
}

fn forward_propagation(features: &Vec<f64>, layers: &mut Vec<Layer>) {
    let input_layer = get_input_layer_mut(layers);
    input_layer.inputs = Array::from_vec(features.clone());
    input_layer.outputs = Array::from_vec(features.clone());
    for l in 1..layers.len() {
        layers[l].inputs = layers[l - 1].outputs.dot(&layers[l].weights_in);
        layers[l].outputs = layers[l]
            .inputs
            .map(|inp| (layers[l].activation_function.apply)(*inp));
    }
}

fn backward_propagation(
    learning_rate: f64,
    cost_function: &CostFunction,
    targets: &Vec<f64>,
    layers: &mut Vec<Layer>,
) {
    // Errors for output layer neurons
    let output_layer = get_output_layer_mut(layers);
    for n in 0..output_layer.outputs.len() {
        let output = output_layer.outputs[n];
        output_layer.errors[n] = (cost_function.apply_derivative)(output, targets[n])
            * (output_layer.activation_function.apply_derivative)(output_layer.inputs[n], output);
    }
    // Errors for hidden layer neurons
    for l in (1..=(layers.len() - 2)).rev() {
        for n in 0..layers[l].size {
            let mut weighted_errors_sum = 0_f64;
            for neighbor_index in 0..layers[l + 1].size {
                weighted_errors_sum += layers[l + 1].weights_in[[n, neighbor_index]]
                    * layers[l + 1].errors[neighbor_index];
            }
            layers[l].errors[n] = weighted_errors_sum
                * (layers[l].activation_function.apply_derivative)(
                    layers[l].inputs[n],
                    layers[l].outputs[n],
                )
        }
    }
    // Weight updates
    for l in 1..layers.len() {
        for prev_node_index in 0..(&layers[l]).weights_in.len() {
            for current_node_index in 0..(&layers[l]).weights_in.row(prev_node_index).len() {
                (&mut layers[l]).weights_in[[prev_node_index, current_node_index]] += learning_rate
                    * (&layers[l]).errors[current_node_index]
                    * layers[l - 1].outputs[prev_node_index];
            }
        }
    }
}

fn get_error(
    testing_dataset: &Dataset,
    cost_function: &CostFunction,
    layers: &mut Vec<Layer>,
) -> f64 {
    let mut error = 0_f64;
    for row in testing_dataset {
        forward_propagation(&row.inputs, layers);
        let output_layer = get_output_layer_mut(layers);
        for i in 0..output_layer.size {
            error += (cost_function.apply)(output_layer.outputs[i], row.outputs[i]);
        }
    }
    error
}

fn add_row(inputs: Vec<f64>, outputs: Vec<f64>, layers: &Vec<Layer>, dataset: &mut Dataset) {
    if layers.len() < 2 {
        panic!("At least an input and output layer must exist before adding a dataset row");
    }
    if inputs.len() != get_input_layer(layers).size {
        panic!("Size of dataset inputs mismatches with size of input layer");
    }
    if outputs.len() != get_output_layer(layers).size {
        panic!("Size of dataset outputs mismatches with size of output layer");
    }
    dataset.push(Row { inputs, outputs });
}

impl NeuralNetwork {
    pub const fn new() -> Self {
        Self {
            layers: Vec::new(),
            training_dataset: Vec::new(),
            testing_dataset: Vec::new(),
        }
    }

    pub fn train(
        mut self,
        no_epochs: u32,
        learning_rate: f64,
        cost_function: CostFunction,
    ) -> Self {
        let layers = &mut self.layers;
        for epoch in 1..=no_epochs {
            for row in &self.training_dataset {
                forward_propagation(&row.inputs, layers);
                backward_propagation(learning_rate, &cost_function, &row.outputs, layers);
            }
            let error = get_error(&self.testing_dataset, &cost_function, layers);
            println!("Epoch #{}, error: {}", epoch, error);
        }
        self
    }

    pub fn add_layer(mut self, size: usize, activation_function: ActivationFunction) -> Self {
        let weights_in = match self.layers.last() {
            Some(layer) => Array::zeros((layer.size, size)),
            None => Array::zeros((1, 1)),
        };
        let outputs = Array1::zeros(size);
        let errors = vec![0_f64; size];
        let inputs = Array1::zeros(size);
        self.layers.push(Layer {
            size,
            weights_in,
            inputs,
            outputs,
            errors,
            activation_function,
        });
        self
    }

    pub fn add_training_row(mut self, inputs: Vec<f64>, outputs: Vec<f64>) -> Self {
        add_row(inputs, outputs, &self.layers, &mut self.training_dataset);
        self
    }

    pub fn add_testing_row(mut self, inputs: Vec<f64>, outputs: Vec<f64>) -> Self {
        add_row(inputs, outputs, &self.layers, &mut self.testing_dataset);
        self
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        get_output_layer(&self.layers).outputs.to_vec()
    }
}
