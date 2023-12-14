use ndarray::{Array, Array1, Array2};

use crate::nn_functions::ActivationFunction;

struct Layer {
    size: u32,
    weights_in: Array2<f64>,
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
    training_dataset: Vec<Dataset>,
    testing_dataset: Vec<Dataset>,
}

fn get_input_layer(layers: &Vec<Layer>) -> &Layer {
    &layers[0]
}

fn get_output_layer(layers: &Vec<Layer>) -> &Layer {
    &layers[layers.len() - 1]
}

fn get_input_layer_mut(layers: &mut Vec<Layer>) -> &mut Layer {
    &mut layers[0]
}

fn get_output_layer_mut(layers: &mut Vec<Layer>) -> &mut Layer {
    let i = layers.len() - 1;
    &mut layers[i]
}

fn forward_propagation(features: &Vec<f64>, layers: &mut Vec<Layer>) {
    get_input_layer_mut(layers).outputs = Array::from_vec(features.clone());
    for l in 1..layers.len() {
        // TODO: ndarray multiplication
        let inputs = layers[l - 1].outputs.dot(&layers[l].weights_in);
        layers[l].outputs = inputs.map(|inp| (layers[l].activation_function.apply)(*inp));
    }
}
