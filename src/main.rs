mod neural_network;
mod nn_functions;

use std::{fmt, str::FromStr};

use neural_network::NeuralNetwork;
#[macro_use]
extern crate scan_fmt;

use crate::nn_functions::{LINEAR, MEAN_SQUARED_ERROR, SIGMOID};

fn get_input<T>(message: &str) -> T
where
    T: FromStr,
    T::Err: fmt::Debug,
{
    println!("{}", message);
    loop {
        if let Ok(input) = scanln_fmt!("{}", T) {
            return input;
        } else {
            println!("Invalid input, try again");
        }
    }
}

fn main() {
    let mut nn = NeuralNetwork::new();
    nn = nn
        .add_layer(4, SIGMOID)
        .add_layer(8, SIGMOID)
        .add_layer(1, LINEAR);
    for i in 1..=2 {
        let no_inputs = scanln_fmt!("{}", u32).unwrap();
        for _ in 1..=no_inputs {
            let (i1, i2, i3, i4, o) =
                scanln_fmt!("{} {} {} {} {}", f64, f64, f64, f64, f64).unwrap();
            let inputs = vec![i1, i2, i3, i4];
            let outputs = vec![o];
            if i == 1 {
                nn = nn.add_testing_row(inputs, outputs);
            } else {
                nn = nn.add_training_row(inputs, outputs);
            }
        }
    }
    nn = nn.train(11, 0.1, MEAN_SQUARED_ERROR);
    loop {
        let cement = get_input::<f64>("Cement");
        let water = get_input::<f64>("Water");
        let superplasticizer = get_input::<f64>("Superplasticizer");
        let age = get_input::<f64>("Age");
        nn = nn.propagate(vec![cement, water, superplasticizer, age]);
        let outputs = nn.get_outputs();
        println!(
            "Concrete compressive strength: {}",
            outputs.first().unwrap()
        );
    }
}
