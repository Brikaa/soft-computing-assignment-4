mod neural_network;
pub mod nn_functions;

use std::{fmt, str::FromStr};

use neural_network::NeuralNetwork;
use nn_functions::SIGMOID;
#[macro_use]
extern crate scan_fmt;

use crate::nn_functions::{LINEAR, MEAN_SQUARED_ERROR};

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
        .add_layer(4, SIGMOID) // Doesn't matter, this is the input layer
        .add_layer(20, SIGMOID)
        .add_layer(1, LINEAR);
    let i1_d = 286_f64;
    let i2_d = 178_f64;
    let i3_d = 6.5_f64;
    let i4_d = 49_f64;
    for i in 1..=2 {
        let no_inputs = scanln_fmt!("{}", u32).unwrap();
        for _ in 1..=no_inputs {
            let (i1, i2, i3, i4, o) =
                scanln_fmt!("{} {} {} {} {}", f64, f64, f64, f64, f64).unwrap();
            let inputs = vec![i1 / i1_d, i2 / i2_d, i3 / i3_d, i4 / i4_d];
            let outputs = vec![o];
            if i == 1 {
                nn = nn.add_testing_row(inputs, outputs);
            } else {
                nn = nn.add_training_row(inputs, outputs);
            }
        }
    }
    nn = nn.train(4000, 0.0001, MEAN_SQUARED_ERROR);
    loop {
        let i1 = get_input::<f64>("Cement");
        let i2 = get_input::<f64>("Water");
        let i3 = get_input::<f64>("Superplasticizer");
        let i4 = get_input::<f64>("Age");
        nn = nn.propagate(vec![i1 / i1_d, i2 / i2_d, i3 / i3_d, i4 / i4_d]);
        let outputs = nn.get_outputs();
        println!(
            "Concrete compressive strength: {}",
            outputs.first().unwrap()
        );
    }
}
