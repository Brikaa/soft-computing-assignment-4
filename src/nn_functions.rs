use std::f64::consts::E;

pub struct ActivationFunction {
    pub apply: fn(f64) -> f64,
    pub apply_derivative: fn(f64) -> f64,
}

pub struct CostFunction {
    pub apply: fn(f64, f64) -> f64,
    pub apply_derivative: fn(f64, f64) -> f64,
}

fn apply_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn apply_sigmoid_derivative(prev_x: f64) -> f64 {
    prev_x * (1.0 - prev_x)
}

fn apply_linear(x: f64) -> f64 {
    x
}

fn apply_linear_derivative(_: f64) -> f64 {
    1.0
}

fn apply_mean_squared_error(output: f64, expected: f64) -> f64 {
    0.5 * (expected - output) * (expected - output)
}

fn apply_mean_squared_error_derivative(output: f64, expected: f64) -> f64 {
    expected - output
}

pub const SIGMOID_WITH_PREV_OUTPUT_DERIVATIVE: ActivationFunction = ActivationFunction {
    apply: apply_sigmoid,
    apply_derivative: apply_sigmoid_derivative,
};

pub const LINEAR: ActivationFunction = ActivationFunction {
    apply: apply_linear,
    apply_derivative: apply_linear_derivative,
};

pub const MEAN_SQUARED_ERROR: CostFunction = CostFunction {
    apply: apply_mean_squared_error,
    apply_derivative: apply_mean_squared_error_derivative,
};
