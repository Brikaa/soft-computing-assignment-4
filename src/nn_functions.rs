use std::f64::consts::E;

pub struct ActivationFunction {
    pub apply: fn(x: f64) -> f64,
    pub apply_derivative: fn(x: f64, function_output: f64) -> f64,
}

pub struct CostFunction {
    pub apply: fn(output: f64, expected: f64) -> f64,
    pub apply_derivative: fn(output: f64, expected: f64) -> f64,
}

fn apply_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn apply_sigmoid_derivative(_: f64, function_output: f64) -> f64 {
    function_output * (1.0 - function_output)
}

fn apply_linear(x: f64) -> f64 {
    x
}

fn apply_linear_derivative(_: f64, _: f64) -> f64 {
    1.0
}

fn apply_mean_squared_error(output: f64, expected: f64) -> f64 {
    0.5 * (expected - output) * (expected - output)
}

fn apply_mean_squared_error_derivative(output: f64, expected: f64) -> f64 {
    expected - output
}

fn apply_relu(x: f64) -> f64 {
    if x < 0_f64 {
        return 0_f64;
    } else {
        return x;
    }
}

fn apply_relu_derivative(x: f64, _: f64) -> f64 {
    if x < 0_f64 {
        return 0_f64;
    } else {
        return 1_f64;
    }
}

pub const SIGMOID: ActivationFunction = ActivationFunction {
    apply: apply_sigmoid,
    apply_derivative: apply_sigmoid_derivative,
};

pub const LINEAR: ActivationFunction = ActivationFunction {
    apply: apply_linear,
    apply_derivative: apply_linear_derivative,
};

pub const RELU: ActivationFunction = ActivationFunction {
    apply: apply_relu,
    apply_derivative: apply_relu_derivative,
};

pub const MEAN_SQUARED_ERROR: CostFunction = CostFunction {
    apply: apply_mean_squared_error,
    apply_derivative: apply_mean_squared_error_derivative,
};
