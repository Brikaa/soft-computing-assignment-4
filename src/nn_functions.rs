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
    let mut res = 1.0 / (1.0 + E.powf(-x));
    if res <= 0.00001 {
        res = 0.00001;
    } else if res >= 0.99999 {
        res = 0.99999;
    }
    return res;
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

fn apply_tanh(x: f64) -> f64 {
    f64::tanh(x)
}

fn apply_tanh_derivative(_: f64, function_output: f64) -> f64 {
    1.0 - function_output * function_output
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

pub const TANH: ActivationFunction = ActivationFunction {
    apply: apply_tanh,
    apply_derivative: apply_tanh_derivative,
};

pub const MEAN_SQUARED_ERROR: CostFunction = CostFunction {
    apply: apply_mean_squared_error,
    apply_derivative: apply_mean_squared_error_derivative,
};
