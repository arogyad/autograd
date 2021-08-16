#![allow(dead_code)]
use super::ftrait::Function;
use super::tensor::Tensor;
///////////////
// Declaration of Binary Functions :)
//////////////

// Add
pub struct Add<'a> {
    parents: [&'a Tensor<'a>; 2],
}
impl<'a> Add<'a> {
    pub fn new(parents: [&'a Tensor; 2]) -> Self {
        Self { parents }
    }
}

impl<'a> Function for Add<'a> {
    fn apply(&self) -> Tensor {
        Tensor::new(self.forward(), Some(self))
    }

    fn forward(&self) -> f64 {
        self.parents[0].data + self.parents[1].data
    }

    fn backward(&self, grad: f64) -> [f64; 2] {
        [grad, grad]
    }

    fn parents(&self) -> [&Tensor; 2] {
        self.parents
    }
}

// Mul
pub struct Mul<'a> {
    parents: [&'a Tensor<'a>; 2],
}
impl<'a> Mul<'a> {
    pub fn new(parents: [&'a Tensor; 2]) -> Self {
        Self { parents }
    }
}

impl<'a> Function for Mul<'a> {
    fn apply(&self) -> Tensor {
        Tensor::new(self.forward(), Some(self))
    }

    fn forward(&self) -> f64 {
        self.parents[0].data * self.parents[1].data
    }

    fn backward(&self, grad: f64) -> [f64; 2] {
        [self.parents[1].data * grad, self.parents[0].data * grad]
    }

    fn parents(&self) -> [&Tensor; 2] {
        self.parents
    }
}
