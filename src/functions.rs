#![allow(dead_code)]
use super::ftrait::Function;
use super::tensor::Wrapper;
use std::rc::Rc;
///////////////
// Declaration of Binary Functions :)
//////////////

// Add
pub struct Add {
    parents: [Rc<Wrapper>; 2],
}
impl Add {
    pub fn apply(p1: Rc<Wrapper>, p2: Rc<Wrapper>) -> Wrapper {
        Wrapper::new(
            Add::forward(&p1.data, &p2.data),
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }

    pub fn forward(x1: &[f64], x2: &[f64]) -> Vec<f64> {
        let mut _temp = Vec::new();
        if x1.len() != x2.len() {
            panic!(
                "Shape of first: {}, doesn't match shape of second:{}",
                x1.len(),
                x2.len()
            );
        } else {
            for i in 0..x1.len() {
                _temp.push(x1[i] + x2[i]);
            }
        }
        _temp
    }
}

impl Function for Add {
    fn backward(&self, grad: Vec<f64>) -> [Vec<f64>; 2] {
        [grad.clone(), grad] // CLONE!!
    }

    fn parents(&self) -> &[Rc<Wrapper>; 2] {
        &self.parents
    }
}

// Mul
pub struct Mul {
    parents: [Rc<Wrapper>; 2],
}
impl Mul {
    pub fn apply(p1: Rc<Wrapper>, p2: Rc<Wrapper>) -> Wrapper {
        Wrapper::new(
            Mul::forward(&p1.data, &p2.data),
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }

    fn forward(x1: &[f64], x2: &[f64]) -> Vec<f64> {
        let mut _temp = Vec::new();
        if x1.len() != x2.len() {
            panic!(
                "Shape of first: {}, doesn't match shape of second:{}",
                x1.len(),
                x2.len()
            );
        } else {
            for i in 0..x1.len() {
                _temp.push(x1[i] * x2[i]);
            }
        }
        _temp
    }
}

impl Function for Mul {
    fn backward(&self, grad: Vec<f64>) -> [Vec<f64>; 2] {
        [
            Mul::forward(&self.parents[1].data, &grad),
            Mul::forward(&self.parents[0].data, &grad),
        ]
    }

    fn parents(&self) -> &[Rc<Wrapper>; 2] {
        &self.parents
    }
}
