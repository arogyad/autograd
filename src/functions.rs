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
            p1.data * p2.data,
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }
}

impl Function for Add {
    fn backward(&self, grad: f64) -> [f64; 2] {
        [grad, grad]
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
            p1.data * p2.data,
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }
}

impl Function for Mul {
    fn backward(&self, grad: f64) -> [f64; 2] {
        [self.parents[1].data * grad, self.parents[0].data * grad]
    }

    fn parents(&self) -> &[Rc<Wrapper>; 2] {
        &self.parents
    }
}
