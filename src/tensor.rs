#![allow(unused)]
use super::ftrait::Function;
use std::cell::Cell;

// The "Tensor" class, with very very basic things implemented
// TODO: Addition of require_grad, is_leaf, addition of utility functions and traits(detach, require_grad, Display, Hash etc.), optimization based on those fields
pub struct Tensor<'a> {
    pub data: f64,
    pub grad: Cell<f64>,
    pub _ctx: Option<Box<&'a dyn Function>>,
}

impl<'a> Tensor<'a> {
    pub fn new(data: f64, _ctx: Option<Box<&'a dyn Function>>) -> Self {
        Self {
            data,
            grad: Cell::new(0.),
            _ctx,
        }
    }

    // Toposort: From tinygrad + pytorch. Checking of it existing or not
    // implemented.
    // TODO: Implement Hash Trait for Tensor
    fn _deepwalk(node: &'a Tensor<'a>, nodes: &'_ mut Vec<&'a Tensor<'a>>) {
        if let Some(n) = &node._ctx {
            for i in n.parents() {
                Self::_deepwalk(i, nodes);
            }
            nodes.push(node);
        }
    }

    fn walk(&'a self) -> Vec<&Tensor> {
        let mut nodes = Vec::new();
        Self::_deepwalk(self, &mut nodes);
        nodes.reverse();
        nodes
    }

    pub fn backward(&mut self) {
        self.grad = Cell::new(1.);
        // TODO: require_grad check and is_leaf checks to prevent unnecessary grad creations
        for t0 in self.walk() {
            let grads = t0._ctx.as_ref().unwrap().backward(t0.grad.get());
            if t0._ctx.as_ref().unwrap().parents().len() == 1 {
                let grads = [grads];
            }
            for (t, g) in t0._ctx.as_ref().unwrap().parents().iter().zip(grads) {
                if t.grad.get() != 0. {
                    t.grad.set(t.grad.get() + g)
                } else {
                    t.grad.set(g);
                }
            }
        }
    }
}
