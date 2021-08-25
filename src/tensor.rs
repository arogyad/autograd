#![allow(unused)]
use super::ftrait::Function;
use core::cmp::{Eq, PartialEq};
use std::cell::Cell;
use std::ops::{Add, Deref, Mul};
use std::rc::Rc;

// The "Wrapper" class, with very very basic things implemented
// The actual tensor class is defined after this definition. Using Rc<Wrapper> makes it less
// idiomatic but more(extremely) clean.
pub struct Wrapper {
    pub data: f64,
    pub grad: Cell<f64>,
    pub _ctx: Option<Box<dyn Function>>,
}

impl Wrapper {
    pub fn new(data: f64, _ctx: Option<Box<dyn Function>>) -> Self {
        Self {
            data,
            grad: Cell::new(0.),
            _ctx,
        }
    }

    // Toposort: From tinygrad + pytorch. Checking of it existing or not
    // implemented.
    fn _deepwalk<'a>(
        node: &'a Wrapper,
        nodes: &'_ mut Vec<&'a Wrapper>,
        visited: &mut Vec<&'a Wrapper>,
    ) {
        if let Some(n) = &node._ctx {
            visited.push(node);
            for i in n.parents() {
                if !visited.contains(&i.as_ref()) {
                    Self::_deepwalk(i, nodes, visited);
                }
            }
            nodes.push(node);
        }
    }

    fn walk(&self) -> Vec<&Wrapper> {
        let mut nodes = Vec::new();
        let mut visited = Vec::new();
        Self::_deepwalk(self, &mut nodes, &mut visited);
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) {
        self.grad.set(1.);
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

impl PartialEq for Wrapper {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for Wrapper {}

// Actual Tensor Implementation, The tensor is Rc<Wrapper> so we put it inside a thin wrapper. This
// is opposite of what should be done.
pub struct Tensor(pub Rc<Wrapper>);

impl Tensor {
    pub fn new(data: f64, _ctx: Option<Box<dyn Function>>) -> Self {
        Self(Rc::new(Wrapper {
            data,
            grad: Cell::new(0.),
            _ctx,
        }))
    }

    fn get(&self) -> Rc<Wrapper> {
        Rc::clone(&self.0)
    }
}
impl Deref for Tensor {
    type Target = Wrapper;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Self::Output {
        Tensor(Rc::new(crate::functions::Add::apply(
            self.get(),
            other.get(),
        )))
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Self::Output {
        Tensor(Rc::new(crate::functions::Mul::apply(
            self.get(),
            other.get(),
        )))
    }
}
