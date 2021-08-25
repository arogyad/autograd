use super::tensor::Wrapper;
use std::rc::Rc;

// The trait that all the functions should implement
// This is currently implemented for BinaryOperations only
// Returing a dynamic size container might do the trick?!
pub trait Function {
    fn backward(&self, grad: Vec<f64>) -> [Vec<f64>; 2]; // The number of grads out == Number of parents/Number of Inputs. Incase of BinaryOperations it's 2
    fn parents(&self) -> &[Rc<Wrapper>; 2];
}
