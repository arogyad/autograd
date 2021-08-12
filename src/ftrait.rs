use super::tensor::Tensor;

// The trait that all the functions should implement
// This is currently implemented for BinaryOperations only
// Returing a dynamic size container might do the trick?!
pub trait Function {
    fn apply(&self) -> Tensor;
    fn backward(&self, grad: f64) -> [f64; 2]; // The number of grads out == Number of parents/Number of Inputs. Incase of BinaryOperations it's 2
    fn forward(&self) -> f64;
    fn parents(&self) -> [&Tensor; 2];
}
