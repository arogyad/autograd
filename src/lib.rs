mod ftrait;
mod functions;
mod tensor;

#[cfg(test)]
mod tests {
    use super::ftrait::Function;
    use super::tensor::Tensor;
    #[test]
    fn test1() {
        // Create 2 "Tensors" of f64
        let a = Tensor::new(2., None);
        let b = Tensor::new(3., None);

        // Create a context of multiplication for creation of DAG for backward pass
        let c_ctx = &a * &b;
        // Applying it to create a tensor containing the context that produced it
        let c = c_ctx.apply();

        // Another tensor
        let d = Tensor::new(4., None);

        // Create a context of addition
        let e_ctx = &c + &d;
        // Applying it
        let mut e = e_ctx.apply();

        // Backpropagation
        e.backward();

        assert!(a.grad.get() == 3.);
        assert!(b.grad.get() == 2.);
    }

    #[test]
    fn test2() {
        // Create 2 "Tensors" of f64
        let a = Tensor::new(2., None);
        let b = Tensor::new(3., None);

        // Create a context of multiplication for creation of DAG for backward pass
        let c_ctx = &a * &b;
        // Applying it to create a tensor containing the context that produced it
        let c = c_ctx.apply();

        // Another tensor
        let d = Tensor::new(4., None);

        // Create a context of multiplication
        let e_ctx = &c * &d;
        // Applying it
        let mut e = e_ctx.apply();

        // Backpropagation
        e.backward();

        assert!(a.grad.get() == 12.);
        assert!(b.grad.get() == 8.);
    }
}
