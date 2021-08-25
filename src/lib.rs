mod ftrait;
mod functions;
mod tensor;

#[cfg(test)]
mod tests {
    use super::tensor::Tensor;

    #[test]
    fn test_rc() {
        let a = Tensor::new(2., None);
        let b = Tensor::new(3., None);
        let c = &a * &b;
        let d = Tensor::new(4., None);
        let e = &c + &d;
        e.backward();
        assert!(a.grad.get() == 3.);
        assert!(b.grad.get() == 2.);
    }
}
