mod ftrait;
mod functions;
mod tensor;

pub mod benches {
    use super::tensor::Tensor;
    pub fn bench_rc(x: usize) {
        let a = Tensor::new(vec![2.; x], None);
        let b = Tensor::new(vec![3.; x], None);
        let c = &a * &b;
        let d = Tensor::new(vec![4.; x], None);
        let e = &c + &d;
        e.backward();
        println!("{:?}", a.grad.as_ptr());
    }
}

#[cfg(test)]
mod tests {
    use super::tensor::Tensor;
    #[test]
    fn test1() {
        let a = Tensor::new(vec![2.; 2], None);
        let b = Tensor::new(vec![3.; 2], None);
        let c = &a * &b;
        let d = Tensor::new(vec![4.; 2], None);
        let e = &c + &d;
        e.backward();
        println!("{:?}", a.grad.borrow());
    }
}
