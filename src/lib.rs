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
    }
}
