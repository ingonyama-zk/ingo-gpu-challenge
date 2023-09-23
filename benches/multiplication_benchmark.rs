extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};
use icicle_gpu_challenge::{BigInt256, sample_random_bigints256, multiply_cuda};
use rustacuda::prelude::DeviceBuffer;
use num_bigint::BigUint;
use ark_ff::PrimeField;


// This number should be large enough to fully saturate the GPU
const SIZE: usize = 1 << 25;

fn bench(c: &mut Criterion) {
    let _ctx = rustacuda::quick_init();

    let (a, b) = sample_random_bigints256(SIZE);

    let a_bigint: Vec<BigInt256> = a.iter().map(|a_num| 
        BigInt256 { s: BigUint::from(a_num.into_bigint()).to_u32_digits().try_into().unwrap() }
    ).collect();
    let b_bigint: Vec<BigInt256> = b.iter().map(|b_num| 
        BigInt256 { s: BigUint::from(b_num.into_bigint()).to_u32_digits().try_into().unwrap() }
    ).collect();
    let mut a_device = DeviceBuffer::from_slice(&a_bigint[..]).unwrap();
    let mut b_device = DeviceBuffer::from_slice(&b_bigint[..]).unwrap();

    c.bench_function(
        &format!("Benchmarking multiplication of size {}", SIZE),
        |b| b.iter(|| {
            multiply_cuda(&mut a_device, &mut b_device, SIZE, true);
        }),
    );
}

criterion_group!(mult_bench, bench);
criterion_main!(mult_bench);
