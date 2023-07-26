use rustacuda::{memory::{DeviceCopy, DevicePointer}, prelude::*};
use num_bigint::{RandomBits, BigUint};
use rand::prelude::Distribution;


pub const TLC: usize = 8;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct BigInt256 {
    pub s: [u32; TLC],
}

unsafe impl DeviceCopy for BigInt256 {}

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct BigInt512 {
    pub s: [u32; 2 * TLC],
}

unsafe impl DeviceCopy for BigInt512 {}


extern "C" {
    fn multiply_test(
        in1: DevicePointer<BigInt256>,
        in2: DevicePointer<BigInt256>,
        out: DevicePointer<BigInt512>,
        count: usize,
    ) -> usize;
}

extern "C" {
    fn multiply_bench(
        in1: DevicePointer<BigInt256>,
        in2: DevicePointer<BigInt256>,
        out: DevicePointer<BigInt512>,
        count: usize,
    ) -> usize;
}

pub fn multiply_cuda(
    in1: &mut DeviceBuffer<BigInt256>,
    in2: &mut DeviceBuffer<BigInt256>,
    count: usize,
    bench: bool,
) -> DeviceBuffer<BigInt512> {
    let mut res: DeviceBuffer<BigInt512> = unsafe { DeviceBuffer::uninitialized(count).unwrap() };
    let err = unsafe {
        if bench {
            multiply_bench(
                in1.as_device_ptr(),
                in2.as_device_ptr(),
                res.as_device_ptr(),
                count,
            )
        } else {
            multiply_test(
                in1.as_device_ptr(),
                in2.as_device_ptr(),
                res.as_device_ptr(),
                count,
            )
        }
    };
    if err != 0 {
        panic!("Error {} occured", err);
    }
    res
}


pub fn sample_random_bigints256(n: usize) -> (Vec<BigUint>, Vec<BigUint>) {
    let mut rng = rand::thread_rng();
    let random_bits = RandomBits::new(32 * TLC as u64);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for _ in 0..n {
        let mut a_sampled: BigUint = random_bits.sample(&mut rng);
        a_sampled.set_bit((32 * TLC - 1) as u64, false);
        a.push(a_sampled);
        let mut b_sampled: BigUint = random_bits.sample(&mut rng);
        b_sampled.set_bit((32 * TLC - 1) as u64, false);
        b.push(b_sampled);
    }
    (a, b)
}

// A test to check the correctness of the CUDA multiplier;
// Uses `num_bigint` crate to check against.
#[test]
fn test_mult() {
    let _ctx = rustacuda::quick_init();

    let n = 1 << 20;
    let (a, b) = sample_random_bigints256(n);

    let a_bigint: Vec<BigInt256> = a.iter().map(|a_num| {
        let mut a_digits = a_num.to_u32_digits();
        a_digits.resize(TLC, 0);
        BigInt256 { s: a_digits.try_into().unwrap() }
    }).collect();
    let b_bigint: Vec<BigInt256> = b.iter().map(|b_num| {
        let mut b_digits = b_num.to_u32_digits();
        b_digits.resize(TLC, 0);
        BigInt256 { s: b_digits.try_into().unwrap() }
    }).collect();
    let mut a_device = DeviceBuffer::from_slice(&a_bigint[..]).unwrap();
    let mut b_device = DeviceBuffer::from_slice(&b_bigint[..]).unwrap();

    let res_device = multiply_cuda(&mut a_device, &mut b_device, n, false);
    let mut res: Vec<BigInt512> = (0..n).map(|_| BigInt512 { s: [0; 2 * TLC] }).collect();
    res_device.copy_to(&mut res[..]).unwrap();

    for i in 0..n {
        let mut res_digits = (a[i].clone() * b[i].clone()).to_u32_digits();
        res_digits.resize(2 * TLC, 0);
        assert_eq!(res[i], BigInt512 { s: res_digits.try_into().unwrap() } );
    }
}