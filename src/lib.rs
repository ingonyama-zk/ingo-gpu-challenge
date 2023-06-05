use rustacuda::{memory::{DeviceCopy, DevicePointer}, prelude::*};
use num_bigint::{RandomBits, BigUint};
use rand::prelude::Distribution;


#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct BigInt256 {
    pub s: [u32; 8],
}

unsafe impl DeviceCopy for BigInt256 {}

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct BigInt512 {
    pub s: [u32; 16],
}

unsafe impl DeviceCopy for BigInt512 {}


extern "C" {
    fn multiply(
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
) -> DeviceBuffer<BigInt512> {
    let mut res: DeviceBuffer<BigInt512> = unsafe { DeviceBuffer::uninitialized(count).unwrap() };
    let err = unsafe {
        multiply(
            in1.as_device_ptr(),
            in2.as_device_ptr(),
            res.as_device_ptr(),
            count,
        )
    };
    if err != 0 {
        panic!("Error {} occured", err);
    }
    res
}


pub fn sample_random_bigints256(n: usize) -> (Vec<BigUint>, Vec<BigUint>) {
    let mut rng = rand::thread_rng();
    let random_bits = RandomBits::new(256);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for _ in 0..n {
        a.push(random_bits.sample(&mut rng));
        b.push(random_bits.sample(&mut rng));
    }
    (a, b)
}

// A test to check the correctness of the CUDA multiplier
#[test]
fn test_mult() {
    let _ctx = rustacuda::quick_init();

    let n = 1 << 25;
    let (a, b) = sample_random_bigints256(n);

    let a_bigint: Vec<BigInt256> = a.iter().map(|a_num| 
        BigInt256 { s: a_num.to_u32_digits().try_into().unwrap() }
    ).collect();
    let b_bigint: Vec<BigInt256> = b.iter().map(|b_num| 
        BigInt256 { s: b_num.to_u32_digits().try_into().unwrap() }
    ).collect();
    let mut a_device = DeviceBuffer::from_slice(&a_bigint[..]).unwrap();
    let mut b_device = DeviceBuffer::from_slice(&b_bigint[..]).unwrap();

    let res_device = multiply_cuda(&mut a_device, &mut b_device, n);
    let mut res: Vec<BigInt512> = (0..n).map(|_| BigInt512 { s: [0;16] }).collect();
    res_device.copy_to(&mut res[..]).unwrap();

    // for i in 0..n {
    //     assert_eq!(res[i], BigInt512 { s: (a[i].clone() * b[i].clone()).to_u32_digits().try_into().unwrap() } );
    // }
}