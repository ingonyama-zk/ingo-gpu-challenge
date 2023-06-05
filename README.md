# Ingo GPU challenge

Integer arithmetic is the backbone of cryptographic algorithms. However, the typical sizes that provide acceptable security are not directly supported by standard hardware such as CPUs and GPUs. Therefore, we need to represent our integers as arrays of multiple "limbs", either 32 or 64 bits in size.

## The challenge

Design a kernel that multiplies pairs 256-bit numbers, getting 512-bit results. The goal is to maximize the throughput of the mutliplier.

## Testing suite

To benchmark and test the code, we use a Rust wrapper. For Rust installation, see [this](https://www.rust-lang.org/tools/install) link.

Once Rust is installed, the correctness of the code can be verified by running:

```
cargo test
```

And the performance can be measured by running:

```
cargo bench
```

There is baseline code that implements section 4 of [this](http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf) paper in `cuda/mult.cu` file. It is expected that only this file will be edited, the rest of the repo is meant to provide the infrastructure for easy testing and benchmarking.

You can parallelize the multiplier, or do the opposite - the only optimization goal is throughput.

Any machine can be used, the only limitation is using a single GPU for measurements.

If there are any questions, you can file an issue in this repository.

Good luck and have fun!