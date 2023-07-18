#include <cstdint>
#include <cuda.h>
#include <stdexcept>


namespace ptx {

    __device__ __forceinline__ uint32_t add_cc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }
    
    __device__ __forceinline__ uint32_t addc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("addc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }
    
    __device__ __forceinline__ uint32_t addc_cc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }

    __device__ __forceinline__ uint32_t sub_cc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
      }
      
      __device__ __forceinline__ uint32_t subc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("subc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
      }
      
      __device__ __forceinline__ uint32_t subc_cc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
      }      
    
    __device__ __forceinline__ uint32_t mul_lo(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }
    
    __device__ __forceinline__ uint32_t mul_hi(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }
    
    __device__ __forceinline__ uint32_t mad_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

    __device__ __forceinline__ uint32_t madc_hi(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }
    
    __device__ __forceinline__ uint32_t madc_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }
    
    __device__ __forceinline__ uint32_t madc_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

} // namespace ptx


struct __align__(16) bigint {
    uint32_t limbs[8];
};

struct __align__(16) bigint_wide {
    uint32_t limbs[16];
};

// stands for "total limbs count"
const int TLC = 8;

static __device__ __forceinline__ void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    #pragma unroll
    for (size_t i = 0; i < n; i += 2) {
        acc[i] = ptx::mul_lo(a[i], bi);
        acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
}

static __device__ __forceinline__ void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

    #pragma unroll
    for (size_t i = 2; i < n; i += 2) {
        acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
}

static __device__ __forceinline__ void mad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    cmad_n(odd, a + 1, bi, n - 2);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, 0);
    odd[n - 1] = ptx::madc_hi(a[n - 1], bi, 0);
    cmad_n(even, a, bi, n);
    odd[n - 1] = ptx::addc(odd[n - 1], 0);
}

template <bool SUBTRACT, bool CARRY_OUT> 
static constexpr __device__ __forceinline__ uint32_t add_sub_limbs_device(const uint32_t *x, const uint32_t *y, uint32_t *r, size_t n = (TLC >> 1)) {
    r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    for (unsigned i = 1; i < (CARRY_OUT ? n : n - 1); i++)
        r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
    if (!CARRY_OUT) {
        r[n - 1] = SUBTRACT ? ptx::subc(x[n - 1], y[n - 1]) : ptx::addc(x[n - 1], y[n - 1]);
        return 0;
    }
    return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
}

static __device__ __forceinline__ void multiply_short_raw_device(const uint32_t *a, const uint32_t *b, uint32_t *even) {
    __align__(8) uint32_t odd[TLC - 2];
    mul_n(even, a, b[0], TLC >> 1);
    mul_n(odd, a + 1, b[0], TLC >> 1);
    mad_row(&even[2], &odd[0], a, b[1], TLC >> 1);
    size_t i;
#pragma unroll
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
        mad_row(&odd[i], &even[i], a, b[i], TLC >> 1);
        mad_row(&even[i + 2], &odd[i], a, b[i + 1], TLC >> 1);
    }
    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < TLC - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
}

static __device__ __forceinline__ void multiply_raw_device(const bigint &as, const bigint &bs, bigint_wide &rs) {
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t *r = rs.limbs;
    multiply_short_raw_device(a, b, r);
    multiply_short_raw_device(&a[TLC >> 1], &b[TLC >> 1], &r[TLC]);
    __align__(16) uint32_t temp1[TLC];
    __align__(16) uint32_t temp[TLC];
    uint32_t carry1 = add_sub_limbs_device<false, true>(a, &a[TLC >> 1], temp);
    uint32_t carry2 = add_sub_limbs_device<false, true>(b, &b[TLC >> 1], &temp[TLC >> 1]);
    //add_sub_limbs_device<true, false>(&r[TLC >> 1], temp1, &r[TLC >> 1], TLC);
    multiply_short_raw_device(temp, &temp[TLC >> 1], temp1);
    if (carry1)
        add_sub_limbs_device<false, false>(&temp1[TLC >> 1], &temp[TLC >> 1], &temp1[TLC >> 1]);
    if (carry2)
        add_sub_limbs_device<false, false>(&temp1[TLC >> 1], temp, &temp1[TLC >> 1]);
    add_sub_limbs_device<true, false>(temp1, &r[TLC], temp1, TLC);
    add_sub_limbs_device<true, false>(temp1, r, temp1, TLC);
    uint32_t carry = add_sub_limbs_device<false, true>(&r[TLC >> 1], temp1, &r[TLC >> 1], TLC);
    if (carry) {
        // printf("here \n");
        r[TLC + (TLC >> 1)] = ptx::addc(r[TLC + (TLC >> 1)], 0);
    }
}

// a method to create a 256-bit number from 512-bit result to be able to perpetually
// repeat the multiplication using registers
bigint __device__ __forceinline__ get_upper_half(const bigint_wide &x) {
    bigint out{};
    #pragma unroll
    for (unsigned i = 0; i < TLC; i++)
        out.limbs[i] = x.limbs[TLC - 1 + i];
    return out;
  }


// The kernel that does element-wise multiplication of arrays in1 and in2 N times
template <int N>
__global__ void multVectorsKernel(bigint *in1, const bigint *in2, bigint_wide *out, size_t n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
    {
        bigint i1 = in1[tid];
        const bigint i2 = in2[tid];
        bigint_wide o = {0};
        // #pragma unroll
        for (int i = 0; i < N - 1; i++) {
            multiply_raw_device(i1, i2, o);
            i1 = get_upper_half(o);
        }
        multiply_raw_device(i1, i2, out[tid]);
    }
}

template <int N>
int mult_vectors(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    // Set the grid and block dimensions
    int threads_per_block = 128;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block + 1;

    multVectorsKernel<N><<<num_blocks, threads_per_block>>>(in1, in2, out, n);

    return 0;
}


extern "C"
int multiply_test(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    try
    {
        mult_vectors<1>(in1, in2, out, n);
        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        return -1;
    }
}

extern "C"
int multiply_bench(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    try
    {
        // for benchmarking, we need to give each thread a number of multiplication tasks that would ensure
        // that we're mostly measuring compute and not global memory accesses, which is why we do 500 multiplications here
        mult_vectors<500>(in1, in2, out, n);
        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        return -1;
    }
}

