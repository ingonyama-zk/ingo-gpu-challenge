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

    __device__ __forceinline__ uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

    __device__ __forceinline__ uint32_t madc_lo(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.lo.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
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
    
    __device__ __forceinline__ uint32_t mad_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

    __device__ __forceinline__ uint32_t madc_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

} // namespace ptx


// stands for "total limbs count"
const int TLC = 8;

struct __align__(16) bigint {
    uint32_t limbs[TLC];
};

struct __align__(16) bigint_wide {
    uint32_t limbs[2 * TLC];
};

static __device__ __forceinline__ void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    size_t i;
    #pragma unroll
    for (i = 0; i < n - 2; i += 2) {
        acc[i] = ptx::mul_lo(a[i], bi);
        acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
    acc[i] = ptx::mul_lo(a[i], bi);
    if (i == n - 2) acc[i + 1] = ptx::mul_hi(a[i], bi);
}

static __device__ __forceinline__ uint32_t mul_n_plus_extra(uint32_t *acc, const uint32_t *a, uint32_t bi, uint32_t *extra, size_t n = (TLC >> 1)) {
    acc[0] = ptx::mad_lo_cc(a[0], bi, extra[0]);

    #pragma unroll
    for (size_t i = 1; i < n - 1; i += 2) {
        acc[i] = ptx::madc_hi_cc(a[i - 1], bi, extra[i]);
        acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, extra[i + 1]);
    }

    acc[n - 1] = ptx::madc_hi_cc(a[n - 2], bi, extra[n - 1]);
    return ptx::addc(0, 0);
}

template <bool CARRY_IN = false>
static __device__ __forceinline__ void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC, uint32_t optional_carry = 0) {
    if (CARRY_IN)
        ptx::add_cc(UINT32_MAX, optional_carry);
    acc[0] = CARRY_IN ? ptx::madc_lo_cc(a[0], bi, acc[0]) : ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

    #pragma unroll
    for (size_t i = 2; i < n; i += 2) {
        acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
}

template <bool EVEN_PHASE>
static __device__ __forceinline__ void cmad_n_msb(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    if (EVEN_PHASE) {
        acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
        acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
    } else {
        acc[1] = ptx::mad_hi_cc(a[0], bi, acc[1]);
    }

    #pragma unroll
    for (size_t i = 2; i < n; i += 2) {
        acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
}

template <bool CARRY_OUT = false, bool CARRY_IN = false>
static __device__ __forceinline__ uint32_t mad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC, uint32_t ci = 0, uint32_t di = 0, uint32_t carry_for_high = 0, uint32_t carry_for_low = 0) {
    cmad_n<CARRY_IN>(odd, a + 1, bi, n - 2, carry_for_low);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, ci);
    odd[n - 1] = CARRY_OUT ? ptx::madc_hi_cc(a[n - 1], bi, di) : ptx::madc_hi(a[n - 1], bi, di);
    uint32_t cr = CARRY_OUT ? ptx::addc(0, 0) : 0;
    cmad_n(even, a, bi, n);
    odd[n - 1] = CARRY_OUT ? ptx::addc_cc(odd[n - 1], carry_for_high) : ptx::addc(odd[n - 1], carry_for_high);
    if (CARRY_OUT)
        cr = ptx::addc(cr, 0);
    return cr;
}

template <bool EVEN_PHASE>
static __device__ __forceinline__ void mad_row_msb(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    cmad_n_msb<!EVEN_PHASE>(odd, EVEN_PHASE ? a : (a + 1), bi, n - 2);
    odd[EVEN_PHASE ? (n - 1) : (n - 2)] = ptx::madc_lo_cc(a[n - 1], bi, 0);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::madc_hi(a[n - 1], bi, 0);
    cmad_n_msb<EVEN_PHASE>(even, EVEN_PHASE ? (a + 1) : a, bi, n - 1);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::addc(odd[EVEN_PHASE ? n : (n - 1)], 0);
}

static __device__ __forceinline__ void cmad_n_lsb(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    if (n > 1)
        acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    else
        acc[0] = ptx::mad_lo(a[0], bi, acc[0]);

    size_t i;
    #pragma unroll
    for (i = 1; i < n - 1; i += 2) {
        acc[i] = ptx::madc_hi_cc(a[i - 1], bi, acc[i]);
        if (i == n - 2)
            acc[i + 1] = ptx::madc_lo(a[i + 1], bi, acc[i + 1]);
        else
            acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, acc[i + 1]);
    }
    if (i == n - 1) acc[i] = ptx::madc_hi(a[i - 1], bi, acc[i]);
}

static __device__ __forceinline__ void mad_row_lsb(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    if (bi != 0) {
        if (n > 1) cmad_n_lsb(odd, a + 1, bi, n - 1);
        cmad_n_lsb(even, a, bi, n);
    }
    return;
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

// This method multiplies `a` and `b` and adds `in1` and `in2` to the result
// It is used to compute the "middle" part of Karatsuba: `a0 * b1 + b0 * a1`
// So under the assumption that the top bits of `a` and `b` are unset, we can ignore all the carries from here
static __device__ __forceinline__ void multiply_and_add_short_raw_device(const uint32_t *a, const uint32_t *b, uint32_t *even, uint32_t *in1, uint32_t *in2) {
    __align__(16) uint32_t odd[TLC - 2];
    uint32_t first_row_carry = mul_n_plus_extra(even, a, b[0], in1);
    uint32_t carry = mul_n_plus_extra(odd, a + 1, b[0], &in2[1]);

    size_t i;
    #pragma unroll
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
        carry = mad_row<true, false>(&even[i], &odd[i - 2], a, b[i - 1], TLC >> 1, in1[(TLC >> 1) + i - 2], in1[(TLC >> 1) + i - 1], carry);
        carry = mad_row<true, false>(&odd[i], &even[i], a, b[i], TLC >> 1, in2[(TLC >> 1) + i - 1], in2[(TLC >> 1) + i], carry);
    }
    mad_row<false, true>(&even[TLC >> 1], &odd[(TLC >> 1) - 2], a, b[(TLC >> 1) - 1], TLC >> 1, in1[TLC - 2], in1[TLC - 1], carry, first_row_carry);
    // merge |even| and |odd| plus the parts of in2 we haven't added yet
    even[0] = ptx::add_cc(even[0], in2[0]);
    for (i = 0; i < (TLC - 2); i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], in2[i + 1]);
}

static __device__ __forceinline__ void multiply_short_raw_device(const uint32_t *a, const uint32_t *b, uint32_t *even) {
    __align__(16) uint32_t odd[TLC - 2];
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

static __device__ __forceinline__ void multiply_and_add_lsb_raw_device(const bigint &as, const bigint &bs, bigint &cs, bigint_wide &rs) {
    // r = a * b is correct for the first TLC digits
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t *even = rs.limbs;
    __align__(16) uint32_t odd[2 * TLC - 2];
    size_t i;
    if (b[0] == UINT32_MAX) {
        add_sub_limbs_device<true, false>(cs.limbs, a, even, TLC);
        for (i = 0; i < TLC - 1; i += 1)
            odd[i] = a[i];
    } else {
        mul_n_plus_extra(even, a, b[0], cs.limbs, TLC);
        mul_n(odd, a + 1, b[0], TLC - 1);
    }
    mad_row_lsb(&even[2], &odd[0], a, b[1], TLC - 1);
#pragma unroll
    for (i = 2; i < TLC - 1; i += 2) {
        mad_row_lsb(&odd[i], &even[i], a, b[i], TLC - i);
        mad_row_lsb(&even[i + 2], &odd[i], a, b[i + 1], TLC - i - 1);
    }

    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < TLC + 1; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
}

static __device__ __forceinline__ void multiply_msb_raw_device(const bigint& as, const bigint& bs, bigint_wide& rs) {
    // r = a * b is almost correct for the last TLC + 1 digits
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t *even = rs.limbs;
    __align__(16) uint32_t odd[2 * TLC - 2];

    even[TLC - 1] = ptx::mul_hi(a[TLC - 2], b[0]);
    odd[TLC - 2] = ptx::mul_lo(a[TLC - 1], b[0]);
    odd[TLC - 1] = ptx::mul_hi(a[TLC - 1], b[0]);
    size_t i;
#pragma unroll
    for (i = 2; i < TLC - 1; i += 2) {
        mad_row_msb<true>(&even[TLC - 2], &odd[TLC - 2], &a[TLC - i - 1], b[i - 1], i + 1);
        mad_row_msb<false>(&odd[TLC - 2], &even[TLC - 2], &a[TLC - i - 2], b[i], i + 2);
    }
    mad_row(&even[TLC], &odd[TLC - 2], a, b[TLC - 1]);

    // merge |even| and |odd|
    ptx::add_cc(even[TLC - 1], odd[TLC - 2]);
    for (i = TLC - 1; i < 2 * TLC - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
}

static constexpr unsigned slack_bits = 3;

static constexpr __device__ __forceinline__ bigint get_higher_with_slack(const bigint_wide &xs) {
    bigint out{};
    for (unsigned i = 0; i < TLC; i++) {
        out.limbs[i] = __funnelshift_lc(xs.limbs[i + TLC - 1], xs.limbs[i + TLC], 2 * slack_bits);
    }
    return out;
}

static constexpr __device__ __forceinline__ bigint get_higher(const bigint_wide &xs) {
    bigint out{};
    for (unsigned i = 0; i < TLC; i++) {
        out.limbs[i] = xs.limbs[i + TLC];
    }
    return out;
}

static constexpr __device__ __forceinline__ bigint get_lower(const bigint_wide &xs) {
    bigint out{};
    for (unsigned i = 0; i < TLC; i++)
        out.limbs[i] = xs.limbs[i];
    return out;
}

static __device__ __forceinline__ void multiply_raw_device(const bigint &as, const bigint &bs, bigint_wide &rs) {
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t *r = rs.limbs;
    multiply_short_raw_device(a, b, r);
    multiply_short_raw_device(&a[TLC >> 1], &b[TLC >> 1], &r[TLC]);
    __align__(16) uint32_t middle_part[TLC];
    __align__(16) uint32_t diffs[TLC];
    uint32_t carry1 = add_sub_limbs_device<true, true>(&a[TLC >> 1], a, diffs);
    uint32_t carry2 = add_sub_limbs_device<true, true>(b, &b[TLC >> 1], &diffs[TLC >> 1]);
    multiply_and_add_short_raw_device(diffs, &diffs[TLC >> 1], middle_part, r, &r[TLC]);
    if (carry1)
        add_sub_limbs_device<true, false>(&middle_part[TLC >> 1], &diffs[TLC >> 1], &middle_part[TLC >> 1]);
    if (carry2)
        add_sub_limbs_device<true, false>(&middle_part[TLC >> 1], diffs, &middle_part[TLC >> 1]);
    add_sub_limbs_device<false, true>(&r[TLC >> 1], middle_part, &r[TLC >> 1], TLC);

    for (size_t i = TLC + (TLC >> 1); i <  2 * TLC; i++)
        r[i] = ptx::addc_cc(r[i], 0);
}

static constexpr __device__ __forceinline__ bigint get_m() {
    return bigint {0x151e79ea, 0xf5204c21, 0x8d69e258, 0xfd0a180b, 0xfaa80548, 0xe4e51e49, 0xc40b2c9e, 0x36d9491e};
}

static constexpr __device__ __forceinline__ bigint get_neg_modulus() {
    return bigint {0xffffffff, 0xf5ee7fff, 0x2ffffffe, 0xa6558901, 0xa3c84ffe, 0x9f4bb2e1, 0x65d35aa9, 0xed549aa1};
}

static constexpr __device__ __forceinline__ bigint_wide get_modulus_wide() {
    return bigint_wide {
      0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe, 0x5c37b001, 0x60b44d1e, 0x9a2ca556, 0x12ab655e,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
}

static constexpr __device__ __forceinline__ bigint_wide get_two_modulus_wide() {
    return bigint_wide { 
      0x00000002, 0x14230000, 0xa0000002, 0xb354edfd, 0xb86f6002, 0xc1689a3c, 0x34594aac, 0x2556cabd,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
}

static __device__ __forceinline__ uint32_t sub_limbs_partial_device(const bigint_wide &as, const bigint_wide &bs, bigint_wide &rs, uint32_t num_limbs) {
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t *r = rs.limbs;
    return add_sub_limbs_device<true, true>(a, b, r, num_limbs);
}

// a method that reduces modulo some prime number (currently - bls12-377 scalar field prime)
static __device__ __forceinline__ bigint reduce(const bigint_wide& xs) {
    bigint xs_hi = get_higher_with_slack(xs); // xy << slack_bits
    bigint_wide l = {};
    multiply_msb_raw_device(xs_hi, get_m(), l);      // MSB mult
    bigint l_hi = get_higher(l);
    bigint_wide r_wide = {};
    bigint xs_lo = get_lower(xs);
    multiply_and_add_lsb_raw_device(l_hi, get_neg_modulus(), xs_lo, r_wide); // LSB mult
    bigint_wide r_wide_reduced = {};
    // uint32_t carry = sub_limbs_partial_device(r_wide, get_two_modulus_wide(), r_wide_reduced, TLC);
    // if (carry == 0) // continue to reduce
    //     r_wide = r_wide_reduced;
    uint32_t carry = sub_limbs_partial_device(r_wide, get_modulus_wide(), r_wide_reduced, TLC);
    if (carry == 0) // continue to reduce
        r_wide = r_wide_reduced;

    bigint r = get_lower(r_wide);
    return r;
}


// The kernel that does element-wise multiplication of arrays in1 and in2 N times
template <int N>
__global__ void multVectorsKernel(bigint *in1, const bigint *in2, bigint *out, size_t n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
    {
        bigint i1 = in1[tid];
        const bigint i2 = in2[tid];
        bigint_wide o = {0};
        for (int i = 0; i < N - 1; i++) {
            multiply_raw_device(i1, i2, o);
            i1 = reduce(o);
        }
        multiply_raw_device(i1, i2, o);
        out[tid] = reduce(o);
    }
}

template <int N>
int mult_vectors(bigint in1[], const bigint in2[], bigint *out, size_t n)
{
    // Set the grid and block dimensions
    int threads_per_block = 128;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block + 1;

    multVectorsKernel<N><<<num_blocks, threads_per_block>>>(in1, in2, out, n);

    return 0;
}


extern "C"
int multiply_test(bigint in1[], const bigint in2[], bigint *out, size_t n)
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
int multiply_bench(bigint in1[], const bigint in2[], bigint *out, size_t n)
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
