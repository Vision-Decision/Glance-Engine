// Adapted from StargazerX0/SpecPilot (Apache-2.0), commit
// 1058dc29481150e66ae11b8e8cf55ed60caf254a, PRO6000 D2 specpilot_d2_w12.py.
// Changes: standalone PyTorch binding, exact linear specializations only, direct
// driver API and Glance-selected stream-K geometry. See adjacent LICENSE.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#define NCW 8
#define RB 128
#define WREC 12288

__device__ __forceinline__ void mbar_init(uint32_t a, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n W12W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W12W_%=;\n}\n"
               ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar), "l"(0x12F0000000000000ull) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}
__device__ __forceinline__ float ldg_early(const float* p) {
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void cons_bar(int n) { asm volatile("bar.sync 1, %0;\n" ::"r"(n) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&b)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&b)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// = lib w12_nest decode_n<0>: Q words (2 uint4: one byte per weight [c0 m6 m5 m4 | s c3 c2 c1]) + B words (1 uint4:
// [m3..m0] nibbles; word j low nibbles -> Q word 2j, high -> 2j+1) -> A fragments of 4 k16 steps
__device__ __forceinline__ void decode_n(const uint4 cq0, const uint4 cq1, const uint4 cb, uint32_t (&A)[4][4]) {
  const uint32_t Q[8] = {cq0.x, cq0.y, cq0.z, cq0.w, cq1.x, cq1.y, cq1.z, cq1.w};
  const uint32_t B[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c8 = 0; c8 < 2; ++c8) {
      const uint32_t q = Q[2 * j + c8];
      const uint32_t b = c8 ? (B[j] >> 4) & 0x0F0F0F0Fu : B[j] & 0x0F0F0F0Fu;
      const uint32_t lo = (q & 0xF0F0F0F0u) | b;
      const uint32_t hi = ((((q << 4) & 0x80808080u) | (q & 0x07070707u))) | 0x38383838u;
      A[j][2 * c8] = __byte_perm(lo, hi, 0x5140);
      A[j][2 * c8 + 1] = __byte_perm(lo, hi, 0x7362);
    }
  }
}

// Q view (d4): the 8-bit nested prefix only, low mantissa nibble = midpoint 1000
__device__ __forceinline__ void decode_q(const uint4 cq0, const uint4 cq1, uint32_t (&A)[4][4]) {
  const uint32_t Q[8] = {cq0.x, cq0.y, cq0.z, cq0.w, cq1.x, cq1.y, cq1.z, cq1.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c8 = 0; c8 < 2; ++c8) {
      const uint32_t q = Q[2 * j + c8];
      const uint32_t lo = (q & 0xF0F0F0F0u) | 0x08080808u;
      const uint32_t hi = ((((q << 4) & 0x80808080u) | (q & 0x07070707u))) | 0x38383838u;
      A[j][2 * c8] = __byte_perm(lo, hi, 0x5140);
      A[j][2 * c8 + 1] = __byte_perm(lo, hi, 0x7362);
    }
  }
}

// ABL ablation bits (benchmarks only): 1 no decode (raw Q words as A), 2 no mma/ldmatrix, 4 no x (no TMA, B = const),
// 8 no escapes, 16 consumers skip the corr wait/add (escape warps still compute), 32 escape warps signal without
// computing. ABL = 0 is the production kernel. NCB: 1 or 2 corr buffers (2 = escape warps run one segment ahead).
template <int NT, int NS, int ABL, int NCB, int QV>
__global__ void __launch_bounds__(NCW * 32 + 96, 1) w12m_kernel(
    const __grid_constant__ CUtensorMap xmap, const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
    const unsigned char* __restrict__ planes, int64_t tstride, int S, int U,
    const float* __restrict__ scale, const int* __restrict__ esc_rp, const int* __restrict__ esc_col,
    const float* __restrict__ esc_val, __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
    float* __restrict__ ws, int maxseg, int* __restrict__ counters, int silu, int nout, int esc_seg, int trig, int rot) {
  constexpr int NQ = NT / 8;
  constexpr int XB = NT * 128;
  constexpr int RBY = QV ? 8192 : WREC;  // Q pass streams only the Q region (first 8 KB) of each 12 KB record
  constexpr int SB = RBY + XB;  // multiple of 1024: x stages stay 1024-B aligned (128B swizzle atom)
  constexpr int CB = RB * NT;
  constexpr uint32_t TXB = (ABL & 4) ? RBY : SB;
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS + 2 * NCB];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  float* corrbuf = reinterpret_cast<float*>(smem_al + NS * SB);
  float* xchg = corrbuf + NCB * CB;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS;
  const uint32_t cfull0 = full0 + 16 * NS, cempty0 = cfull0 + 8 * NCB;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW); }
    for (int b = 0; b < NCB; ++b) { mbar_init(cfull0 + 8 * b, 64); mbar_init(cempty0 + 8 * b, NCW * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (trig) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (u0 >= u1) return;
  if (warp == NCW) {  // producer
    if (lane == 0) {
      int tile = 0, sbeg = 0, slen = 1, srot = 0;
      auto walk = [&](int u) -> int {
        if (u == u0 || u - tile * S == S) {
          tile = u / S;
          sbeg = u - tile * S;
          slen = min(u1, (tile + 1) * S) - u;
          srot = rot ? (int)(((int64_t)tile * rot + p) % slen) : 0;
        }
        int s = (u - tile * S) - sbeg + srot;
        if (s >= slen) s -= slen;
        return s + sbeg;
      };
      const int npre = min(NS, u1 - u0);
      int spre[NS];
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        if (i < npre) {
          const int s = walk(u0 + i);
          spre[i] = s;
          const uint32_t fb = full0 + 8 * i;
          mbar_expect_tx(fb, TXB);
          bulk_g2s(sbase + i * SB, planes + (int64_t)tile * tstride + (int64_t)s * WREC, RBY, fb);
        }
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
      if (!(ABL & 4)) {
#pragma unroll
        for (int i = 0; i < NS; ++i)
          if (i < npre) tma2d(sbase + i * SB + RBY, &xmap, spre[i] * 64, 0, full0 + 8 * i);
      }
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const int s = walk(u);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, TXB);
        bulk_g2s(dst, planes + (int64_t)tile * tstride + (int64_t)s * WREC, RBY, fb);
        if (!(ABL & 4)) tma2d(dst + RBY, &xmap, s * 64, 0, fb);
      }
    }
    return;
  }
  if (warp > NCW) {  // 2 escape warps: exact correction of this segment's k-range into the (single) corr buffer
    if (ABL & 8) return;
    const int ew = warp - NCW - 1;
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    int j = 0;
    for (int u = u0; u < u1;) {
      const int tile = u / S, s0 = u - tile * S;
      const int uend = min(u1, (tile + 1) * S), s1 = uend - tile * S;
      if (!esc_seg && s0 != 0) { u = uend; continue; }
      const int k0 = esc_seg ? s0 * 64 : 0, k1 = esc_seg ? s1 * 64 : S * 64;
      const int b = j % NCB;
      if (!(ABL & 16) && j >= NCB) mbar_wait(cempty0 + 8 * b, ((j / NCB) - 1) & 1);
      float* cb = corrbuf + b * CB;
      for (int r = lane + 32 * ew; r < RB && !(ABL & 32); r += 64) {
        float c[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) c[m] = 0.f;
        const int n = tile * RB + r;
        if (n < N) {
          const int e0 = esc_rp[n], e1 = esc_rp[n + 1];
          for (int e = e0; e < e1; ++e) {
            const int col = esc_col[e];
            if (col < k0 || col >= k1) continue;
            const float ev = esc_val[e];
            const __nv_bfloat16* xc = x + col;
#pragma unroll
            for (int m = 0; m < NT; ++m)
              if (m < M) {
                const float xv = __bfloat162float(xc[(int64_t)m * ldx]);
                c[m] = fmaf(ev, xv, fmaf(QV ? -3.2424926757812e-05f : -3.0517578125e-05f, xv, c[m]));
              }
          }
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) cb[r * NT + ((m + r) % NT)] = c[m];
      }
      mbar_arrive(cfull0 + 8 * b);
      ++j;
      u = uend;
    }
    return;
  }
  // consumers: warp w owns weight rows [w*16, w*16+16) of the tile; tokens as NQ n-tiles of 8
  const int g = lane / 4, t = lane % 4;
  float acc[NQ * 4];
  // ldmatrix.x4 lane address: matrix mi = lane/8 (bit0: k half, bit1: n-tile of the pair), row r = lane%8
  const int lm_r = lane & 7, lm_kh = (lane >> 3) & 1, lm_nt = lane >> 4;
  auto stage = [&](int i) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (i / NS) & 1);
    const uint4* sw = reinterpret_cast<const uint4*>(smem_al + slot * SB);
    uint32_t A[4][4];
    const uint4 q0 = sw[warp * 64 + lane], q1 = sw[warp * 64 + 32 + lane];
    const uint4 bb = QV ? make_uint4(0u, 0u, 0u, 0u) : sw[512 + warp * 32 + lane];
    if (ABL & 1) {
      const uint32_t Q[12] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w, bb.x, bb.y, bb.z, bb.w};
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) A[j][r] = Q[(j * 4 + r) % 12];
    } else if (QV) {
      decode_q(q0, q1, A);
    } else {
      decode_n(q0, q1, bb, A);
    }
    const uint32_t xs = sbase + slot * SB + RBY;
    if (ABL & 2) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[(j * 4 + r) % (NQ * 4)] += __uint_as_float(A[j][r] & 0x3F003F00u);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int qp = 0; qp < NQ / 2; ++qp) {
          uint32_t b[4];
          if (ABL & 4) {
            b[0] = b[1] = b[2] = b[3] = 0x3F803F80u;
          } else {
            const int n = (2 * qp + lm_nt) * 8 + lm_r, ch = 2 * j + lm_kh;
            ldsm_x4(b, xs + n * 128 + ((ch ^ lm_r) << 4));
          }
          mma16816(acc + (2 * qp) * 4, A[j], b[0], b[1]);
          mma16816(acc + (2 * qp + 1) * 4, A[j], b[2], b[3]);
        }
        if (NQ & 1) {
          uint32_t b[2];
          if (ABL & 4) {
            b[0] = b[1] = 0x3F803F80u;
          } else {
            const int n = (NQ - 1) * 8 + lm_r, ch = 2 * j + lm_kh;
            ldsm_x2(b, xs + n * 128 + ((ch ^ lm_r) << 4));
          }
          mma16816(acc + (NQ - 1) * 4, A[j], b[0], b[1]);
        }
      }
    }
    __syncwarp();
    // the slot was read through the generic proxy (LDS, ldmatrix); the producer refills it with TMA (async proxy):
    // without this fence a warp's late reads raced the refill (~2% of back-to-back calls at NT=24 differed)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
  };
  int jc = 0;
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
    const int n0 = tile * RB + warp * 16 + g;
    const float sc0 = n0 < N ? ldg_early(scale + n0) : 1.f, sc1 = n0 + 8 < N ? ldg_early(scale + n0 + 8) : 1.f;
#pragma unroll
    for (int q = 0; q < NQ * 4; ++q) acc[q] = 0.f;
    for (int uu = u; uu < uend; ++uu) stage(uu - u0);
    if (!(ABL & 24) && (esc_seg || s0 == 0)) {
      const int b = jc % NCB;
      const float* cb = corrbuf + b * CB;
      mbar_wait(cfull0 + 8 * b, (jc / NCB) & 1);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = warp * 16 + g + 8 * h;
#pragma unroll
        for (int q = 0; q < NQ; ++q)
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            const int m = q * 8 + 2 * t + v;
            acc[q * 4 + 2 * h + v] += cb[r * NT + ((m + r) % NT)];
          }
      }
      mbar_arrive(cempty0 + 8 * b);
      ++jc;
    }
    const bool whole = (s0 == 0) && (uend == (tile + 1) * S);
    bool emit = whole;
    if (!whole) {
      const int ufirst = tile * S, ulast = ufirst + S - 1;
      const int pfirst = (int)((((int64_t)ufirst + 1) * P - 1) / U), plast = (int)((((int64_t)ulast + 1) * P - 1) / U);
      const int seg = p - pfirst, nseg = plast - pfirst + 1;
      float* wt = ws + ((int64_t)tile * maxseg) * RB * NT;
      const int rl = warp * 16 + g;
      if (seg > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NQ; ++q)
            __stcg(reinterpret_cast<float2*>(wt + ((int64_t)seg * RB + rl + 8 * h) * NT + q * 8 + 2 * t),
                   make_float2(acc[q * 4 + 2 * h], acc[q * 4 + 2 * h + 1]));
        cons_bar(NCW * 32);
        if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" ::"l"(counters + tile) : "memory");
      } else {
        if (threadIdx.x == 0) {
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(c) : "l"(counters + tile) : "memory");
          } while (c < nseg - 1);
          counters[tile] = 0;
        }
        cons_bar(NCW * 32);
        constexpr int PB = NT <= 32 ? 4 : (NT <= 48 ? 2 : 1);
        for (int s2 = 1; s2 < nseg; s2 += PB) {
          float2 v[PB][2][NQ];
#pragma unroll
          for (int jj = 0; jj < PB; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int q = 0; q < NQ; ++q)
                v[jj][h][q] = s2 + jj < nseg ? __ldcg(reinterpret_cast<const float2*>(
                                                   wt + ((int64_t)(s2 + jj) * RB + rl + 8 * h) * NT + q * 8 + 2 * t))
                                             : make_float2(0.f, 0.f);
#pragma unroll
          for (int jj = 0; jj < PB; ++jj)
            if (s2 + jj < nseg) {
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                  acc[q * 4 + 2 * h] += v[jj][h][q].x;
                  acc[q * 4 + 2 * h + 1] += v[jj][h][q].y;
                }
            }
        }
        emit = true;
      }
    }
    if (emit) {
      if (!silu) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int n = n0 + 8 * h;
          const float sc = h ? sc1 : sc0;
#pragma unroll
          for (int q = 0; q < NQ; ++q)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int m = q * 8 + 2 * t + v;
              if (m < M && n < N) {
                float sum = acc[q * 4 + 2 * h + v];
                if constexpr (ABL == 8) {
                  // Apply each sparse correction once after reducing all K segments.
                  // This releases the 128 x NT FP32 correction tile for extra TMA stages.
                  for (int e = esc_rp[n]; e < esc_rp[n + 1]; ++e) {
                    float a = __bfloat162float(x[(int64_t)m * ldx + esc_col[e]]);
                    sum = fmaf(a, esc_val[e], fmaf(a, -0x1p-15f, sum));
                  }
                }
                out[(int64_t)m * ldo + n] = __float2bfloat16(sum * sc);
              }
            }
        }
      } else {
        const int wr = (warp & 3) * 16 + g;
        if (warp >= 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < NQ; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v)
                xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v] =
                    __bfloat162float(__float2bfloat16(acc[q * 4 + 2 * h + v] * (h ? sc1 : sc0)));
        }
        cons_bar(NCW * 32);
        if (warp < 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = tile * 64 + wr + 8 * h;
#pragma unroll
            for (int q = 0; q < NQ; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v) {
                const int m = q * 8 + 2 * t + v;
                const float gg = __bfloat162float(__float2bfloat16(acc[q * 4 + 2 * h + v] * (h ? sc1 : sc0)));
                const float uu = xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v];
                if (m < M && n < nout) out[(int64_t)m * ldo + n] = __float2bfloat16(gg / (1.f + __expf(-gg)) * uu);
              }
          }
        }
        cons_bar(NCW * 32);
      }
    }
    u = uend;
  }
}

template <int NT, int NS, int ABL, int NCB, int QV>
void launch(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col,
            torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg,
            int64_t N, int64_t silu, int64_t esc_seg, int64_t pdl, int64_t trig, int64_t rot) {
  const int M = x.size(0), K = x.size(1), nout = out.size(1);
  TORCH_CHECK(M <= NT && M > 0, "W12m: bad M");
  const int T = planes.size(0), S = K / 64, U = T * S;
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};
  cuuint32_t es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstr, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "W12m tensor map encode failed ", (int)r);
  constexpr int SB = (QV ? 8192 : WREC) + NT * 128;
  const int smem = NS * SB + (ABL == 8 ? 0 : NCB * RB * NT * 4) + (silu ? 64 * NT * 4 : 0) + 1024;
  auto kern = w12m_kernel<NT, NS, ABL, NCB, QV>;
  static bool attr = false;
  if (!attr) {
    cudaError_t ea = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(ea == cudaSuccess, "W12m smem attr failed: ", cudaGetErrorString(ea));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(NCW * 32 + (ABL == 8 ? 32 : 96));
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern,
      map, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int64_t)x.stride(0), M,
      reinterpret_cast<const unsigned char*>(planes.data_ptr()), (int64_t)planes.stride(0), S, U,
      (const float*)scale.data_ptr<float>(), (const int*)rp.data_ptr<int>(), (const int*)col.data_ptr<int>(),
      (const float*)val.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int64_t)out.stride(0),
      (int)N, ws.data_ptr<float>(), (int)maxseg, cnt.data_ptr<int>(), (int)silu, nout, (int)esc_seg, (int)trig, (int)rot);
  TORCH_CHECK(e == cudaSuccess, "W12m launch failed: ", cudaGetErrorString(e));
}


void gemm(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp,
          torch::Tensor col, torch::Tensor val, torch::Tensor out, torch::Tensor ws,
          torch::Tensor cnt, int64_t blocks, int64_t maxseg, bool whole_tile_correction, bool epilogue) {
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t m = x.size(0), n = out.size(1);
  if (epilogue) {
#define EP(nt, ns) launch<nt, ns, 8, 1, 0>(x, planes, scale, rp, col, val, out, ws, cnt, blocks, maxseg, n, 0, 0, 0, 0, 37)
    if (m <= 8) EP(8, 4);
    else if (m <= 16) EP(16, 4);
    else if (m <= 24) EP(24, 4);
    else if (m <= 32) EP(32, 4);
    else if (m <= 40) EP(40, 4);
    else if (m <= 48) EP(48, 4);
    else if (m <= 64) EP(64, 4);
    else if (m <= 96) EP(96, 3);
    else EP(128, 3);
#undef EP
    return;
  }
  // NT, ring depth and correction buffers from the SM120 D2 exact path.
#define CALL(nt, ns, ncb) launch<nt, ns, 0, ncb, 0>(x, planes, scale, rp, col, val, out, ws, cnt, blocks, maxseg, n, 0, !whole_tile_correction, 0, 0, 37)
  if (m <= 8) CALL(8, 3, 2);
  else if (m <= 16) CALL(16, 3, 2);
  else if (m <= 24) CALL(24, 3, 2);
  else if (m <= 32) CALL(32, 2, 2);
  else if (m <= 40) CALL(40, 2, 2);
  else if (m <= 48) CALL(48, 2, 2);
  else if (m <= 64) CALL(64, 2, 1);
  else if (m <= 96) CALL(96, 2, 1);
  else CALL(128, 1, 1);
#undef CALL
}



// planes -> bf16 weight in the original row order (escapes are scattered back by the caller); one block = one
// (tile, 64-k chunk): decode the 128 x 64 tile into smem, then coalesced 16-B row stores
__global__ void __launch_bounds__(256) w12m_decode_kernel(const uint4* __restrict__ planes, int64_t tstride16, int nkc,
                                                          const float* __restrict__ scale, __nv_bfloat16* __restrict__ out,
                                                          int N, int K, int silu) {
  __shared__ uint32_t tl[128][33];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int kc = blockIdx.x % nkc, tile = blockIdx.x / nkc;
  const uint4* blk = planes + tile * tstride16 + (int64_t)kc * 768;
  uint32_t A[4][4];
  decode_n(blk[w * 64 + lane], blk[w * 64 + 32 + lane], blk[512 + w * 32 + lane], A);
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int rl = w * 16 + g + 8 * h;
    const uint32_t E = (__float_as_uint(scale[tile * 128 + rl]) >> 23) & 0xFF;
    const uint32_t add = ((E - 127u) << 7) & 0xFFFFu;
    const uint32_t add2 = add | (add << 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int c8 = 0; c8 < 2; ++c8) tl[rl][j * 8 + c8 * 4 + t] = __vadd2(A[j][2 * c8 + h], add2);
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int idx = threadIdx.x + 256 * q, rl = idx >> 3, c16 = idx & 7;
    const int r = tile * 128 + rl;
    int orig = r;
    if (silu) orig = rl < 64 ? tile * 64 + rl : N / 2 + tile * 64 + rl - 64;
    if (orig >= N) continue;
    const uint4 v = make_uint4(tl[rl][c16 * 4], tl[rl][c16 * 4 + 1], tl[rl][c16 * 4 + 2], tl[rl][c16 * 4 + 3]);
    *reinterpret_cast<uint4*>(out + (int64_t)orig * K + kc * 64 + c16 * 8) = v;
  }
}

void w12m_decode(torch::Tensor planes, torch::Tensor scale, torch::Tensor out, int64_t N, int64_t K, int64_t silu) {
  c10::cuda::CUDAGuard guard(planes.device());
  const int T = planes.size(0), nkc = K / 64;
  w12m_decode_kernel<<<(unsigned)(T * nkc), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(planes.data_ptr()), planes.stride(0) / 16, nkc, scale.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)N, (int)K, (int)silu);
}


// Glance split-K variant. Four independent warps stream packed MMA fragments
// directly into registers. The second launch owns the final FP32 reduction.
template <int GROUPS>
__global__ void direct_part(const uint4* planes, const __nv_bfloat16* x, float* partial,
                            int n, int k, int m, int stages_per_split) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int row16 = blockIdx.x * 4 + warp;
  if (row16 * 16 >= n) return;
  int g = lane >> 2, t = lane & 3;
  int stages = k / 64;
  float acc[GROUPS][4] = {};
  for (int s = blockIdx.y * stages_per_split; s < min(stages, (blockIdx.y + 1) * stages_per_split); ++s) {
    const uint4* base = planes + ((row16 / 8) * stages + s) * 768;
    int w = row16 % 8;
    uint32_t a[4][4];
    decode_n(base[w*64 + lane], base[w*64 + 32 + lane], base[512 + w*32 + lane], a);
#pragma unroll
    for (int q = 0; q < GROUPS; ++q) {
      const int token = q*8 + g;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int offset = s*64 + j*16 + t*2;
        uint32_t b0 = 0, b1 = 0;
        if (token < m) {
          const auto* ptr = reinterpret_cast<const uint32_t*>(x + token*k + offset);
          b0 = ptr[0]; b1 = ptr[4];
        }
        mma16816(acc[q], a[j], b0, b1);
      }
    }
  }
#pragma unroll
  for (int q = 0; q < GROUPS; ++q) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int r = row16*16 + g + 8*h;
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        int token = q*8 + 2*t + v;
        if (token < m && r < n) partial[(blockIdx.y*m+token)*n+r] = acc[q][2*h+v];
      }
    }
  }
}

__global__ void direct_reduce(const float* partial, const __nv_bfloat16* x, const float* scale,
                              const int* offsets, const int* columns, const float* corrections,
                              __nv_bfloat16* out, int n, int k, int m, int splits) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= m*n) return;
  int row = idx % n, token = idx / n;
  float sum = 0.f;
  for (int s = 0; s < splits; ++s) sum += partial[s*m*n + idx];
  for (int e = offsets[row]; e < offsets[row+1]; ++e) {
    float a = __bfloat162float(x[token*k + columns[e]]);
    sum = fmaf(a, corrections[e], fmaf(a, -0x1p-15f, sum));
  }
  out[idx] = __float2bfloat16(sum*scale[row]);
}

void direct(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp,
            torch::Tensor col, torch::Tensor val, torch::Tensor out, torch::Tensor partial, int span) {
  c10::cuda::CUDAGuard guard(x.device());
  int n = out.size(1), k = x.size(1), m = x.size(0), splits = partial.size(0);
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid((n+63)/64, splits);
#define PART(q) direct_part<q><<<grid,128,0,stream>>>(reinterpret_cast<const uint4*>(planes.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), partial.data_ptr<float>(), n,k,m,span)
  if (m <= 8) PART(1);
  else if (m <= 16) PART(2);
  else if (m <= 32) PART(4);
  else PART(8);
#undef PART
  direct_reduce<<<(m*n+255)/256,256,0,stream>>>(partial.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), scale.data_ptr<float>(), rp.data_ptr<int>(),
      col.data_ptr<int>(), val.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n,k,m,splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("direct", &direct);
  module.def("gemm", &gemm);
  module.def("decode", &w12m_decode);
}
