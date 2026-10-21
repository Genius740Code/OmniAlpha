//! Game-agnostic conv kernels: im2col + blocked SGEMM (no Meridian types here).
//!
//! The reference implementation in [`crate::nn`] walks each output pixel per
//! (output, in-channel, tap) with <= 19-wide saxpys — huge loop overhead and no
//! blocking. Here every conv becomes one GEMM
//! `C[M][N] = relu?(bias[M] + A[M][K] · B[K][N])`:
//!
//! * `M` = output channels, `K` = cin·k·k, `N` = board points.
//! * 3x3 and 5x5 convs build an im2col matrix `B` first; its rows are in
//!   exactly the weight's `(c, kh, kw)` tap order, so each output element
//!   accumulates in the same order as the reference conv (near-bit-exact).
//! * 1x1 convs skip im2col: the input plane layout `[c][n]` already *is* `B`.
//!
//! Row strides are caller-controlled (`ld >= n`): padding rows to a multiple
//! of 4 f32s (16 bytes) plus an [`Align16`] buffer makes every strip of the
//! wasm SIMD kernel provably 16-byte aligned — `v128_load`/`v128_store`
//! require it. On native the same loops auto-vectorize (SSE2 baseline).
//!
//! The kernel is M-blocked by 4 and N-blocked by 8 so the accumulators sit
//! in registers and the inner saxpy vectorizes. On
//! wasm32 built with `-C target-feature=+simd128` an explicit
//! `core::arch::wasm32` kernel is used (its `unsafe` blocks are the only
//! unsafe code in this crate); without the feature the same safe Rust loops
//! run scalar, and the crate still builds everywhere.
//!
//! Everything is std-only and allocation-free: callers own the workspace.

/// N-block width of the scalar/auto-vectorized kernel. 8 keeps the register
/// footprint at 10 SSE vectors (8 accumulator rows + 2 loaded B rows) so a
/// fully unrolled inner loop never spills.
const NB: usize = 8;
/// M-block width: 4 output rows share each loaded B row.
const MB: usize = 4;
/// N-block width of the wasm SIMD kernel (8 f32 = 2 v128 loads).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const WB: usize = 8;

/// A fixed-size f32 buffer with a 16-byte-aligned start, so the wasm SIMD
/// kernel can load and store whole `v128`s at padded row strides.
///
/// `zeroed()` needs no `unsafe`; buffers are meant to be reused, never grown.
#[repr(align(16))]
pub struct Align16<const N: usize>(#[doc(hidden)] pub [f32; N]);

impl<const N: usize> Align16<N> {
    /// A zero-filled buffer of `N` floats.
    pub const fn zeroed() -> Self {
        Self([0f32; N])
    }

    /// The floats, `&mut [f32; N]`.
    pub fn as_mut(&mut self) -> &mut [f32; N] {
        &mut self.0
    }

    /// The floats as a shared slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// The floats as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }
}

/// Builds the im2col matrix for a `B`×`B` board — the only place board size
/// enters the kernels, as a const parameter so all index arithmetic is
/// compile-time and a per-game adapter just instantiates `im2col::<BOARD>`.
///
/// `input` is `[cin][in_ld]` and `col` is `[cin·k·k][ld]`; row
/// `(c·k·k + kh·k + kw)` of `col` holds the input shifted by
/// `(kh - pad, kw - pad)` — the exact order the decoded weights address their
/// taps, so a following [`sgemm`] accumulates in the reference conv's order.
/// Border taps are zero-filled per row segment (the valid interior is ~93%,
/// so segment-wise zeroing beats clearing the whole buffer every conv). Any
/// `ld > n` padding columns are never touched and stay zero.
pub fn im2col<const B: usize>(
    input: &[f32],
    in_ld: usize,
    cin: usize,
    k: usize,
    pad: usize,
    col: &mut [f32],
    ld: usize,
) {
    let n = B * B;
    debug_assert!(k % 2 == 1 && pad == k / 2, "odd kernels, same padding");
    debug_assert!(ld >= n);
    let kk = k * k;
    for row in 0..cin * kk {
        let c = row / kk;
        let kh = (row % kk) / k;
        let kw = row % k;
        // Ranges of output h (and w) where the tap stays inside the board —
        // identical arithmetic to the reference conv.
        let h_lo = pad - kh.min(pad);
        let h_hi = (B + pad - kh).min(B);
        let w_lo = pad - kw.min(pad);
        let w_hi = (B + pad - kw).min(B);
        let dst = &mut col[row * ld..row * ld + n];
        for h in 0..h_lo {
            dst[h * B..(h + 1) * B].fill(0.0);
        }
        for h in h_lo..h_hi {
            let ih = h + kh - pad;
            let ix = w_lo + kw - pad;
            let seg = &mut dst[h * B..(h + 1) * B];
            seg[..w_lo].fill(0.0);
            seg[w_hi..].fill(0.0);
            seg[w_lo..w_hi].copy_from_slice(
                &input[c * in_ld + ih * B + ix..c * in_ld + ih * B + ix + (w_hi - w_lo)],
            );
        }
        for h in h_hi..B {
            dst[h * B..(h + 1) * B].fill(0.0);
        }
    }
}

/// `C[M][N] = relu?(bias[M] + A[M][K] · B[K][N])`, where `B` and `C` have row
/// stride `ld >= N` and `A` is `[M][K]` contiguous (conv weights, K in tap
/// order). `M` must be a multiple of 4 or the scalar fallback runs; `ld` a
/// multiple of 4 keeps the SIMD strips 16-byte aligned.
pub fn sgemm(
    m: usize,
    n: usize,
    ld: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    relu: bool,
) {
    debug_assert!(ld >= n);
    if m % MB == 0 {
        sgemm_blocked(m, n, ld, k, a, b, bias, c, relu);
    } else {
        sgemm_simple(m, n, ld, k, a, b, bias, c, relu);
    }
}

/// The blocked scalar kernel: every NB-wide strip of N x 4-wide group of M
/// keeps its accumulators in registers. All inner loops are over fixed-size
/// arrays (`&[f32; NB]`), so they fully unroll, the bounds checks vanish
/// and the accumulators stay in registers — the whole point of this kernel;
/// the inner saxpy auto-vectorizes (AVX2 with `target-cpu=native`, SSE2
/// otherwise). The last, narrow strip of N runs scalar via [`sgemm_tail`].
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn sgemm_blocked(m: usize, n: usize, ld: usize, k: usize, a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], relu: bool) {
    let full = n - n % NB;
    for nb in (0..full).step_by(NB) {
        for m0 in (0..m).step_by(MB) {
            let mut acc = [[0f32; NB]; MB];
            for kk in 0..k {
                let brow: &[f32; NB] = b[kk * ld + nb..kk * ld + nb + NB]
                    .try_into()
                    .expect("full NB strip");
                for j in 0..MB {
                    let wj = a[(m0 + j) * k + kk];
                    let ar = &mut acc[j];
                    for x in 0..NB {
                        ar[x] += wj * brow[x];
                    }
                }
            }
            for j in 0..MB {
                let vb = bias[m0 + j];
                let crow = &mut c[(m0 + j) * ld + nb..(m0 + j) * ld + nb + NB];
                let arow = &acc[j];
                for x in 0..NB {
                    let val = vb + arow[x];
                    crow[x] = if relu && val < 0.0 { 0.0 } else { val };
                }
            }
        }
    }
    if full < n {
        sgemm_tail(m, full, n, ld, k, a, b, bias, c, relu);
    }
}

/// wasm32 + simd128: the same blocked GEMM with an explicit v128 kernel
/// (4 output rows x 8 columns = 8 accumulator registers). Compiles only when
/// the build passes `-C target-feature=+simd128`; plain wasm builds use the
/// scalar kernel above and stay correct.
///
/// The `unsafe` blocks below are the only unsafe code in the crate. Safety:
/// callers pass [`Align16`] buffers, and with `ld % 4 == 0` (checked) every
/// accessed address `base + kk·ld + nb + t·4` is a multiple of 16 bytes —
/// the alignment `v128_load`/`v128_store` require — and every pointer is
/// derived from an in-bounds slice, so it is valid for the accessed width.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn sgemm_blocked(m: usize, n: usize, ld: usize, k: usize, a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], relu: bool) {
    use core::arch::wasm32::{f32x4_add, f32x4_max, f32x4_mul, f32x4_splat, v128, v128_load, v128_store};
    debug_assert_eq!(b.as_ptr() as usize % 16, 0, "B must be Align16-backed");
    debug_assert_eq!(c.as_ptr() as usize % 16, 0, "C must be Align16-backed");
    debug_assert_eq!(ld % 4, 0, "row stride must be 16-byte aligned");
    let zero = f32x4_splat(0.0);
    // Full WB-wide strips first; the tail (n % 8) runs scalar below.
    let full = n - n % WB;
    for nb in (0..full).step_by(WB) {
        for m0 in (0..m).step_by(MB) {
            let mut acc = [[v128::from(zero); 2]; MB];
            for kk in 0..k {
                let brow = &b[kk * ld + nb..kk * ld + nb + WB];
                // SAFETY: 16-byte aligned (see comment above) and in bounds;
                // each v128 covers 4 f32s, so WB = 8 is exactly two loads.
                let bp = brow.as_ptr().cast::<v128>();
                let br = [unsafe { v128_load(bp) }, unsafe { v128_load(bp.add(1)) }];
                for j in 0..MB {
                    let wj = f32x4_splat(a[(m0 + j) * k + kk]);
                    for t in 0..2 {
                        acc[j][t] = f32x4_add(acc[j][t], f32x4_mul(wj, br[t]));
                    }
                }
            }
            for j in 0..MB {
                let vb = f32x4_splat(bias[m0 + j]);
                let crow = &mut c[(m0 + j) * ld + nb..(m0 + j) * ld + nb + WB];
                // SAFETY: 16-byte aligned and in bounds, as above.
                let cp = crow.as_mut_ptr().cast::<v128>();
                for t in 0..2 {
                    let v = f32x4_add(acc[j][t], vb);
                    unsafe { v128_store(cp.add(t), if relu { f32x4_max(v, zero) } else { v }) };
                }
            }
        }
    }
    if full < n {
        sgemm_tail(m, full, n, ld, k, a, b, bias, c, relu);
    }
}

/// The (single, at most 7-wide) tail strip of a blocked GEMM, computed
/// scalar — a negligible fraction of any realistic N.
fn sgemm_tail(m: usize, nb: usize, n: usize, ld: usize, k: usize, a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], relu: bool) {
    for mi in 0..m {
        for x in nb..n {
            let mut acc = bias[mi];
            for kk in 0..k {
                acc += a[mi * k + kk] * b[kk * ld + x];
            }
            if relu && acc < 0.0 {
                acc = 0.0;
            }
            c[mi * ld + x] = acc;
        }
    }
}

/// Scalar fallback for M not divisible by 4 (never hit by this net's shapes;
/// keeps the kernel honest if the architecture changes).
fn sgemm_simple(m: usize, n: usize, ld: usize, k: usize, a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], relu: bool) {
    for mi in 0..m {
        let arow = &a[mi * k..(mi + 1) * k];
        let crow = &mut c[mi * ld..mi * ld + n];
        for (x, slot) in crow.iter_mut().enumerate() {
            let mut acc = bias[mi];
            for kk in 0..k {
                acc += arow[kk] * b[kk * ld + x];
            }
            *slot = if relu && acc < 0.0 { 0.0 } else { acc };
        }
    }
}

/// Fused value head: `pooled[M] = mean_n relu(bias[M] + A[M][K] · B[K][N])`
/// in a single pass — the 1x1 conv, its ReLU and the global average pool
/// never materialize the M x N plane. `B` has row stride `ld >= n`; the mean
/// is over the `n` valid columns only (any stride padding is ignored).
pub fn sgemm_relu_pool(m: usize, n: usize, ld: usize, k: usize, a: &[f32], b: &[f32], bias: &[f32], pooled: &mut [f32]) {
    debug_assert_eq!(pooled.len(), m);
    debug_assert!(ld >= n);
    pooled.fill(0.0);
    if m % MB == 0 {
        let full = n - n % NB;
        for nb in (0..full).step_by(NB) {
            for m0 in (0..m).step_by(MB) {
                let mut acc = [[0f32; NB]; MB];
                for kk in 0..k {
                    let brow: &[f32; NB] = b[kk * ld + nb..kk * ld + nb + NB]
                        .try_into()
                        .expect("full NB strip");
                    for j in 0..MB {
                        let wj = a[(m0 + j) * k + kk];
                        let ar = &mut acc[j];
                        for x in 0..NB {
                            ar[x] += wj * brow[x];
                        }
                    }
                }
                for j in 0..MB {
                    let vb = bias[m0 + j];
                    for x in 0..NB {
                        let v = (vb + acc[j][x]).max(0.0);
                        pooled[m0 + j] += v;
                    }
                }
            }
        }
        for m0 in (0..m).step_by(MB) {
            for j in 0..MB {
                for x in full..n {
                    let mut v = bias[m0 + j];
                    for kk in 0..k {
                        v += a[(m0 + j) * k + kk] * b[kk * ld + x];
                    }
                    pooled[m0 + j] += v.max(0.0);
                }
            }
        }
    } else {
        for mi in 0..m {
            let arow = &a[mi * k..(mi + 1) * k];
            let mut sum = 0f32;
            for x in 0..n {
                let mut v = bias[mi];
                for kk in 0..k {
                    v += arow[kk] * b[kk * ld + x];
                }
                sum += v.max(0.0);
            }
            pooled[mi] = sum;
        }
    }
    let inv = 1.0 / n as f32;
    for p in pooled.iter_mut() {
        *p *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocked kernel must match the naive triple loop (both accumulate
    /// k-ascending per element, so this is exact up to contraction), for
    /// both a tight and a padded row stride.
    #[test]
    fn sgemm_matches_naive() {
        for &(m, n, ld) in &[(8usize, 37usize, 40usize), (4, 32, 32), (12, 61, 64)] {
            let k = 11usize;
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37 % 1.0) - 0.5).collect();
            let b: Vec<f32> = (0..k * ld).map(|i| ((i * 7) as f32 * 0.11 % 1.0) - 0.5).collect();
            let bias: Vec<f32> = (0..m).map(|i| i as f32 * 0.1).collect();
            let mut c1 = vec![0f32; m * ld];
            sgemm(m, n, ld, k, &a, &b, &bias, &mut c1, true);
            let mut c2 = vec![0f32; m * ld];
            sgemm_simple(m, n, ld, k, &a, &b, &bias, &mut c2, true);
            let max = c1
                .iter()
                .zip(&c2)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            assert!(max < 1e-5, "blocked vs naive diff {max} (m={m} n={n} ld={ld})");
        }
    }

    /// im2col + SGEMM must reproduce a direct padded conv on random data,
    /// independently of the fixture parity run — on a small, non-Meridian
    /// board with a padded stride, proving the kernels are game-agnostic.
    #[test]
    fn im2col_conv_matches_direct() {
        const B: usize = 7; // 49 points, deliberately not 19x19
        let n = B * B;
        let ld = n.div_ceil(4) * 4; // 52: padded like the production stride
        for &(cin, cout, k, pad) in &[(3, 4, 3, 1), (2, 4, 5, 2)] {
            let input: Vec<f32> = (0..cin * n).map(|i| ((i * 13) as f32 * 0.07) % 1.0 - 0.5).collect();
            let w: Vec<f32> = (0..cout * cin * k * k).map(|i| ((i * 29) as f32 * 0.03) % 1.0 - 0.5).collect();
            let b: Vec<f32> = vec![0.1, -0.2, 0.3, 0.0];
            let mut col = Align16::<{ 3 * 25 * 52 }>::zeroed();
            im2col::<B>(&input, n, cin, k, pad, col.as_mut(), ld);
            let mut c1 = vec![0f32; cout * ld];
            sgemm(cout, n, ld, cin * k * k, &w, col.as_mut(), &b, &mut c1, false);

            let mut c2 = vec![0f32; cout * ld];
            for o in 0..cout {
                for p in 0..n {
                    c2[o * ld + p] = b[o];
                }
                for c in 0..cin {
                    for kh in 0..k {
                        let h_lo = (pad as isize - kh as isize).max(0) as usize;
                        let h_hi = (B + pad - kh).min(B);
                        for h in h_lo..h_hi {
                            let ih = h + kh - pad;
                            for kw in 0..k {
                                let w_lo = (pad as isize - kw as isize).max(0) as usize;
                                let w_hi = (B + pad - kw).min(B);
                                let ws = w[(o * cin + c) * k * k + kh * k + kw];
                                for w0 in w_lo..w_hi {
                                    c2[o * ld + h * B + w0] +=
                                        ws * input[c * n + ih * B + w0 + kw - pad];
                                }
                            }
                        }
                    }
                }
            }
            let max = c1
                .iter()
                .zip(&c2)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            assert!(max < 1e-5, "im2col vs direct diff {max} (k={k})");
        }
    }

    /// The fused pool must equal conv + ReLU + mean over valid columns,
    /// ignoring any stride padding.
    #[test]
    fn relu_pool_matches_conv() {
        let (m, n, ld, k) = (8usize, 50usize, 52usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.31 % 1.0) - 0.5).collect();
        let b: Vec<f32> = (0..k * ld).map(|i| ((i * 3) as f32 * 0.17 % 1.0) - 0.5).collect();
        let bias: Vec<f32> = vec![0.2; m];
        let mut pooled = vec![0f32; m];
        sgemm_relu_pool(m, n, ld, k, &a, &b, &bias, &mut pooled);
        for mi in 0..m {
            let mut sum = 0f32;
            for x in 0..n {
                let mut v = bias[mi];
                for kk in 0..k {
                    v += a[mi * k + kk] * b[kk * ld + x];
                }
                sum += v.max(0.0);
            }
            assert!((pooled[mi] - sum / n as f32).abs() < 1e-5, "pool row {mi}");
        }
    }

    /// Align16 must actually be 16-byte aligned (the wasm SIMD contract).
    #[test]
    fn align16_is_aligned() {
        let mut buf = Align16::<{ 16 * 16 }>::zeroed();
        assert_eq!(buf.as_mut().as_ptr() as usize % 16, 0);
        assert_eq!(buf.as_mut().len(), 256);
    }
}
