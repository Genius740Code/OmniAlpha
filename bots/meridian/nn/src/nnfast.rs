//! The production forward pass: same network, same contract, fast kernels.
//!
//! Mirrors [`crate::nn::forward`]'s dataflow exactly (stem -> 4 residual
//! blocks -> value head / policy head) but every conv is an im2col + SGEMM,
//! and the value head's 1x1 conv, ReLU and global average pool are fused
//! into one pass. The scratch buffers are fixed-size, 16-byte-aligned
//! thread-local planes with rows padded to [`crate::nn::PSTRIDE`] — zero
//! allocation, ever, and every wasm SIMD access provably aligned.

use std::cell::RefCell;

use meridian_engine::NUM_MOVES;

use crate::convfast::{Align16, im2col, sgemm, sgemm_relu_pool};
use crate::nn::{BOARD, CH, IN_PLANES, PCH, POINTS, PSTRIDE, VCH, net};

/// Plane sizes, derived from the per-game config block only.
const TRUNK: usize = CH * PSTRIDE;
/// Largest im2col in this architecture: the 5x5 stem over all input planes.
const COL: usize = IN_PLANES * 25 * PSTRIDE;
const POLICY: usize = PCH * PSTRIDE;

/// Everything one forward pass needs: reused, never grown, never allocated.
struct Workspace {
    t0: Align16<TRUNK>,
    t1: Align16<TRUNK>,
    t2: Align16<TRUNK>,
    /// im2col matrix, sized for the biggest conv (the 5x5 stem).
    col: Align16<COL>,
    p1: Align16<POLICY>,
    p2: Align16<POLICY>,
}

impl Workspace {
    const fn new() -> Self {
        Self {
            t0: Align16::zeroed(),
            t1: Align16::zeroed(),
            t2: Align16::zeroed(),
            col: Align16::zeroed(),
            p1: Align16::zeroed(),
            p2: Align16::zeroed(),
        }
    }
}

thread_local! {
    static WS: RefCell<Workspace> = const { RefCell::new(Workspace::new()) };
}

/// The fast forward pass. Returns the value and, when `policy` is set, the
/// move logits — the same contract as the reference implementation, which
/// `nnbench --verify` checks against this one (bit-near, < 1e-5).
pub(crate) fn forward_fast(
    input: &[f32; IN_PLANES * POINTS],
    policy: bool,
) -> (f64, Option<[f32; NUM_MOVES]>) {
    debug_assert_eq!(PCH * POINTS, NUM_MOVES, "policy head must flatten to move IDs");
    let net = net();
    WS.with_borrow_mut(|ws| {
        let t0 = ws.t0.as_mut_slice();
        let t1 = ws.t1.as_mut_slice();
        let t2 = ws.t2.as_mut_slice();
        let col = ws.col.as_mut_slice();

        // Stem: 5x5 conv, stride 1, pad 2 + ReLU. The public input planes are
        // compact ([IN_PLANES][POINTS]), so their row stride is POINTS.
        im2col::<BOARD>(input, POINTS, IN_PLANES, 5, 2, col, PSTRIDE);
        sgemm(CH, POINTS, PSTRIDE, IN_PLANES * 25, &net.stem_w, col, &net.stem_b, t0, true);

        // Residual blocks: t0 = relu(t0 + conv2(relu(conv1(t0)))). The stride
        // padding columns are never written (n = POINTS below) and stay zero,
        // so the fused add may run over the whole padded plane.
        for block in &net.blocks {
            im2col::<BOARD>(t0, PSTRIDE, CH, 3, 1, col, PSTRIDE);
            sgemm(CH, POINTS, PSTRIDE, CH * 9, &block.c1_w, col, &block.c1_b, t1, true);
            im2col::<BOARD>(t1, PSTRIDE, CH, 3, 1, col, PSTRIDE);
            sgemm(CH, POINTS, PSTRIDE, CH * 9, &block.c2_w, col, &block.c2_b, &mut *t2, false);
            for (a, b) in t0.iter_mut().zip(&*t2) {
                *a = (*a + *b).max(0.0);
            }
        }

        // Value head, fused: 1x1 conv + ReLU + global average pool in one
        // pass (valid POINTS columns only), then the 16->1 linear and tanh.
        let mut pooled = [0f32; VCH];
        sgemm_relu_pool(VCH, POINTS, PSTRIDE, CH, &net.v_w, t0, &net.v_b, &mut pooled);
        let mut acc = net.v_fc_b[0];
        for c in 0..VCH {
            acc += net.v_fc_w[c] * pooled[c];
        }
        let value = f64::from(acc.tanh());

        // Policy head: 3x3 conv + ReLU, then a 1x1 conv straight on p1 (no
        // im2col for 1x1). Channel c == direction c, so the flattened output
        // is indexed by move ID; the padded plane compacts into the move
        // array at the end.
        let mut logits = None;
        if policy {
            let p1 = ws.p1.as_mut_slice();
            let p2 = ws.p2.as_mut_slice();
            im2col::<BOARD>(t0, PSTRIDE, CH, 3, 1, col, PSTRIDE);
            sgemm(PCH, POINTS, PSTRIDE, CH * 9, &net.p1_w, col, &net.p1_b, p1, true);
            sgemm(PCH, POINTS, PSTRIDE, PCH, &net.p2_w, p1, &net.p2_b, p2, false);
            let mut out = [0f32; NUM_MOVES];
            for c in 0..PCH {
                out[c * POINTS..(c + 1) * POINTS].copy_from_slice(&p2[c * PSTRIDE..c * PSTRIDE + POINTS]);
            }
            logits = Some(out);
        }
        (value, logits)
    })
}
