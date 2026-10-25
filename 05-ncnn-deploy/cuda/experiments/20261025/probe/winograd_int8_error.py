#!/usr/bin/env python3
"""H-084: int8 Winograd F(2x2,3x3) error simulation on the real layer shapes.

Compares the existing dp4a direct conv (exact int32 accumulation) against an
int8 Winograd pipeline that quantizes the transformed input and weights to int8
with per-position scales, which is what a dp4a implementation would have to do.
"""
import numpy as np

rng = np.random.default_rng(20261025)

BT = np.array([[1, 0, -1, 0],
               [0, 1, 1, 0],
               [0, -1, 1, 0],
               [0, 1, 0, -1]], dtype=np.float64)
G = np.array([[1.0, 0.0, 0.0],
              [0.5, 0.5, 0.5],
              [0.5, -0.5, 0.5],
              [0.0, 0.0, 1.0]])
AT = np.array([[1, 1, 1, 0],
               [0, 1, -1, -1]], dtype=np.float64)


def q8(v, scale):
    return np.clip(np.round(v / scale), -127, 127)


def run(C, tiles=200):
    errs = []
    clip_rates = []
    for _ in range(tiles):
        d = rng.integers(-127, 128, size=(C, 4, 4)).astype(np.float64)
        g = rng.integers(-127, 128, size=(C, 3, 3)).astype(np.float64)

        # direct dp4a result for the 2x2 output block
        direct = np.zeros((2, 2), dtype=np.float64)
        for oy in range(2):
            for ox in range(2):
                direct[oy, ox] = np.sum([np.dot(d[c, oy:oy + 3, ox:ox + 3].ravel(),
                                                g[c].ravel()) for c in range(C)])

        # Winograd: U = G g G^T, V = B^T d B, elementwise over channels
        U = np.einsum("ij,cjk,lk->cil", G, g, G)      # [C,4,4]
        V = np.einsum("ij,cjk,lk->cil", BT, d, BT)    # [C,4,4]

        su = np.maximum(np.abs(U).reshape(-1, 16).max(axis=0) / 127.0, 1e-6)
        sv = np.maximum(np.abs(V).reshape(-1, 16).max(axis=0) / 127.0, 1e-6)
        Uq = np.round(U / su.reshape(1, 4, 4))
        Vq = np.round(V / sv.reshape(1, 4, 4))
        clip_rates.append(np.mean((np.abs(Uq) > 127) | (np.abs(Vq) > 127)))
        Uq = np.clip(Uq, -127, 127)
        Vq = np.clip(Vq, -127, 127)

        M = np.einsum("cil,cil->il", Uq, Vq) * (su * sv).reshape(4, 4)
        Y = AT @ M @ AT.T
        errs.append(np.abs(Y - direct).max() / (np.abs(direct).max() + 1e-9))

    return float(np.mean(errs)), float(np.max(errs)), float(np.mean(clip_rates))


print("C    mean_rel_max_err  max_rel_max_err  clip_rate")
for C in (64, 128, 256, 512):
    m, mx, cr = run(C)
    print("%-4d %-16.4f %-16.4f %.4f" % (C, m, mx, cr))
