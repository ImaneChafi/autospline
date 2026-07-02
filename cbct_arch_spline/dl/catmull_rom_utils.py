"""
Catmull-Rom spline evaluation -- matches the curve type actually used
to draw the manual splines in 3D Slicer (an INTERPOLATING piecewise
cubic spline that passes exactly through every annotated point), unlike
a single global-degree Bezier least-squares fit, which can oscillate
between sparse/sharp-turning points (Runge's-phenomenon-style
artifact) -- this is what caused the zigzag artifacts.
"""
import numpy as np


def _catmull_rom_segment(p0, p1, p2, p3, t):
    """Standard uniform Catmull-Rom basis, t in [0,1], interpolates p1->p2."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def catmull_rom_eval(points, n_samples=300):
    """points: (N,2) ordered annotated points (the RAW points, no fitting).
    Returns a dense (n_samples,2) curve that passes exactly through every
    input point -- pads the ends by duplicating the first/last point
    (standard boundary handling) so endpoint segments are well-defined."""
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if N < 2:
        return np.repeat(points, n_samples, axis=0)
    padded = np.vstack([points[0], points, points[-1]])  # N+2 points

    n_segments = N - 1
    samples_per_seg = max(2, n_samples // n_segments)
    curve = []
    for i in range(n_segments):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        ts = np.linspace(0, 1, samples_per_seg,
                         endpoint=(i == n_segments - 1))
        for t in ts:
            curve.append(_catmull_rom_segment(p0, p1, p2, p3, t))
    return np.array(curve)


def catmull_rom_basis_matrix(n_control, n_samples):
    """Precompute a fixed (n_samples, n_control) matrix M such that
    dense_curve = M @ control_points reproduces catmull_rom_eval() exactly
    for a FIXED n_control (architecture hyperparameter, e.g. 24). Each
    output row has at most 4 nonzero entries (the 4 control points its
    segment depends on) -- this lets curve evaluation be a single matrix
    multiply, which is what makes it differentiable/batchable in torch,
    the same trick used for the Bezier Bernstein matrix but adapted to
    Catmull-Rom's piecewise/local structure."""
    n_segments = n_control - 1
    samples_per_seg = max(2, n_samples // n_segments)
    M = np.zeros((n_segments * samples_per_seg, n_control))
    row = 0
    for i in range(n_segments):
        a = max(i - 1, 0)
        b = i
        c = min(i + 1, n_control - 1)
        d = min(i + 2, n_control - 1)
        ts = np.linspace(0, 1, samples_per_seg,
                         endpoint=(i == n_segments - 1))
        for t in ts:
            t2, t3 = t * t, t * t * t
            w0 = 0.5 * (-t + 2 * t2 - t3)
            w1 = 0.5 * (2 - 5 * t2 + 3 * t3)
            w2 = 0.5 * (t + 4 * t2 - 3 * t3)
            w3 = 0.5 * (-t2 + t3)
            M[row, a] += w0
            M[row, b] += w1
            M[row, c] += w2
            M[row, d] += w3
            row += 1
    return M[:row]


def resample_by_arclength(curve, n_points):
    """Resample a dense curve to n_points evenly spaced by arc length."""
    diffs = np.diff(curve, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seglens)])
    total = cumlen[-1] if cumlen[-1] > 0 else 1.0
    target = np.linspace(0, total, n_points)
    x = np.interp(target, cumlen, curve[:, 0])
    y = np.interp(target, cumlen, curve[:, 1])
    return np.stack([x, y], axis=1)


def fit_catmull_rom_target(raw_points, n_control=24, dense_samples=300):
    """The new ground-truth representation: evaluate the TRUE Catmull-Rom
    curve through the raw annotated points (faithful to Slicer), then
    resample it to a fixed n_control points by arc length. These
    n_control points can later be fed straight back into
    catmull_rom_eval() to reconstruct a smooth curve -- no separate
    fitting step needed, unlike Bezier, and no oscillation risk since
    Catmull-Rom is local/piecewise."""
    dense_curve = catmull_rom_eval(raw_points, n_samples=dense_samples)
    return resample_by_arclength(dense_curve, n_control)
