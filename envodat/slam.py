"""Trajectory loading and SLAM evaluation (ATE, RPE, scale drift; EnvoDat Eqs. 1-3)."""

import copy
from pathlib import Path


def _np():
    import numpy as np
    return np


def get_traj_np(path):
    """Load a trajectory as an ``[N, 8]`` array ``t, x, y, z, qx, qy, qz, qw``.

    Accepts the EnvoDat ground-truth CSV (``gt-poses-*.csv``, comma separated, optional
    header) and TUM files (space separated).
    """
    np = _np()
    rows = []
    for line in Path(path).read_text().splitlines():
        v = line.replace(",", " ").split()
        if len(v) < 8 or line.lstrip().startswith("#"):
            continue
        try:
            rows.append([float(x) for x in v[:8]])
        except ValueError:
            continue  # header line
    return np.asarray(rows, dtype=float)


def to_tum(path, out=None):
    """Convert an EnvoDat ground-truth CSV to a TUM trajectory file and return its path."""
    traj = get_traj_np(path)
    out = Path(out) if out else Path(path).with_suffix(".tum")
    _np().savetxt(out, traj, fmt="%.9f")
    return out


def _evo():
    try:
        from evo.core import metrics, sync
        from evo.core.trajectory import PoseTrajectory3D
    except ImportError as exc:
        raise ImportError("Install evo:  pip install 'envodat[slam]'") from exc
    return metrics, sync, PoseTrajectory3D


def _as_evo(traj):
    _, _, PoseTrajectory3D = _evo()
    np = _np()
    a = get_traj_np(traj) if isinstance(traj, (str, Path)) else np.asarray(traj, dtype=float)
    # EnvoDat/TUM quaternions are x, y, z, w; evo expects w, x, y, z.
    return PoseTrajectory3D(positions_xyz=a[:, 1:4], orientations_quat_wxyz=a[:, [7, 4, 5, 6]],
                            timestamps=a[:, 0])


def _stat(metric, data):
    metrics = _evo()[0]
    metric.process_data(data)
    return (round(metric.get_statistic(metrics.StatisticsType.mean), 2),
            round(metric.get_statistic(metrics.StatisticsType.std), 2))


def evaluate_traj(est_traj, gt_traj, align="se3", delta=1, max_diff=0.02, sd_form="ratio"):
    """Evaluate an estimated trajectory against ground truth, as in the EnvoDat benchmark.

    Args:
        est_traj: Estimated trajectory: path (TUM or EnvoDat CSV) or ``[N, 8]`` array.
        gt_traj: Ground truth: path to ``gt-poses-*.csv`` / TUM file, or ``[N, 8]`` array.
        align: ``"none"``, ``"se3"`` or ``"sim3"`` alignment for ATE/RPE.
        delta: Frame offset for RPE and scale drift.
        max_diff: Maximum timestamp difference for association [s].
        sd_form: Scale drift as ``"ratio"`` (optimum 1) or ``"abs"`` (optimum 0).

    Returns:
        Dict with ``ate``, ``rpe`` and ``sd``, each a ``(mean, std)`` tuple.
    """
    np = _np()
    metrics, sync, _ = _evo()
    gt, est = sync.associate_trajectories(_as_evo(gt_traj), _as_evo(est_traj), max_diff=max_diff)
    est_a = copy.deepcopy(est)
    if align in ("se3", "sim3"):
        est_a.align(gt, correct_scale=(align == "sim3"))
    ate = _stat(metrics.APE(metrics.PoseRelation.translation_part), (gt, est_a))
    rpe = _stat(metrics.RPE(metrics.PoseRelation.translation_part, delta=delta,
                            delta_unit=metrics.Unit.frames, all_pairs=False), (gt, est_a))
    e, g = est.positions_xyz, gt.positions_xyz
    n = min(len(e), len(g))
    est_seg = np.linalg.norm(e[delta:n] - e[:n - delta], axis=1)
    gt_seg = np.linalg.norm(g[delta:n] - g[:n - delta], axis=1)
    valid = gt_seg > 1e-6
    sd = None
    if np.any(valid):
        r = est_seg[valid] / gt_seg[valid]
        r = r if sd_form == "ratio" else np.abs(r - 1.0)
        sd = (round(float(np.mean(r)), 2), round(float(np.std(r)), 2))
    return {"ate": ate, "rpe": rpe, "sd": sd}
