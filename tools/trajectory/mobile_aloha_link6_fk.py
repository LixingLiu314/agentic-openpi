#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    return _rz(y) @ _ry(p) @ _rx(r)


def _T(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


# From AgileX mobile_aloha_sim v2.0.0 aloha_tracer2_dabai_dark.urdf
# Left and right joint chains share the same relative transforms.
_LINK6_JOINTS = [
    ((0.0, 0.0, 0.123), (0.0, 0.0, -1.5708)),
    ((0.0, 0.0, 0.0), (1.5708, -0.034907, -1.5708)),
    ((0.28358, 0.028726, 0.0), (0.0, 0.0, 0.06604341)),
    ((-0.24221, 0.068514, 0.0), (-1.5708, 0.0, 1.3826)),
    ((0.0, 0.0, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.091, 0.0014165), (-1.5708, -1.5708, 0.0)),
]

_LINK7_ZERO_T = _T((0.0, 0.0, 0.13503), (1.5708, 0.0, 1.5708))


def fk_link6(joints_rad) -> np.ndarray:
    q = np.asarray(joints_rad, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    for idx, (xyz, rpy) in enumerate(_LINK6_JOINTS):
        T = T @ _T(xyz, rpy)
        T_joint = np.eye(4, dtype=np.float64)
        T_joint[:3, :3] = _rz(float(q[idx]))
        T = T @ T_joint
    return T


def fk_gripper_center(joints_rad) -> np.ndarray:
    # Approximates the gripper centerline origin from the mobile_aloha URDF by
    # applying the zero-opening joint7 fixed transform on top of link6.
    return fk_link6(joints_rad) @ _LINK7_ZERO_T


def fk_full(joints_rad) -> np.ndarray:
    # For wrist-camera calibration we want the link6 frame because the wrist
    # camera mount in the URDF is attached to link6, not the gripper tool.
    return fk_link6(joints_rad)
