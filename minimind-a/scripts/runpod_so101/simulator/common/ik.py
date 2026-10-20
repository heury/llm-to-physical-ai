"""수치 Inverse Kinematics (Damped Least Squares, Jacobian 기반).

목표: site(손끝 기준점)의 위치를 target_pos로, 필요하면 site의 z축 방향을 target_z로 맞추는
관절 각도를 찾는다. MuJoCo가 계산해 주는 Jacobian(mj_jacSite)을 이용해
"관절을 조금 움직이면 손끝이 어디로 가는가"를 매 반복마다 선형 근사하고,
그 반대 방향으로 관절을 조금씩 고쳐 나간다.

물리 시뮬레이션은 돌리지 않는다. 별도의 MjData 사본에서 mj_forward(운동학만 계산)를 반복한다.
"""

import mujoco
import numpy as np


def solve_ik(model, data, site_name, target_pos, target_dir=None, arm_joint_names=None,
             axis=0, max_iters=300, tol=1e-3, damping=1e-2, step=0.5):
    """관절 각도 배열(arm_joint_names 순서)과 최종 오차를 돌려준다.

    target_dir: site 좌표계의 axis 번째 축(SO-101의 gripperframe은 x축이 손가락 방향)이
                가리켜야 할 단위 벡터. None 이면 위치만 맞춘다.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm_joint_names]
    qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]

    scratch = mujoco.MjData(model)          # 실제 시뮬레이션 상태를 건드리지 않기 위한 사본
    scratch.qpos[:] = data.qpos
    target_pos = np.asarray(target_pos, dtype=float)
    if target_dir is not None:
        target_dir = np.asarray(target_dir, dtype=float)
        target_dir /= np.linalg.norm(target_dir)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err_norm = np.inf
    for _ in range(max_iters):
        mujoco.mj_forward(model, scratch)
        pos = scratch.site_xpos[site_id]
        err = target_pos - pos                                   # 위치 오차 (3)
        if target_dir is not None:
            dir_cur = scratch.site_xmat[site_id].reshape(3, 3)[:, axis]
            err_rot = np.cross(dir_cur, target_dir)              # 회전 오차 (3): 작을수록 방향이 같다
            err = np.concatenate([err, err_rot])
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = jacp[:, dof_adr] if target_dir is None else np.vstack([jacp, jacr])[:, dof_adr]
        # Damped Least Squares: dq = J^T (J J^T + λ²I)^-1 err
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(JJt.shape[0]), err)
        q = scratch.qpos[qpos_adr] + step * dq
        scratch.qpos[qpos_adr] = np.clip(q, lo, hi)              # 관절 가동 범위 안으로

    return scratch.qpos[qpos_adr].copy(), err_norm
