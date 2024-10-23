import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose： (20,4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs：指定参数，可能包含kp,kd
    输出： global_torque: (20,3)的numpy数组，表示每个关节的全局坐标下的目标力矩，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    parent_index = physics_info.parent_index  # len(parent_index) =len(joint_name) = 20
    joint_name = physics_info.joint_name
    kp = np.zeros(len(joint_name))
    kd = np.zeros(len(joint_name))

    kp = kargs.get('kp', 500)  # 如果没有传入kp，默认为500
    kd = kargs.get('kd', 20)  # 如果没有传入kd，默认为20

    kp = np.ones(len(joint_name)) * kp
    kd = np.ones(len(joint_name)) * kd

    # 一组效果不错的kp和kd值

    depth2parent = np.zeros(len(parent_index))
    for i in range(len(parent_index)):
        j = i
        cnt = 0
        while parent_index[j] != -1:
            j = parent_index[j]
            cnt += 1
        depth2parent[i] = cnt

    kp_ref = np.array([600, 800, 800, 800, 600, 400, 400])
    kd_ref = np.array([100, 30, 15, 10, 8, 5, 5]) * 1.1
    for i in range(len(joint_name)):
        kp[i] = kp_ref[int(depth2parent[i])]
        kd[i] = kd_ref[int(depth2parent[i])]

    # 注意关节没有自己的朝向和角速度，这里用子body的朝向和角速度表示此时关节的信息
    joint_orientation = physics_info.get_joint_orientation()
    # print(physics_info.get_root_pos_and_vel())
    parent_index = physics_info.parent_index
    joint_avel = physics_info.get_body_angular_velocity()

    global_torque = np.zeros((20, 3))

    for i in range(0, len(joint_orientation)):  # 跳过根节点
        if i != physics_info.root_idx:
            parent_orientation = R.from_quat(joint_orientation[parent_index[i]])
        else:
            # set parent_orientation to identity
            parent_orientation = R.from_quat([0, 0, 0, 1])
        # 当前关节朝向（四元数）
        current_orientation = parent_orientation.inv() * R.from_quat(joint_orientation[i])
        # 目标关节朝向（四元数）
        target_orientation = R.from_quat(pose[i])

        # 计算当前与目标的旋转误差（四元数）
        error_orientation = target_orientation * current_orientation.inv()
        # 将旋转误差转换为旋转矢量
        error_angle = error_orientation.as_rotvec()

        # 计算PD控制力矩
        torque = parent_orientation.apply(error_angle) * kp[i] + kd[i] * (-joint_avel[i])

        global_torque[i] += torque

    # 对力矩进行剪裁以避免过大的力矩
    max_torque = 200  # 根据需要调整最大力矩值
    global_torque = np.clip(global_torque, -max_torque, max_torque)

    return global_torque

def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力
          global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
    '''
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 3000)  # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 200)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    global_root_force = kp * (target_position - root_position) - kd * root_velocity
    global_root_torque = global_torque[0]
    return global_root_force, global_root_torque, global_torque

def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    其余同上
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均
        为了仿真稳定最好不要在Toe关节上加额外力矩
    '''
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    
    joint_positions = physics_info.get_joint_translation()
    # 适当前移
    tar_pos = tar_pos * 0.8 + joint_positions[9] * 0.1 + joint_positions[10] * 0.1

    torque = np.zeros((20,3))
    return torque

