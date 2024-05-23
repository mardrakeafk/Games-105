import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import torch



# 求旋转矩阵
def get_rotation_matrix(a, b):
    a=a/np.linalg.norm(a)
    b=b/np.linalg.norm(b)
    # 叉乘
    n = np.cross(a, b)
    # 旋转矩阵是正交矩阵，矩阵的每一行每一列的模，都为1；并且任意两个列向量或者任意两个行向量都是正交的。
    # n=n/np.linalg.norm(n)
    # 计算夹角
    cos_theta = np.dot(a, b)
    sin_theta = np.linalg.norm(n)
    theta = np.arctan2(sin_theta, cos_theta)
    # 构造旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    rotation_matrix = np.array([[n[0]*n[0]*v+c, n[0]*n[1]*v-n[2]*s, n[0]*n[2]*v+n[1]*s],
                                 [n[0]*n[1]*v+n[2]*s, n[1]*n[1]*v+c, n[1]*n[2]*v-n[0]*s],
                                 [n[0]*n[2]*v-n[1]*s, n[1]*n[2]*v+n[0]*s, n[2]*n[2]*v+c]])
    return rotation_matrix


def inv_safe(data):
    # return R.from_quat(data).inv()
    if np.allclose(data, [1, 0, 0, 0]):
        return np.eye(3)
    else:
        return np.linalg.inv(R.from_quat(data).as_matrix())


def from_quat_safe(data):
    # return R.from_quat(data)
    if np.allclose(data, [1, 0, 0, 0]):
        return np.eye(3)
    else:
        return R.from_quat(data).as_matrix()



def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    parent_idx = meta_data.joint_parent
    name = meta_data.joint_name
    # local_rotation是用于最后计算不在链上的节点
    no_caled_orientation = copy.deepcopy(joint_orientations)
    local_rotation = [
        R.from_matrix(inv_safe(joint_orientations[parent_idx[i]]) * from_quat_safe(joint_orientations[i])).as_quat() for
        i
        in range(len(joint_orientations))]
    local_rotation[0] = R.from_matrix(from_quat_safe(joint_orientations[0])).as_quat()
    local_position = [joint_positions[i] - joint_positions[parent_idx[i]] for i
                      in range(len(joint_orientations))]
    local_position[0] = joint_positions[0]

    path_end_id = path1[0]  ## lWrist_end 就是手掌 只是加了end不叫hand而已
    for k in range(0, 300):
        # k：循环次数
        # 正向的，path1是从手到root之前
        for idx in range(len(path2) - 1, -1, -1):  # len(path2)-1 --> 0
            path_joint_id = path2[idx]
            vec_to_end = joint_positions[path_end_id] - joint_positions[path_joint_id]
            vec_to_target = target_pose - joint_positions[path_joint_id]
            # 获取end->target的旋转矩阵
            # debug
            # rot_matrix=rotation_matrix(np.array([0.72,0.35,0]),np.array([0.5,0.35,0]))
            # rot_matrix=np.linalg.inv(rot_matrix)
            rot_matrix = get_rotation_matrix(vec_to_end, vec_to_target)

            # 计算前的朝向。注意path2是反方向的，要改父节点才行
            initial_orientation = from_quat_safe(joint_orientations[path_joint_id])
            # 旋转矩阵，格式换算
            rot_matrix_R = R.from_matrix(rot_matrix).as_matrix()
            # 计算后的朝向
            calculated_orientation = rot_matrix_R.dot(initial_orientation)
            # 写回结果列表
            joint_orientations[path_joint_id] = R.from_matrix(calculated_orientation).as_quat()

            # 其他节点的朝向也会有所变化
            for i in range(idx + 1, len(path2)):
                path_joint_id = path2[i]
                joint_orientations[path_joint_id] = R.from_matrix(
                    rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            # idx-1 就是当前节点的下一个更接近尾端的节点，一直向前迭代到1
            for i in range(len(path1) - 1, 0, -1):
                path_joint_id = path1[i]
                # 遍历路径后的节点,都乘上旋转
                joint_orientations[path_joint_id] = R.from_matrix(
                    rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            path_joint_id = path2[idx]
            # 修改父节点，或者说更靠近手的那些节点的位置
            # path2上的
            for i in range(idx+1, len(path2)):
                # path_joint_id=path1[i]
                # 节点id
                prev_joint_id = path2[i]
                # 指向上一个节点的向量
                vec_to_next = joint_positions[prev_joint_id] - joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir = rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next = calculated_vec_to_next_dir / np.linalg.norm(
                    calculated_vec_to_next_dir) * np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[prev_joint_id] = joint_positions[path_joint_id] + calculated_vec_to_next
            # path1上的
            for i in range(len(path1) - 1, -1, -1):
                # path_joint_id=path1[i]
                # 节点id
                prev_joint_id = path1[i]
                # 指向上一个节点的向量
                vec_to_next = joint_positions[prev_joint_id] - joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir = rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next = calculated_vec_to_next_dir / np.linalg.norm(
                    calculated_vec_to_next_dir) * np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[prev_joint_id] = calculated_vec_to_next + joint_positions[path_joint_id]
        for idx in range(1, len(path1)):
            # idx：路径上的第几个节点了，第0个是手，最后一个是root
            path_joint_id = path1[idx]

            vec_to_end = joint_positions[path_end_id] - joint_positions[path_joint_id]
            vec_to_target = target_pose - joint_positions[path_joint_id]
            # 获取end->target的旋转矩阵
            # debug
            # rot_matrix=rotation_matrix(np.array([1,0,0]),np.array([1,1,0]))
            rot_matrix = get_rotation_matrix(vec_to_end, vec_to_target)

            # 计算前的朝向。这个朝向实际上是累乘到父节点的
            initial_orientation = from_quat_safe(joint_orientations[path_joint_id])
            # 旋转矩阵，格式换算
            rot_matrix_R = R.from_matrix(rot_matrix).as_matrix()
            # 计算后的朝向
            calculated_orientation = rot_matrix_R.dot(initial_orientation)
            # 写回结果列表
            joint_orientations[path_joint_id] = R.from_matrix(calculated_orientation).as_quat()

            # 子节点的朝向也会有所变化
            # idx-1 就是当前节点的下一个更接近尾端的节点，一直向前迭代到1
            for i in range(idx - 1, 0, -1):
                path_joint_id = path1[i]
                # 遍历路径后的节点,都乘上旋转
                joint_orientations[path_joint_id] = R.from_matrix(
                    rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            path_joint_id = path1[idx]
            # 修改子节点的位置
            for i in range(idx - 1, -1, -1):
                # path_joint_id=path1[i]
                # 节点id
                next_joint_id = path1[i]
                # 指向下个节点的向量
                vec_to_next = joint_positions[next_joint_id] - joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir = rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next = calculated_vec_to_next_dir / np.linalg.norm(
                    calculated_vec_to_next_dir) * np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[next_joint_id] = calculated_vec_to_next + joint_positions[path_joint_id]
        # for idx in range(1, len(path1)):
        # # 求旋转矩阵
        #     joint_idx = path1[idx]
        #     vec_parent2target = target_pose - joint_positions[joint_idx]
        #     vec_parent2end = joint_positions[path_end_id] - joint_positions[joint_idx]
        #     # 从到子节点的向量转到目标节点的向量
        #     rot_matrix = get_rotation_matrix(vec_parent2end, vec_parent2target)
        #
        #     # 计算前的朝向。这个朝向实际上是累乘到父节点的
        #     initial_orientation = from_quat_safe(joint_orientations[joint_idx])
        #     # 旋转矩阵，格式换算
        #     rot_matrix_R = R.from_matrix(rot_matrix).as_matrix()
        #     # 计算后的朝向
        #     calculated_orientation = rot_matrix_R.dot(initial_orientation)
        #     # 写回结果列表
        #     joint_orientations[joint_idx] = R.from_matrix(calculated_orientation).as_quat()
        #     # 这条链上的joint全部转一下
        #     for i in range(idx-1, 0, -1):
        #         path_id = path1[i]
        #         joint_orientations[i] = R.from_matrix(rot_matrix_R.dot(from_quat_safe(joint_orientations[path_id]))).as_quat()
        #     # 修改位置
        #     origin_id = path1[idx]
        #     for i in range(idx-1,-1,-1):
        #         path_id = path1[i]
        #         # 以joint_idx为下表的点为圆心转
        #         vec_origin2current = joint_positions[path_id] - joint_positions[origin_id]
        #         # 左乘，改变向量
        #         calculated_vec_to_next_dir = rot_matrix.dot(vec_origin2current)
        #         # 防止长度不对
        #         calculated_vec_to_next = calculated_vec_to_next_dir / np.linalg.norm(
        #             calculated_vec_to_next_dir) * np.linalg.norm(vec_origin2current)
        #         # 还原回去
        #         joint_positions[path_id] = calculated_vec_to_next + joint_positions[origin_id]

        # path2是从脚到root，所以要倒着
        # debug
        # for idx in range(len(path2)-1,len(path2)-3,-1): # len(path2)-1 --> 0

        # debug
        # rot_matrix=rotation_matrix(np.array([1,0,0]),np.array([1,0,1]))
        # joint_orientations[0]=R.from_matrix(rot_matrix).as_quat()
        # joint_orientations[1]=R.from_matrix(rot_matrix).as_quat()
        joint_orientations[path_end_id] = joint_orientations[path1[1]]
        cur_dis = np.linalg.norm(joint_positions[path_end_id] - target_pose)
        if cur_dis < 0.01:
            break
    print("距离", cur_dis, "迭代了", k, "次")
    # 更新不在链上的节点
    for k in range(len(joint_orientations)):
        if k in path:
            pass
        elif k == 0:
            # 要单独处理，不然跟节点的-1就会变成从最后一个节点开始算
            pass
        else:
            # 先获取局部旋转
            # 这里如果直接存的就是矩阵就会有问题？
            local_rot_matrix = R.from_quat(local_rotation[k]).as_matrix()
            # 再获取我们已经计算了的父节点的旋转
            parent_rot_matrix = from_quat_safe(joint_orientations[parent_idx[k]])
            # 乘起来
            # re=local_rot_matrix.dot(parent_rot_matrix)
            re = parent_rot_matrix.dot(local_rot_matrix)
            joint_orientations[k] = R.from_matrix(re).as_quat()

            # 父节点没旋转的时候是：
            initial_o = from_quat_safe(no_caled_orientation[parent_idx[k]])
            # 父节点的旋转*delta_orientation=子节点旋转
            # 反求delta_orientation
            delta_orientation = np.dot(re, np.linalg.inv(initial_o))
            # 父节点的位置加原本基础上的旋转
            joint_positions[k] = joint_positions[parent_idx[k]] + delta_orientation.dot(local_position[k])

    return joint_positions, joint_orientations


class Cache_Data:
    def __init__(self, IK_joint_positions, IK_joint_orientations, Root_position):
        self.IK_joint_positions = IK_joint_positions
        self.IK_joint_orientations = IK_joint_orientations
        self.RootPosition = Root_position


animation_cache = {}
def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    target_pose = joint_positions[0] + np.array([relative_x, target_height - joint_positions[0][1], relative_z])
    Ori_joint_positions = copy.deepcopy(joint_positions)
    Ori_joint_orientations = copy.deepcopy(joint_orientations)
    relative_high = target_height - joint_positions[0][1]
    if relative_high in animation_cache:
        for i in range(len(joint_orientations)):
            if i in path:
                joint_positions[i] = animation_cache[relative_high].IK_joint_positions[i]+joint_positions[0]-animation_cache[relative_high].RootPosition
                joint_orientations[i] = animation_cache[relative_high].IK_joint_orientations[i]
            else:
                joint_positions[i] = Ori_joint_positions[i]
                joint_orientations[i] = Ori_joint_orientations[i]
        return joint_positions, joint_orientations
    IK_joint_positions, IK_joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                                                                         target_pose)
    animation_cache[relative_high] = Cache_Data(IK_joint_positions, IK_joint_orientations, joint_positions[0])
    for i in range(len(joint_orientations)):
        if i in path:
            joint_positions[i] = IK_joint_positions[i]
            joint_orientations[i] = IK_joint_orientations[i]
        else:
            joint_positions[i] = Ori_joint_positions[i]
            joint_orientations[i] = Ori_joint_orientations[i]
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations

def jocobian_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    joint_parent = meta_data.joint_parent
    joint_offset = [meta_data.joint_initial_position[i] - meta_data.joint_initial_position[joint_parent[i]] for i in
                    range(len(joint_positions))]
    joint_offset[0] = np.array([0., 0., 0.])
    joint_ik_path, _, _, _ = meta_data.get_path_from_root_to_end()
    # 用于迭代计算IK链条上各个关节的旋转
    local_rotation = [R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i]) for i
                      in range(len(joint_orientations))]
    local_rotation[0] = R.from_quat(joint_orientations[0])
    # 梯度下降方法
    joint_offset_t = [torch.tensor(data) for data in joint_offset]
    joint_positions_t = [torch.tensor(data) for data in joint_positions]
    joint_orientations_t = [torch.tensor(R.from_quat(data).as_matrix(), requires_grad=True) for data in
                            joint_orientations]
    local_rotation_t = [torch.tensor(data.as_matrix(), requires_grad=True) for data in local_rotation]
    target_pose_t = torch.tensor(target_pose)

    epoch = 300
    alpha = 0.5
    for _ in range(epoch):
        for j in range(len(joint_ik_path)):
            # 更新链上结点的位置
            a = chain_current = joint_ik_path[j]
            b = chain_parent = joint_ik_path[j - 1]
            if j == 0:
                local_rotation_t[a] = local_rotation_t[a]
                joint_positions_t[a] = joint_positions_t[a]
            elif b == joint_parent[a]:  # 当前结点是前一结点的子节点，正向
                joint_orientations_t[a] = joint_orientations_t[b] @ local_rotation_t[a]
                joint_positions_t[a] = joint_positions_t[b] + joint_offset_t[a] @ torch.transpose(
                    joint_orientations_t[b], 0, 1)
            else:  # a = joint_parent[b] 当前结点是前一节点的父结点，逆向
                joint_orientations_t[a] = joint_orientations_t[b] @ torch.transpose(local_rotation_t[b], 0, 1)
                joint_positions_t[a] = joint_positions_t[b] + (-joint_offset_t[a]) @ torch.transpose(
                    joint_orientations_t[a], 0, 1)

        optimize_target = torch.norm(joint_positions_t[joint_ik_path[-1]] - target_pose_t)
        if optimize_target < 0.01:
            break
        # 这里会自动对带了require_grad的矩阵求雅可比矩阵
        optimize_target.backward()
        for num in joint_ik_path:
            if local_rotation_t[num].grad is not None:
                tmp = local_rotation_t[num] - alpha * local_rotation_t[num].grad
                local_rotation_t[num] = torch.tensor(tmp, requires_grad=True)

    for j in range(len(joint_ik_path)):
        a = chain_current = joint_ik_path[j]
        b = chain_parent = joint_ik_path[j - 1]
        if j == 0:
            local_rotation[a] = R.from_matrix(local_rotation_t[a].detach().numpy())
            joint_positions[a] = joint_positions[a]
        elif b == joint_parent[a]:  # 当前结点是前一结点的子节点，正向
            joint_orientations[a] = (R.from_quat(joint_orientations[b]) * R.from_matrix(
                local_rotation_t[a].detach().numpy())).as_quat()
            joint_positions[a] = joint_positions[b] + joint_offset[a] * np.asmatrix(
                R.from_quat(joint_orientations[b]).as_matrix()).transpose()
        else:  # a = joint_parent[b] 当前结点是前一节点的父结点，逆向
            joint_orientations[a] = (R.from_quat(joint_orientations[b]) * R.from_matrix(
                local_rotation_t[b].detach().numpy()).inv()).as_quat()
            joint_positions[a] = joint_positions[b] + (-joint_offset[b]) * np.asmatrix(
                R.from_quat(joint_orientations[a]).as_matrix()).transpose()

    # 我们获得了链条上每个关节的Orientation和Position，然后我们只需要更新非链上结点的位置
    ik_path_set = set(joint_ik_path)
    for i in range(len(joint_positions)):
        if i in ik_path_set:
            joint_orientations[i] = R.from_matrix(joint_orientations_t[i].detach().numpy()).as_quat()
        else:
            joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * local_rotation[i]).as_quat()
            joint_positions[i] = joint_positions[joint_parent[i]] + joint_offset[i] * np.asmatrix(
                R.from_quat(joint_orientations[joint_parent[i]]).as_matrix()).transpose()

    return joint_positions, joint_orientations


def part2_jacobian_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    target_pose = joint_positions[0] + np.array([relative_x, target_height - joint_positions[0][1], relative_z])
    Ori_joint_positions = copy.deepcopy(joint_positions)
    Ori_joint_orientations = copy.deepcopy(joint_orientations)
    relative_high = target_height - joint_positions[0][1]
    if relative_high in animation_cache:
        for i in range(len(joint_orientations)):
            if i in path:
                joint_positions[i] = animation_cache[relative_high].IK_joint_positions[i] + joint_positions[0] - \
                                     animation_cache[relative_high].RootPosition
                joint_orientations[i] = animation_cache[relative_high].IK_joint_orientations[i]
            else:
                joint_positions[i] = Ori_joint_positions[i]
                joint_orientations[i] = Ori_joint_orientations[i]
        return joint_positions, joint_orientations
    IK_joint_positions, IK_joint_orientations = jocobian_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                                                                         target_pose)
    animation_cache[relative_high] = Cache_Data(IK_joint_positions, IK_joint_orientations, joint_positions[0])
    for i in range(len(joint_orientations)):
        if i in path:
            joint_positions[i] = IK_joint_positions[i]
            joint_orientations[i] = IK_joint_orientations[i]
        else:
            joint_positions[i] = Ori_joint_positions[i]
            joint_orientations[i] = Ori_joint_orientations[i]
    return joint_positions, joint_orientations