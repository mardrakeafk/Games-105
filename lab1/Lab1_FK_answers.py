import numpy as np
from scipy.spatial.transform import Rotation as R


class Node(object):
    def __init__(self, name=None, offset=None):
        self.name = name
        self.offset = offset
        self.children = []

    def add_child(self, NodeName):
        self.children.append(NodeName)


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        joint = []
        global root
        for i in range(len(lines)):
            data = lines[i].strip()
            if data.startswith('MOTION'):
                break
            if data.startswith('ROOT') or data.startswith('JOINT') or data.startswith('End'):
                name = data.split()[1]
                if data.startswith('ROOT'):
                    name = 'RootJoint'
                if data.startswith('End'):
                    name = joint[-1].name + '_end'
                node = Node(name)
                if len(joint) != 0:
                    joint[-1].add_child(node)
                else:
                    root = node
                joint.append(node)
            if data.startswith('OFFSET'):
                vec = np.array([float(x) for x in data.split()[1:]]).reshape(1, -1)
                joint[-1].offset=vec
            if data.startswith('}'):
                joint.pop()

    joint_name = []
    joint_parent = []
    joint_offset = []
    def preorder(root, parent_index):
        joint_name.append(root.name)
        joint_parent.append(parent_index)
        joint_offset.append(root.offset)
        parent = len(joint_name)-1
        for i in root.children:
            preorder(i, parent)
    preorder(root, -1)
    joint_offset = np.concatenate(joint_offset, axis=0)
    # print(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    motion_channels_data = motion_data[frame_id]
    # bvh文件motion数据的前三位表示根节点的位置
    root_position = np.array(motion_channels_data[0:3])
    joint_local_rotation = []
    # 读取每个joint的rotation，注意motion data里面没有end_joint的旋转，end_joint旋转为0，0，0
    count = 1
    for joint in joint_name:
        if '_end' in joint:
            joint_local_rotation.append([0., 0., 0.])
        else:
            joint_local_rotation.append(motion_channels_data[count*3:count*3+3])
            count += 1
    joint_positions = []
    joint_orientations = []
    for i in range(len(joint_name)):
        # 根节点
        # 问题根节点有旋转吗？
        # 还是有的，比如走路的朝向之类的
        if joint_parent[i] == -1:
            joint_position = root_position.reshape(1,-1)
            joint_orientation = R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
        else:
            joint_position = joint_positions[joint_parent[i]]+joint_offset[i] * np.asmatrix(R.from_quat(joint_orientations[joint_parent[i]]).as_matrix()).transpose()
            joint_orientation = R.from_quat(joint_orientations[joint_parent[i]])* R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
        joint_positions.append(np.array(joint_position))
        joint_orientations.append(joint_orientation.as_quat().reshape(1, -1))
    joint_positions = np.concatenate(joint_positions, axis=0)
    joint_orientations = np.concatenate(joint_orientations, axis=0)
    # print(joint_positions)
    # print(joint_orientations)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_joint_name, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, _, _ = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion = load_motion_data(A_pose_bvh_path)
    global count
    count = 0
    index_list = {}
    for i in range(len(A_joint_name)):
        if '_end' in A_joint_name[i]:
            count += 1
        else:
            index_list[A_joint_name[i]] = i-count
    # print(index_l_t, index_r_t,index_l_a,index_r_a)
    motion_data = []
    for motion in A_motion:
        frame_data = []
        for joint in T_joint_name:
            if joint == 'RootJoint':
                frame_data += list(motion[0:6])
            elif '_end' in joint:
                continue
            elif joint == 'lShoulder':
                rotation = (R.from_euler('XYZ', motion[index_list['lShoulder']*3+3:index_list['lShoulder']*3+6], degrees=True)*R.from_euler('XYZ',[0.,0.,-45.])).as_euler('XYZ',True)
                frame_data+= list(rotation)
            elif joint == 'rShoulder':
                rotation = (R.from_euler('XYZ', motion[index_list['rShoulder'] * 3 + 3:index_list['rShoulder'] * 3 + 6],
                                         degrees=True) * R.from_euler('XYZ', [0., 0., 45.])).as_euler('XYZ', True)
                frame_data += list(rotation)
            else:
                frame_data += list(motion[index_list[joint]*3+3:index_list[joint]*3+6])
        motion_data.append(np.array(frame_data).reshape(1, -1))
    motion_data = np.concatenate(motion_data, axis=0)
    return motion_data
