import numpy as np

def apply_tf(x, pose2d):
    # x is in frame B
    # pose2d is AT_B
    # result is x in frame A
    return rotate(x, pose2d[2]) + np.array(pose2d[:2])

def apply_tf_to_pose(pose, pose2d):
    # same as apply_tf but assumes pose is x y theta instead of x y
    xy = rotate(np.array([pose[:2]]), pose2d[2])[0] + np.array(pose2d[:2])
    th = pose[2] + pose2d[2]
    return np.array([xy[0], xy[1], th])

def apply_tf_to_vel(vel, pose2d):
    # same as apply_tf but assumes vel is xdot ydot thetadot instead of x y
    # for xdot ydot frame transformation applies, 
    # but thetadot is invariant due to frames being fixed.
    xy = rotate(np.array([vel[...,:2]]), pose2d[2])[0]
    th = vel[...,2]
    return np.array([xy[0], xy[1], th])

def rotate(x, th):
    rotmat = np.array([
        [np.cos(th), -np.sin(th)],
        [np.sin(th), np.cos(th)],
        ])
    return np.matmul(rotmat, x.T).T

def inverse_pose2d(pose2d):
    inv_th = -pose2d[2] # theta
    inv_xy = rotate(np.array([-pose2d[:2]]), inv_th)[0]
    return np.array([inv_xy[0], inv_xy[1], inv_th])
