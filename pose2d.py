import numpy as np

def Pose2D(tf_msg):
    import tf
    return np.array([tf_msg[0][0], tf_msg[0][1], 
        tf.transformations.euler_from_quaternion(tf_msg[1])[2]])

def posemsg_to_pose2d(posemsg):
    import tf
    quaternion = (
        posemsg.orientation.x,
        posemsg.orientation.y,
        posemsg.orientation.z,
        posemsg.orientation.w,
    )
    xytheta = np.array([0., 0., 0.])
    xytheta[0] = posemsg.position.x
    xytheta[1] = posemsg.position.y
    xytheta[2] = tf.transformations.euler_from_quaternion(quaternion)[2]
    return xytheta

def apply_tf(x, pose2d):
    """ Applies transform to x
    x is a list of x y points in frame B
    pose2d is the transform AT_B, (frame B in frame A)
    result is x in frame A

    Parameters
    ----------
        pose2d : np.array
            Array of shape (3,), x, y, theta components of the tf of frame B in frame A
        x: np.array
            Array of shape (N, 2), x, y coordinates of N points in frame B
    Returns
    -------
        result: np.array
            Array of shape (N, 2), x, y coordinates of N points in frame A

    Note
    ----
    pose2d is under the form [x, y, theta], where 
    x and y are coordinates of frame B in frame A, 
    and theta is the orientation of frame B in frame A.

    Example
    -------
    >>> BX = np.array([[0, 0], [1, 0]])
    >>> AT_B = np.array([2, 1, np.pi/2])
    >>> apply_tf(BX, AT_B)
    array([[ 2.,  1.],
           [ 2.,  2.]])
    """
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
    xy = rotate(vel[...,:2], pose2d[2])
    th = vel[...,2]
    xyth = np.zeros_like(vel)
    xyth[...,:2] = xy
    xyth[...,2] = th
    return xyth

def old_apply_tf_to_vel(vel, pose2d):
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
