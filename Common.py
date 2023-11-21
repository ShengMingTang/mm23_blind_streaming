import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from pathlib import Path
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

meshProxy = {}

def makeLSE(x, y):
    '''
    return a function a results in least-square-error estimation from x to y
    '''
    meanX, meanY = np.mean(x), np.mean(y)
    sigX, sigY = np.std(x), np.std(y)
    if sigY == 0:
        r = 0
    else:
        r = np.corrcoef([x, y])
        r = r[0, 1]
    def f(x):
        return r * sigY / (sigX + 1e-3) * (x - meanX) + meanY
    return f
def convertUnityPoses7ToMIVCoord(unityPoses):
    '''
    * unityPoses: poses in unity frame (7,) or (N, 7)
    
    MIV coord:
        Initial orientation (1,0,0) in MIV frame
        front = (1,0,0) in MIV frame
        left  = (0,1,0) in MIV frame
        up    = (0,0,1) in MIV frame
        Rotation follows right-hand rule for every axis
        Rotation order = roll, pitch, then yaw
    
    Deal position and orientation separately
    Quaternion recorded in Unity means the same in MIV frame (describe the same rotation)
    For position:
        MIV_x = Unity_z (MIV_front = Unity_front)
        MIV_y = Unity_(-x) (MIV left = Unity left = Unity -X)
        MIV_z = Unity_y (MIV up = Unity up = Unity Y)
    For orientation:
        MIV_roll = right hand at MIV_X = reverse of Unity rotation at Z-rotation axis
        MIV_pitch = right hand at MIV_Y = Unity rotation at X-rotation axis
        MIV_yaw = right hand at MIV_Z = reverse of Unity rotation at Y-rotation axis
    '''
    unityPoses = unityPoses.reshape((-1, 7))
    # x, y, z, yaw, pitch, roll
    ret = np.zeros((unityPoses.shape[0], 6))
    
    ret[:, 0] = unityPoses[:, 2]
    ret[:, 1] = -unityPoses[:, 0]
    ret[:, 2] = unityPoses[:, 1]
    
    q02 = R.from_quat(unityPoses[:, 3:])
      
    # MIV's Roll, Pitch, then Yaw
    rot = q02.as_euler('zxy', degrees=True)
    ret[:, 3] = -rot[:, 2] # y
    ret[:, 4] = rot[:, 1] # x
    ret[:, 5] = -rot[:, 0] # z
    
    return ret
def convertUnityPoses7ToO3d7(pose: np.array):
    '''
    * pose: (N, 7)
    world's x axes === axes shown in the editor
    
    Unity's Quaternion (record from HMD)
        Initial orientation = (0,0,1) in Unity frame
        (1) +x rotation axis = right-handed anchored at opposite direction of world's x axis
        (2) +y rotation axis = right-handed anchored at opposite direction of world's y axis
        (3) +z rotation axis = right-handed anchored at opposite direction of world's z axis
    Unity's Position (record from HMD)
        +x, y, z axes are aligned with the world's axes
        
    (4) .obj loaded in open3d: X-axis is reversed of Unity's X-axis, y,z axes are aligned
    To convert:
    position: x reversed due to (1)
    rotation:
        x: no change, due to (1) and (4)
        y, z: reversed, due to (2) and (3)
        
    converted O3d7 (same, scalar last) same as scipy.spatial.transform.Rotation
    '''
    ret = np.array(pose).reshape((-1, 7))
    ret[:, 0] = -ret[:, 0] # reverse position x
    ret[:, [4, 5]] = -ret[:, [4, 5]] # reverse rotation y, z
    return np.squeeze(ret)
def convertO3d7ToUnityPoses7(pose: np.array):
    return convertUnityPoses7ToO3d7(pose)
def convertO3d7ToO3dEyeCenterUp(pose: np.array):
    '''
    pose: (7,) or (..., 7)
    '''
    p = np.array(pose)
    p = p.reshape((-1, 7))
    eye = p[..., :3]
    r = R.from_quat(p[..., 3:])
    center = np.array([0, 0, 1])
    up = np.array([0, 1, 0])
    center = r.apply(center) + eye
    up = r.apply(up)
    if len(pose.shape) == 1:
        eye, center, up = eye.reshape((3,)), center.reshape((3,)), up.reshape((3,))
    return np.squeeze(eye), np.squeeze(center), np.squeeze(up)
def raysToLength(rays: np.array):
    '''
    * rays: (H, W, 6) rays created by ContentCreator.createRays
        direction is the vector from center to the plane (not unit vectors)
    return (H, W) representing length of each ray
    '''
    rayLen = np.array(rays)
    rayLen = np.sqrt(np.sum(np.square(rays[..., 3:]), axis=-1))
    return rayLen
def makeMeshFromRaysDepth(rays: np.array, depth: np.array):
    '''
    return a mesh created from ray-depth array
    
    * rays: (H, W, 6) rays created by ContentCreator.createRays
        direction is the vector from center to the plane (not unit vectors)
    * depth: (H, W) in rendered by ContentCreator.renderD
    
    (i -1, j - 1) ---------- (i - 1,j)
          |                /   |
          |      /             |
      (i, j - 1)  ---------- (i, j)
    '''
    if (depth.shape in meshProxy) is False:
        meshProxy[depth.shape] = o3d.geometry.TriangleMesh(makeMesh(depth.shape[0], depth.shape[1]))
    mesh = o3d.geometry.TriangleMesh(meshProxy[depth.shape])
    # mesh = meshProxy[depth.shape]
    v = depth[..., None] * rays[::-1, ::-1, 3:] + rays[::-1, ::-1, :3] # fast
    # ! must cast type to np.float64, see https://github.com/isl-org/Open3D/issues/1045
    mesh.vertices = o3d.utility.Vector3dVector(v.reshape((-1, 3)).astype(np.float64))
    return mesh
def makeMesh(h, w):
    '''
    create a mesh with hxw vertices, position invalid and should be set later for use
    '''
    mesh = o3d.geometry.TriangleMesh()
    f = []
    for i in range(1, h):
        for j in range(1, w):
            f.append([(i-1)*w + j, (i-1)*w + (j - 1), i*w + (j - 1)])
            f.append([(i-1)*w + j, i*w + (j - 1), i*w + j])
    v = np.ones((h, w, 3))
    mesh.vertices = o3d.utility.Vector3dVector(v.reshape((-1, 3)))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(f, dtype=np.int32))
    return mesh
def makeCoverageMap(dEst: np.array, dTrue: np.array, thres=1e-1):
    '''
    * dEst: (H, W) or (-1, H, W) estimated depth map
    * dTrue: (H, W) or (-1, H, W) ground truth depth map
    * thres: threshold to reject false depth
    
    return dEst - dTrue with thresholding
    '''
    diff = np.abs(dEst - dTrue)
    diff[diff == np.nan] = np.inf # happens
    diff[diff > thres] = np.inf
    return diff != np.inf
def makeFoveationWeights(h, w, sizeY, sizeX, shiftY, shiftX, edgeRatioY, edgeRatioX):
    '''
    h: weights height
    w: weights width
    sizeY: vertical foveation size [0, 1]
    sizeX: horizontal foveation size [0, 1]
    shiftY: vertical center shift [0, 1]
    shiftX: horizontal center shift [0, 1]
    edgeRatioY: int, vertical edge compression ratio
    edgeRatioX: int, horizontal edge compression ratio
    
    return W (h, w)
    np.sum(W) == 1
    '''
    W = np.ones((h, w)) / (edgeRatioX * edgeRatioY)
    shift = np.array([shiftY, shiftX])
    screen = np.array([h, w])
    size = np.array([sizeY, sizeX])
    center = 0.5 + shift
    c0 = center - size / 2 # top-left
    c1 = center + np.array([-size[0]/2, size[1]/2]) # top-right
    c2 = center + np.array([size[0]/2, -size[1]/2]) # bottom-left
    c3 = center + size / 2 # bottom-right
    c0 = np.around(screen * c0).astype(np.int32)
    c1 = np.around(screen * c1).astype(np.int32)
    c2 = np.around(screen * c2).astype(np.int32)
    c3 = np.around(screen * c3).astype(np.int32)
    
    # ! treat non-center region the same
    W[c0[0]:c3[0], c0[1]:c3[1]] = 1
    
    # ! cut into 8 slices
    # W[0:c0[0], 0:c0[1]] = 1 / (edgeRatioX * edgeRatioY) # top-left
    # W[0:c0[0], c0[1]:c1[0]] = 1 / edgeRatioY # top-mid
    # W[0:c0[0], c1[0]:c1[1]] = 1 / (edgeRatioX * edgeRatioY) # top-right
    # W[c0[0]:c2[0], 0:c1[1]]  = 1 / edgeRatioX # mid-left
    # ...
    
    W = W / np.sum(W)
    return W
def optProbRatio(cspPolicy, m, h, P=None):
    '''
    * cspPolicy: '1Prob'
    See the candidate generator session
    return asymptotic optimal number of candidates given m, h
    '''    
    if cspPolicy == '1Prob':
        return np.sqrt((m + h) / m)
    else:
        return NotImplemented
def pixelActivation(x, a):
    '''
    See Solver section (f(c) in paper)
    1 - e^(-ax) for modeling pixel level quality score
    '''
    assert a <= np.log(0.5)
    if device == 'cpu':
        return 1 - np.exp(a * x)
    else:
        return 1 - torch.exp(a * x)


# ================== backup ==================
# scalar arithmetic
'''
array for placing the result of mean/var should be pre-padded with 0/1 for scalar/angular

https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
'''
def scalarMeanUpdate(x, u_n1, n):
    '''
    x: new data
    u_n1: mean of u_{n-1} (previous output of this function)
    n: number of element update so far (after this call)
    '''
    return ((n-1) * u_n1 + x) / n
def scalarVarUpdate(v_n1, u_n, u_n1, n):
    '''
    v_n1: v_{n-1} previous variance
    u_n: mean (previous output of scalarMeanUpdate)
    u_n1: mean (the last 2nd output of scalarMeanUpdate)
    n: number of element update so far (after this call)
    '''
    return (n - 1)/n * v_n1 + (n-1) * np.square(u_n1 - u_n)
def scalarMeanMN(u, m, n):
    '''
    calculate mean in (m, n]
    u: output series in scalvarMeanUpdate
    u_m: mean up to m
    u_n: mean up to n
    
    u[i] = mean containing previous i element (i >= 0)
    mean(x[m:n]) == scalarMeanMN(u[m], u[n], m, n)
    '''
    if m == n:
        return 0
    u_m, u_n = u[m], u[n]
    return (n * u_n - m * u_m) / (n - m)
def scalarVarMN(v, u, m, n):
    '''
    v: output series in scalarVarUpdate
    u: output series in scalvarMeanUpdate
    calculate var in (m, n]
    v_m: var up to m
    v_n: var up to n
    u_m: mean up to m
    u_n: mean up to n
    u_mn: mean from m to n
    '''
    if m == n:
        return 0
    v_m, v_n = v[m], v[n]
    u_m, u_n = u[m], u[n]
    u_mn = scalarMeanMN(u, m, n)
    return ( 
        n*(v_n + np.square(u_n - u_mn)) - m * (v_m + np.square(u_m - u_mn))
        ) / (n - m)
# anlge complex arithmetic
def angularMeanUpdate(a, eju_n1, n):
    '''
    calculate e^{angular mean}
    a: new angle in rad
    u_n1: complex average (previous output of this function)
    n: number of element update so far (after this call)
    '''
    return np.power(
        np.exp(1j * a)
        , (1/n)
    ) * np.power(
        eju_n1
        , (n-1)/n
    )
def e_ab2(eja, ejb):
    '''
    compute e^((a - b) ^ 2)
    '''
    # return np.power(
    #     (   np.power(eja, np.log(eja))
    #         *
    #         np.power(ejb, np.log(ejb))
    #     )
    #     /
    #     np.square(
    #         np.power(
    #             eja
    #             , np.log(ejb)
    #         )
    #     )
    #     , 1/1j
    # )
    return np.power(
        eja / ejb,
        np.angle(eja) - np.angle(ejb)
    )
def angularVarUpdate(ejv_n1, eju_n, eju_n1, n):
    return np.power(ejv_n1, (n - 1)/n) * np.power(
        e_ab2(eju_n, eju_n1)
        , n-1
    )
def angularMeanMN(eju, m, n):
    '''
    calculate e^{angular mean} in (m, n]
    ejum: output of e^{angular mean up to m}
    ejun: output of e^{angular mean up to n}
    '''
    if m == n:
        return 1.0 + 0j
    ejum, ejun = eju[m], eju[n]
    return np.power(
        np.power(ejun, n)
        /
        np.power(ejum, m)
        , 1/(n - m)
    )
def angularVarMN(ejv, eju, m, n):
    '''
    calculate e^{angular mean} in (m, n]
    ejsm: output of e^{angular var up to m}
    ejsn: output of e^{angular var up to n}
    ejum: output of e^{angular mean up to m}
    ejun: output of e^{angular mean up to n}
    ejumn: output of e^{angular mean from m to n}
    '''
    if m == n:
        return 1.0 + 0j
    ejv_m, ejv_n = ejv[m], ejv[n]
    eju_m, eju_n = eju[m], eju[n]
    ejumn = angularMeanMN(eju, m, n)
    return np.power(
        np.power(ejv_n * e_ab2(eju_n, ejumn), n)
        /
        np.power(ejv_m * e_ab2(eju_m, ejumn), m)
        , 1/(n - m)
    )