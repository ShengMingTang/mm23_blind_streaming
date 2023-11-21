from PoseFeeder import *
from PoseSplitter import *
from pathlib import Path
from CamPlace import *
from Solver import *
import open3d as o3d
from ContentCreator import *
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import time
from itertools import combinations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
Phase 1:
Test pose feeder and local placement using average position/quaternion
'''
# * pass
def testPoseFeeder():
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    pfd.summary(dir)
    for u, pose in enumerate(pfd):
        print(u, pose.shape)
        print(pose[0, 0])
# * pass
def testPoseSplitterUniform():
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    pfd.summary(dir)
    
    psSt = PoseSplitter.GetDefaultSettings()
    psSt['uniform']['numPart'] = 17
    ps = PoseSplitter(psSt)
    ps.summary(dir)
    
    for u, pose in enumerate(pfd): # pose (U, N, 7)
        indices = ps.splitPose(pose)
        longPose = np.array(pose).reshape((-1, 7))
        for i in range(3):
            plt.subplot(311 + i)
            plt.scatter(np.arange(longPose.shape[0]), longPose[:, i], s=1)
            plt.vlines(x=indices.flatten()[::2], ymin = np.min(longPose[:, i]), ymax = np.max(longPose[:, i]),
                colors = 'red', linestyles='dashed'
            )
            plt.vlines(x=list(range(0, pose.shape[0], pose.shape[0] * pose.shape[1])), ymin = np.min(longPose[:, i]), ymax = np.max(longPose[:, i]),
                colors = 'green', linestyles='dashed'
            )
        plt.savefig(str(dir/f'uniform-u0-{u}.png'))
        plt.clf()
# * pass
def testPoseSplitterIXR():
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['poseDir'] = Path('Trace_FPS50_LEN30')/'FurnishedCabin'
    pfd = PoseFeeder(pfdSt)
    pfd.summary(dir)
    
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['width'] = ccSt['width'] // 1
    ccSt['height'] = ccSt['height'] // 1
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    
    psSt = PoseSplitter.GetDefaultSettings()
    psSt['policy'] = 'IXR'
    psSt['IXR']['thres'] = 0.85
    psSt['IXR']['depthThres'] = 1e-2
    
    psSt['IXR']['ccSt'] = ccSt
    ps = PoseSplitter(psSt)
    # ps.summary(dir)
    
    for u, pose in enumerate(pfd): # pose (U, N, 7)
        indices = ps.splitPose(pose)
        print(len(indices), indices)
        # longPose = np.array(pose).reshape((-1, 7))
        # for i in range(3):
        #     plt.subplot(311 + i)
        #     plt.scatter(np.arange(longPose.shape[0]), longPose[:, i], s=1)
        #     plt.vlines(x=indices.flatten()[::2], ymin = np.min(longPose[:, i]), ymax = np.max(longPose[:, i]),
        #         colors = 'red', linestyles='dashed'
        #     )
        #     plt.vlines(x=list(range(0, pose.shape[0], pose.shape[0] * pose.shape[1])), ymin = np.min(longPose[:, i]), ymax = np.max(longPose[:, i]),
        #         colors = 'green', linestyles='dashed'
        #     )
        # plt.savefig(str(dir/f'uniform-u0-{u}.png'))
        # plt.clf()
# * pass
def testPoseSplitterMotion():
    '''
    # segments = 4921, # samples = 23200, avg = 4.714488925015241 samples/seg
    expected segments = 48 * 30 = 1440
    ''' 
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    pfd.summary(dir)
    
    psSt = PoseSplitter.GetDefaultSettings()
    psSt['policy'] = 'motion'
    psSt['motion']['posThres'] = [0.1**2, 0.1**2, 0.1**2, 0.1**2]
    psSt['motion']['rotThres'] = 5 # 5 degree
    psSt['motion']['minSample'] = 3
    psSt['force'] = True
    ps = PoseSplitter(psSt)
    ps.summary(dir)
    totalSegments = 0
    nUsers = 0
    a = 0
    uMax = 0
    if dir is not None:
        plt.figure(figsize=(10, 10))
    for u, poses in enumerate(pfd): # pose (U, N, 7)
        print(u)
        uMax = u
        indices = [ps.splitPose(pose) for pose in poses] # list of (M, 2)
        # stat
        for ind in indices:
            totalSegments += ind.shape[0]
        nUsers = poses.shape[0]
        # we plot user a as an example
        ind = indices[a]
        pose = poses[a]
        titles = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        for i in range(7):
            plt.subplot(int(f'71{i+1}'))
            plt.plot(pose[:, i], linewidth=3)
            # mark prediction
            plt.vlines(x=indices[a].flatten()[::2], ymin = np.min(pose[:, i]), ymax = np.max(pose[:, i]),
                colors = 'red', linestyles='dashed'
            )
            if i < 3:
                for j in range(ind.shape[0]):
                    x = np.arange(ind[j, 0], ind[j, 1])
                    y = pose[ind[j, 0]:ind[j, 1], i]
                    lsef = makeLSE(x, y)
                    x = np.append(x, ind[j, 1])
                    yPred = lsef(x)
                    plt.plot(x, yPred, 'r')
            else:
                for j in range(ind.shape[0]):
                    x = np.arange(ind[j, 0], ind[j, 1])
                    if x.shape[0] > 1:
                        slerp = Slerp([ind[j, 0], ind[j, 1] - 1], R.from_quat(pose[[ind[j, 0], ind[j, 1] - 1], 3:]))
                        qPred = slerp(x).as_quat()
                        plt.plot(x, qPred[:, i - 3], 'r')
            plt.title(titles[i])
        if dir is not None:
            plt.savefig(str(dir/f'motion-u0-{u}.png'))
            plt.clf()
        else:
            plt.show()
            s = input()
            if s == 'q':
                return
    print(f'# segments = {totalSegments}, # samples = {uMax * nUsers * pfdSt["size"]}, avg = {uMax * nUsers * pfdSt["size"] / totalSegments} samples/seg')
# * passed
def testPoseSplitterFixedNumber():
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['poseDir'] = dir
    pfdSt['size'] = 50
    pfd = PoseFeeder(pfdSt)
    pfd.summary(dir)
    
    psSt = PoseSplitter.GetDefaultSettings()
    psSt['policy'] = 'fixedNumber'
    psSt['fixedNumber']['numPart'] = 64
    ps = PoseSplitter(psSt)
    ps.summary(dir)
    for u, pose in enumerate(pfd): # pose (U, N, 7)
        indices = ps.splitPose(pose)
        print(pose.shape)
        print(indices)
        longPose = np.array(pose).reshape((-1, 7))
        for i in range(3):
            plt.subplot(311 + i)
            plt.scatter(np.arange(longPose.shape[0]), longPose[:, i])
            plt.vlines(x=indices.flatten(), ymin = np.min(longPose[:, i]), ymax = np.max(longPose[:, i]),
                colors = 'red', linestyles='dashed'
            )
        plt.show()
        break
# * passed
def testCamPlace():
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    pfd.summary(dir)
    
    psSt = PoseSplitter.GetDefaultSettings()
    psSt['uniform']['numPart'] = 50*16//10
    ps = PoseSplitter(psSt)
    ps.summary(dir)
    
    camPSt = CamPlace.GetDefaultSettings()
    cam = CamPlace(camPSt)
    
    for u, pose in enumerate(pfd): # pose (U, N, 7)
        longPose = np.array(pose).reshape((-1, 7))
        indices = ps.splitPose(pose)
        for j in range(indices.shape[0]): # idx (2,)
            place = cam.localPlace(longPose[indices[j, 0]:indices[j, 1]])
            assert np.allclose(np.sum(np.square(place[3:])) , 1)
            place = np.round(place, 2)
            print(f'camPlace: {indices[j]}, place at {place}')
            print(f'average: {np.mean(longPose[indices[j, 0]:indices[j, 1]], axis=0)}')
            print('===================')
            s = input()

'''
Phase 2:
Test c.c. for rendering
'''
# * pass
def testMesh(path2Obj):
    '''
    Visualization
    http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Visualize-a-3D-mesh
    '''
    mesh = o3d.io.read_triangle_mesh(path2Obj)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([mesh, origin])
# * pass
def testRenderD():
    '''
    Test if renderD() is correct visually
    '''
    dir = Path('Trace_FPS50_LEN30')/'FurnishedCabin'
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['size'] = 1
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    # pfd.summary(dir)
    
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    cc = ContentCreator(ccSt)
    idx = 0
    outDir = Path('TestPoseDir')/'testRenderD'
    outDir.mkdir(exist_ok=True, parents=True)
    
    poses = next(iter(pfd)).reshape((-1, 7))
    poses = poses[:5]
    print(poses.shape)
    for u, p in enumerate(poses): # pose (U, N, 7)
        d = cc.renderD(p)
        d = 255 * (d / np.max(d))
        cv2.imwrite(str(outDir/f'renderD_Test-u0-d-{idx}.png'), d)
        idx += 1
        print(idx)
        if idx > 5:
            break
    
    poses = next(iter(pfd)).reshape((-1, 7))
    poses = poses[:5]
    d = cc.renderD(poses)
    for i in range(d.shape[0]):
        d[i] = 255 * d[i] / np.max(d[i])
        cv2.imwrite(str(outDir/f'renderD_Test-u0-d-batch-{i}.png'), d[i])
        print(f'batch {i} written')
    print(cc.getCost())
# * pass
def testDepthToMesh():
    '''
    Test create mesh from depth
    
    https://forum.open3d.org/t/add-triangle-to-empty-mesh/197
    http://www.open3d.org/docs/latest/tutorial/Basic/tensor.html
    
    t_hit is in unit of length of that ray
    
    # ray order in rays is reversed (up side down, left side right)
    rays = [
        [position, direction],
        ...
    ]
    
    1.
    d1 -> (create mesh) -> m1 -> (transform) -> m1' -> (render d) -> d1'
    
    2.
    d2  -> (diff) -> delta d -> (reject abnormal depth) -> d -> (coverage) -> cvg
    d1' -^        
    '''
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    ccSt['width'] = 2
    ccSt['height'] = 2
    cc = ContentCreator(ccSt)
    eye, center, up = np.array([0, 1, 0]), np.array([0, 2, 0]), np.array([0, 0, 1])
    rays = cc.createRays(eye, center, up)
    rays = rays.numpy()
    print(rays)
    # r = rays.reshape((-1, 6))
    # print(np.sqrt(np.sum((r[..., 3:] ** 2), axis=-1)))
    # plt.imshow(np.sqrt(np.sum((r[..., 3:] ** 2), axis=-1)).reshape((1080, 1920)))
    # plt.show()
    d = cc.renderD(np.array([0, 0, 0, 1, 0, 0, 0]))
    d = np.abs(np.random.normal(1, 0.2, size=d.shape))
    d = np.array([
        [1, 2],
        [3, 4]
    ])
    
    mesh = makeMeshFromRaysDepth(rays, d)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([origin, mesh],
        front=[0.5, 0.86, 0.125],
        lookat=[0.23, 0.5, 2],
        up=[-0.63, 0.45, -0.63],
        zoom=0.7
    )         
# * pass
def testCvqMap(show=True):
    '''
    test how p1 covers p2
    '''
    dir = Path('Trace_FPS50_LEN30')/'FurnishedCabin'
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['size'] = 1
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['width'] = ccSt['width'] // 1
    ccSt['height'] = ccSt['height'] // 1
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    cc = ContentCreator(ccSt)

    # p1, d1
    for u, p_ in enumerate(pfd):
        poses = p_
        break 
    p1 = poses[0, 0]
    eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(p1))
    rays1 = cc.createRays(eye, center, up)
    d1 = cc.renderD(p1)
    
    mesh1 = makeMeshFromRaysDepth(rays1.numpy(), d1)
    
    # * showing the pcd
    # * The point cloud and mesh should highly match with each other
    if show:
        ans = cc.castRays(rays1)
        hit = ans['t_hit'].isfinite()
        points = rays1[hit][:,:3] + rays1[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
        pcd = o3d.t.geometry.PointCloud(points)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pcd.to_legacy(), origin, mesh1],
            front=[0.5, 0.86, 0.125],
            lookat=[0.23, 0.5, 2],
            up=[-0.63, 0.45, -0.63],
            zoom=0.7
        )
    # advance some poses
    for u, p_ in enumerate(pfd):
        poses = p_
        if u == 2:
            break 
    p3 = poses[0, 0]
    # advance some poses
    for u, p_ in enumerate(pfd):
        poses = p_
        if u == 3:
            break
    p2 = poses[0, 0]
    eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(p2))
    rays2 = cc.createRays(eye, center, up)
    d2 = cc.renderD(p2)
    
    # * point cloud is the ground truth, mesh is the one created by the last pose
    # * These do not need to match
    if show:
        ans = cc.castRays(rays2)
        hit = ans['t_hit'].isfinite()
        points = rays2[hit][:,:3] + rays2[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 0.99
        pcd = o3d.t.geometry.PointCloud(points)
        o3d.visualization.draw_geometries([pcd.to_legacy(), origin, mesh1],
            front=[0.5, 0.86, 0.125],
            lookat=[0.23, 0.5, 2],
            up=[-0.63, 0.45, -0.63],
            zoom=0.7
        )
    
    ccSt['obj'] = mesh1
    cc1 = ContentCreator(ccSt)
    d12 = cc1.renderD(p2)
    
    mesh2 = makeMeshFromRaysDepth(rays2.numpy(), d2)
    ccSt['obj'] = mesh2
    cc2 = ContentCreator(ccSt)
    
    # * plot ground truth depth image vs. estimated
    if show:
        print('Showing ground truth d2, estimated d12, and diff d12 - d2')
        plt.subplot(221)
        plt.imshow(d2) # ground truth
        plt.subplot(222)
        plt.imshow(d12) # estiamted
        plt.subplot(223)
        plt.imshow(d12 - d2)
        plt.suptitle('Showing ground truth d2, estimated d12, and diff d12 - d2')
        # plt.colorbar()
        plt.show()
    
    outDir = Path('TestPoseDir')/'testCvgMap'
    outDir.mkdir(exist_ok=True, parents=True)
    # * visualize how p1, p2 extra/inter-polate p3
    d3 = cc.renderD(p3)
    d13 = cc1.renderD(p3)
    d23 = cc2.renderD(p3)
    
    lCvg = makeExtrapolateCoverageMap(d13, d23)
    exCvg = makeCoverageMap(d13, d3)
    uCvg = np.zeros(d13.shape)
    uCvg[d13 != np.inf] = 1
    
    
    cv2.imwrite(str(outDir/f'coverage13_lower.png'), lCvg.astype(np.uint8)*255)
    cv2.imwrite(str(outDir/f'coverage13_exact.png'), exCvg.astype(np.uint8)*255)
    cv2.imwrite(str(outDir/f'coverage13_upper.png'), uCvg.astype(np.uint8)*255)
    cv2.imwrite(str(outDir/f'coverage13_upper-lower.png'), np.abs(uCvg-lCvg).astype(np.uint8)*255)
    print(f'p13: lower = {np.sum(lCvg)/np.size(lCvg)}, exact = {np.sum(exCvg)/np.size(exCvg)}, upper bound = {np.sum(uCvg != np.inf)/np.size(uCvg)}')
    
    print('p12 cover p3 diff map thresholding')
    for i, j in enumerate([-2, -1, -0.5, 0]):
        dd = np.abs(d13 - d23)
        dd[dd > 10 ** j] = np.inf
        plt.subplot(221 + i)
        plt.imshow(dd)
        plt.title(f'thres={10 ** j}')
    plt.suptitle('p12 cover p3 diff map thresholding')
    plt.savefig(str(outDir/'d12_3_thres.png'))
    if show:
        plt.show()
    else:
        plt.clf()
    
    exCvg = makeCoverageMap(d23, d3)
    uCvg = np.zeros(d23.shape)
    uCvg[d23 != np.inf] = 1
    
    cv2.imwrite(str(outDir/f'coverage23_lower.png'), lCvg.astype(np.uint8)*255)
    cv2.imwrite(str(outDir/f'coverage23_exact.png'), exCvg.astype(np.uint8)*255)
    cv2.imwrite(str(outDir/f'coverage23_upper.png'), uCvg.astype(np.uint8)*255)
    cv2.imwrite(str(outDir/f'coverage23_upper-lower.png'), np.abs(uCvg-lCvg).astype(np.uint8)*255)
    print(f'p23: lower = {np.sum(lCvg)/np.size(lCvg)}, exact = {np.sum(exCvg)/np.size(exCvg)}, upper bound = {np.sum(uCvg != np.inf)/np.size(uCvg)}')
        
        
    # * visualize how p1 covers p2
    cvg = makeCoverageMap(d12, d2)
    cv2.imwrite(str(outDir/f'coverage_12_exact.png'), cvg.astype(np.uint8)*255)
    uCvg = np.zeros(d12.shape)
    uCvg[d12 != np.inf] = 1
    cv2.imwrite(str(outDir/f'coverage12_upper.png'), uCvg.astype(np.uint8)*255)
    print(f'exact = {np.sum(cvg)/np.size(cvg)}, upper bound = {np.sum(uCvg != np.inf)/np.size(uCvg)}')
    
    print('p12 diff map thresholding')
    for i, j in enumerate([-2, -1, -0.5, 0]):
        dd = np.abs(d12 - d2)
        dd[dd > 10 ** j] = np.inf
        plt.subplot(221 + i)
        plt.imshow(dd)
        plt.title(f'thres={10 ** j}')
    plt.suptitle('p12 diff map thresholding')
    plt.savefig(str(outDir/'d12_thres.png'))
    if show:
        plt.show()
    else:
        plt.clf()
    
    d1[d1 == np.inf] = np.max(d1 != np.inf)
    d1 = 255 * (d1 / np.max(d1))
    cv2.imwrite(str(outDir/f'd1.png'), d1)
    
    d2[d2 == np.inf] = np.max(d2 != np.inf)
    d2 = 255 * (d2 / np.max(d2))
    cv2.imwrite(str(outDir/f'd2.png'), d2)
    
    d3[d3 == np.inf] = np.max(d3 != np.inf)
    d3 = 255 * (d3 / np.max(d3))
    cv2.imwrite(str(outDir/f'd3.png'), d3)
    
    d12[d12 == np.inf] = np.max(d12 != np.inf)
    d12 = 255 * (d12 / np.max(d12))
    cv2.imwrite(str(outDir/f'd12.png'), d12)
    
    d13[d13 == np.inf] = np.max(d13 != np.inf)
    d13 = 255 * (d13 / np.max(d13))
    cv2.imwrite(str(outDir/f'd13.png'), d13)
    
    d23[d23 == np.inf] = np.max(d23 != np.inf)
    d23 = 255 * (d23 / np.max(d23))
    cv2.imwrite(str(outDir/f'd23.png'), d23)
# * pass
def testCvqEst(thres=10 ** (-0.75), saved=False):
    '''
    Tet lower bound, upper bound and exact evaluation
    '''
    outDir = Path('TestPoseDir') / f'bounds-{thres}'
    outDir.mkdir(exist_ok=True, parents=True)
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['size'] = 50
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)

    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['width'] = ccSt['width'] // 1
    ccSt['height'] = ccSt['height'] // 1
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    cc = ContentCreator(ccSt)

    camSt = CamPlace.GetDefaultSettings()
    # camSt['policy'] = 'interpolation'
    cam = CamPlace(camSt)
    
    sample = next(iter(pfd))
    sample = sample.reshape((-1, 7))
    m, h, P = 0.1, 0.5, sample.shape[0]
    psSt = PoseSplitter.GetDefaultSettings()
    psSt['uniform']['numPart'] = np.around(optProbRatio(m, h, P) * m * P).astype(np.int)
    ps = PoseSplitter(psSt)
    
    print(f'no. partition = {psSt["uniform"]["numPart"]}')
    
    t0 = time.time()
    t1 = time.time()
    for u, allUposes in enumerate(pfd):
        cvgs = []
        ubs = []
        lbs = []
        ind = []
        vlines = []
        count = 0
        cc1, cc2 = None, None
               
        poses = allUposes.reshape((-1, 7))
        indices = ps.splitPose(poses)
        ind.append(indices)
        poses = [poses[indices[i, 0]:indices[i, 1]] for i in range(indices.shape[0])]
        print(f'# segements = {len(poses)}')
        # plt.figure((10, 30))
        for i in range(len(poses)):
            spPoses = poses[i]
            p = cam.localPlace(spPoses)
            
            d = cc.renderD(p)
            eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(p))
            rays = cc.createRays(eye, center, up)
            mesh = makeMeshFromRaysDepth(rays.numpy(), d)
            
            ccSt['obj'] = mesh
            cc1, cc2 = cc2, ContentCreator(ccSt)
            
            # * better retrieve placement index
            for j in range(spPoses.shape[0]):
                dEst = cc2.renderD(spPoses[j]) # d23
                dTrue = cc.renderD(spPoses[j])
                cvg = makeCoverageMap(dEst, dTrue, thres)
                cvgs.append(np.sum(cvg)/np.size(cvg))
                ubs.append(np.sum(dEst != np.inf)/np.size(dEst))
                
                d3 = None
                if cc1 != None and cc2 != None:
                    d13 = cc1.renderD(spPoses[j])
                    d23 = dEst
                    d3 = makeExtrapolateCoverageMap(d13, d23, thres)
                    lbs.append(np.sum(d3)/np.size(d3))
                    print(f'{count} / {ind[-1][-1, -1]} : {lbs[-1]}, {cvgs[-1]}, {ubs[-1]} -> {lbs[-1] <= cvgs[-1] <= ubs[-1]}')
                else:
                    lbs.append(None)
                    print(f'{count} / {ind[-1][-1, -1]} : {lbs[-1]}, {cvgs[-1]}, {ubs[-1]} -> X')
                count += 1
                if saved:
                    if cc1 != None and cc2 != None:
                        cv2.imwrite(str(outDir/f'u0-{u}-{i}-{j}_lower.png'), d3.astype(np.uint8)*255)
                    cv2.imwrite(str(outDir/f'u0-{u}-{i}-{j}_exact.png'), cvg.astype(np.uint8)*255)
                    uCvg = dEst != np.inf
                    cv2.imwrite(str(outDir/f'u0-{u}-{i}-{j}_upper.png'), uCvg.astype(np.uint8)*255)
            t1 = time.time()
            print(f'{t1 - t0}')
            t0 = t1
            
            plt.plot(ubs, marker='x', color='red')
            plt.plot(cvgs, marker='x', color='black')
            plt.plot(lbs, marker='x', color='blue')
            ymin = np.array(lbs)
            if np.size(ymin[ymin != None]) > 0:
                ymin = np.min(ymin[ymin != None])
            else:
                ymin = np.min(cvgs)
            # plt.vlines(vlines, ymin=ymin, ymax=np.max(ubs), colors = 'red', linestyles='dashed')
            # for i in range(len(ind)):
            #     plt.vlines(np.array(ind[i]) + pfdSt['size']*i, ymin=ymin, ymax=1.0, colors = 'green', linestyles='dashed')
            plt.title(f'cvgEst-u0-{u}')
            plt.savefig(str(outDir/f'cvgEst-u0-{u}.png'))
            plt.clf()
# * pass         
def testRenderDTime():
    dir = Path('TestPoseDir')
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['size'] = 50
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['width'] = 860
    ccSt['height'] = 540
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'

    poses = next(iter(pfd))
    p1 = poses[0, 0]
    
    t = time.time()
    times = 100
    # for i in range(times):
    #     cc = ContentCreator(ccSt)
    #     d = cc.renderD(p1)
    #     print(f'render {i} times takes {(time.time()  - t) / (i + 1)} secs')
    
    cc = ContentCreator(ccSt)
    d = cc.renderD(p1)
    rays = cc.createRays([0, 0, 1], [0, 0, 0], [0, 1, 0])
    ccSt['obj'] = makeMeshFromRaysDepth(rays.numpy(), d)
    rays = rays.numpy()
    
    # t = time.time()
    # for i in range(times):
    #     ccSt['obj'] = makeMeshFromRaysDepth(rays, d)
    #     cc = ContentCreator(ccSt)
    # print(f'instantiate {i} times takes {(time.time()  - t) / times} secs on average')
    # t = time.time()
    # for i in range(times):
    #     d = cc.renderD(p1)
    # print(f'render {i} times takes {(time.time()  - t) / times} secs on average')
    # t = time.time()
    # for i in range(times):
    #     makeCoverageMap(d, d, 1e-1)
    # print(f'coverage map {i} times takes {(time.time()  - t) / times} secs on average')
    
    # t = time.time()
    # C = np.ones((240, 240, 240, 160))
    # for i in range(C.shape[0]):
    #     for j in range(C.shape[1]):
    #         C[i, j] = np.random.randn(240, 160)
    # t1 = time.time()
    # print(f'total = {t1 - t}, {(t1-t)/(C.shape[0]*C.shape[1])} on average')
    
    pose = next(iter(pfd))
    pose = pose.reshape((-1, 7))
    pose = pose[:200]
    t = time.time()
    for i in range(pose.shape[0]):
        cc.renderD(pose[i])
    print(f'single rendering takes {(time.time() - t)/pose.shape[0]} on average')
    
    t = time.time()
    cc.renderD(pose)
    print(f'batch rendering takes {(time.time() - t)/pose.shape[0]} on average')
# * pass
def testFovWeights():
    # h, w, sizeY, sizeX, shiftY, shiftX, edgeRatioY, edgeRatioX
    ccSt = ContentCreator.GetDefaultSettings()
    h, w = ccSt['height'], ccSt['width']
    sizeY, sizeX = 0.5, 0.5
    shiftY, shiftX = 0.0, 0.0
    edgeRatioY, edgeRatioX = 2, 2
    
    plt.subplot(221)
    plt.imshow(makeFoveationWeights(h, w, sizeY, sizeX, shiftY, shiftX, edgeRatioY, edgeRatioX))
    plt.subplot(222)
    plt.imshow(makeFoveationWeights(h, w, 0.75, 0.75, shiftY, shiftX, edgeRatioY, edgeRatioX))
    plt.subplot(223)
    plt.imshow(makeFoveationWeights(h, w, sizeY, sizeX, 0.1, 0.1, edgeRatioY, edgeRatioX))
    plt.subplot(224)
    plt.imshow(makeFoveationWeights(h, w, sizeY, sizeX, shiftY, shiftX, 4, 4))
    plt.show()

'''
Phase 3:
Test cloud service provider
'''
# * pass
def testSolver():
    M, H, W, N = 200, 10, 20, 100
    slvrSt = Solver.GetDefaultSettings()
    slvrSt['a'] = np.log(0.4)
    slvrSt['ffrMask'] = makeFoveationWeights(H, W, 0.5, 0.5, 0.0, 0.0, 2, 2)
    solver = Solver(slvrSt)
    C = np.random.randint(0, 2, size=(M, M, H, W))
    
    sol, lb, ub = solver.solve(C, N, 100)
    print(sol, lb, ub)
    
    # for i, comb in enumerate(combinations([i for i in range(M)], N)):
    #     sel = np.zeros((M,), dtype=bool)
    #     sel[list(comb)] = 1
    #     try:
    #         assert solver.opt(sel, C) <= opt
    #     except AssertionError:
    #         print('==========================')
    #         print(sol, solver.opt(sol, C))
    #         print(sel, solver.opt(sel, C))
    #         print(f'diff = {abs(solver.opt(sel, C) - opt)/opt}')
    #         print('==========================')

def testPoseProxy():
    '''
    test low resolution partial scene reconstruction
    
    http://www.open3d.org/docs/0.9.0/tutorial/Basic/visualization.html
    '''
    dir = Path('Trace_FPS50_LEN30')/'FurnishedCabin'
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['size'] = 50
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['width'] = ccSt['width'] // 1
    ccSt['height'] = ccSt['height'] // 1
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    cc = ContentCreator(ccSt)

    # ccSt['width'] = ccSt['width'] // 60
    # ccSt['height'] = ccSt['height'] // 60
    ccSt['width'] = round(3 * 1.7)
    ccSt['height'] = round(3)
    ccProxy = ContentCreator(ccSt)
    
    poses = next(iter(pfd)) # (U, N, 7)
    poses = poses[0]
    pcds = []
    lineSets = []
    for i in range(poses.shape[0]):
        if i % 2 != 0:
            continue
        eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(poses[i]))
        rays = ccProxy.createRays(eye, center, up)
        rays = ccProxy.createRays(eye, center, up)
        ans = ccProxy.castRays(rays)
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
        pcd = o3d.t.geometry.PointCloud(points)
        pcds.append(pcd.to_legacy())
        
        points = o3d.core.append(points, [eye.astype(np.float32)], axis=0)
        lines = [[points.shape[0] - 1, i] for i in range(points.shape[0] - 1)]
        color = np.random.rand(3,)
        colors = [color for i in range(len(lines))]
        lineSet = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points.numpy()),
            lines=o3d.utility.Vector2iVector(lines),
        )
        lineSet.colors = o3d.utility.Vector3dVector(colors)
        lineSets.append(lineSet)
        
    obj = o3d.io.read_triangle_mesh(str(Path('obj')/'FurnishedCabin_Demo.obj'))
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([obj, *pcds, origin, *lineSets],
        front=eye,
        lookat=center,
        up=up,
        zoom=0.7
    )

def testLocalPlacePullBack():
    '''
    test low resolution local placement
    
    http://www.open3d.org/docs/0.9.0/tutorial/Basic/visualization.html
    '''
    dir = Path('Trace_FPS50_LEN30')/'FurnishedCabin'
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['size'] = 50
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['width'] = ccSt['width'] // 1
    ccSt['height'] = ccSt['height'] // 1
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    cc = ContentCreator(ccSt)

    # ccSt['width'] = ccSt['width'] // 60
    # ccSt['height'] = ccSt['height'] // 60
    ccSt['width'] = round(3 * 1.7)
    ccSt['height'] = round(3)
    ccProxy = ContentCreator(ccSt)
    
    poses = next(iter(pfd)) # (U, N, 7)
    poses = poses[1]
    pcds = []
    lineSets = []
    
    pts = []
    dirs = []
    for i in range(poses.shape[0]):
        if i % 10 != 0:
            continue
        eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(poses[i]))
        rays = ccProxy.createRays(eye, center, up)
        rays = ccProxy.createRays(eye, center, up)
        ans = ccProxy.castRays(rays)
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
        pcd = o3d.t.geometry.PointCloud(points)
        pcds.append(pcd.to_legacy())
        
        pts.append(points.numpy())
        dirs.append((rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00).numpy())
        
        points = o3d.core.append(points, [eye.astype(np.float32)], axis=0)
        lines = [[points.shape[0] - 1, i] for i in range(points.shape[0] - 1)]
        color = np.random.rand(3,)
        colors = [[0, 0, 1] for i in range(len(lines))]
        lineSet = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points.numpy()),
            lines=o3d.utility.Vector2iVector(lines),
        )
        lineSet.colors = o3d.utility.Vector3dVector(colors)
        lineSets.append(lineSet) 
    
    camPSt = CamPlace.GetDefaultSettings()
    cam = CamPlace(camPSt)
    # o3d frame
    pts = np.array(pts).reshape((-1, 3))
    dirs = np.array(dirs).reshape((-1, 3))
    avePose = cam.localPlace(poses)
    eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(avePose))
    print(f'e = {eye}, c = {center}, u={up}')
    p0 = torch.from_numpy(eye)
    d0 = torch.from_numpy(center - eye)
    dNorm = pts / np.sqrt(np.sum(pts ** 2, axis=1)).reshape((-1, 1))
    
    
    obj = []
    poses = []
        
    dNorm = torch.from_numpy(dNorm)
    dirs = torch.from_numpy(dirs)
    # dirsNorm = torch.linalg.norm(dirs, dim=1)
    d = torch.from_numpy(pts) 
    k = torch.Tensor([0])
    k.requires_grad_(True)
    optimizer = torch.optim.SGD([k], lr=1e-3)
    for i in range(100):
        optimizer.zero_grad()
        dis = d - (p0 + k*d0).reshape((-1, 3))
        disNorm = torch.linalg.norm(dis, dim=1).detach()
        inner = torch.sum(dis * d0, axis=1)
        # gain = 1/(1 + torch.exp(-2 * 50 * (inner-np.cos(torch.pi/4)))) * 2 - 1 # ! contrast
        gain = inner
        penalty = torch.pow(disNorm, 3) # ! tune
        y = -gain/penalty
        y = torch.sum(y)
        y.backward()
        optimizer.step()
        
    print(k.detach().item())  
    eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(avePose))
    eye = eye + k.detach().item() * d0.detach().numpy()
    center = center + k.detach().item() * d0.detach().numpy()
    rays = cc.createRays(eye, center, up)
    ans = cc.castRays(rays)
    hit = ans['t_hit'].isfinite()
    points2 = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
    points2 = o3d.core.append(points2, [eye.astype(np.float32)], axis=0)
    pcds.append(o3d.t.geometry.PointCloud(points2).to_legacy())
    
    
    rays = ccProxy.createRays(eye, center, up)
    ans = ccProxy.castRays(rays)
    hit = ans['t_hit'].isfinite()
    points2 = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
    points2 = o3d.core.append(points2, [eye.astype(np.float32)], axis=0)
    lines = [[points2.shape[0] - 1, i] for i in range(points2.shape[0] - 1)]
    colors = [[1, 0, 0] for i in range(len(lines))]
    lineSet = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points2.numpy()),
        lines=o3d.utility.Vector2iVector(lines),
    )
    lineSet.colors = o3d.utility.Vector3dVector(colors)
    lineSets.append(lineSet)
    lines = [[points2.shape[0] - 1, i] for i in range(points2.shape[0] - 1)]
    colors = [[1, 0, 0] for i in range(len(lines))]
    lineSet = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points2.numpy()),
        lines=o3d.utility.Vector2iVector(lines),
    )
    lineSet.colors = o3d.utility.Vector3dVector(colors)
        
    eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(avePose))
    rays = ccProxy.createRays(eye, center, up)
    ans = ccProxy.castRays(rays)
    hit = ans['t_hit'].isfinite()
    points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
    points = o3d.core.append(points, [eye.astype(np.float32)], axis=0)
    lines = [[points.shape[0] - 1, i] for i in range(points.shape[0] - 1)]
    colors = [[0, 1, 0] for i in range(len(lines))]
    lineSet = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.numpy()),
        lines=o3d.utility.Vector2iVector(lines),
    )
    lineSet.colors = o3d.utility.Vector3dVector(colors)
    lineSets.append(lineSet)
    
    obj = o3d.io.read_triangle_mesh(str(Path('obj')/'FurnishedCabin_Demo.obj'))
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([obj, *pcds, origin, *lineSets],
        front=eye,
        lookat=center,
        up=up,
        zoom=0.7
    )

def testLocalPlaceMisc(method, plts, placePolicy):
    '''
    test low resolution local placement
    
    http://www.open3d.org/docs/0.9.0/tutorial/Basic/visualization.html
    
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    '''
    dir = Path('Trace_FPS50_LEN30')/'FurnishedCabin'
    pfdSt = PoseFeeder.GetDefaultSettings()
    pfdSt['size'] = 50
    pfdSt['poseDir'] = dir
    pfd = PoseFeeder(pfdSt)
    
    ccSt = ContentCreator.GetDefaultSettings()
    ccSt['width'] = ccSt['width'] // 1
    ccSt['height'] = ccSt['height'] // 1
    ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
    cc = ContentCreator(ccSt)

    # ccSt['width'] = ccSt['width'] // 60
    # ccSt['height'] = ccSt['height'] // 60
    ccSt['width'] = round(3 * 1.7)
    ccSt['height'] = round(3)
    ccProxy = ContentCreator(ccSt)
    
    oDir = Path('testPoseDir')/'localPlaceMisc'
    oDir.mkdir(exist_ok=True)
    
    pcds = []
    lineSets = []
    pposes = next(iter(pfd)) # (U, N, 7)
    for u in range(0, pposes.shape[0], 1):
    # for u in [0, 1, 2]:
        poses = pposes[u]
        
        pts = []
        ptRays = []
        for i in range(poses.shape[0]):
            if i % 10 != 0:
                continue
            if i == 1:
                break
            eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(poses[i]))
            rays = ccProxy.createRays(eye, center, up)
            rays = ccProxy.createRays(eye, center, up)
            ans = ccProxy.castRays(rays)
            hit = ans['t_hit'].isfinite()
            points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
            pcd = o3d.t.geometry.PointCloud(points)
            pcds.append(pcd.to_legacy())
            
            pts.append(points.numpy())
            ptRays.append((rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00).numpy())
            
            points = o3d.core.append(points, [eye.astype(np.float32)], axis=0)
            lines = [[points.shape[0] - 1, i] for i in range(points.shape[0] - 1)]
            colors = [[0, 0, 1] for i in range(len(lines))]
            lineSet = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points.numpy()),
                lines=o3d.utility.Vector2iVector(lines),
            )
            lineSet.colors = o3d.utility.Vector3dVector(colors)
            lineSets.append(lineSet)
            
        
        camPSt = CamPlace.GetDefaultSettings()
        camPSt['policy'] = placePolicy
        cam = CamPlace(camPSt)
        avePoseU = cam.localPlace(poses)
        # o3d frame
        avePoseO = convertUnityPoses7ToO3d7(avePoseU)
        
        if method == '1DoF':
            eye, center, up = convertO3d7ToO3dEyeCenterUp(avePoseO)
            p0 = torch.from_numpy(eye)
            d0 = torch.from_numpy(center - eye)
            k = torch.Tensor([0])
            k.to(device)
            k.requires_grad_(True)
            optimizer = torch.optim.Adam([k], lr=1*1e-5)
        elif method == '6DoF':
            pq = np.zeros((6,))
            rpy = R.from_quat(avePoseO[3:]).as_euler('xyz', degrees=False)
            pq[:3] = avePoseO[:3]
            pq[3:] = rpy[::-1] # stored as yaw, pitch, roll
            pq = torch.from_numpy(pq)
            pq.to(device)
            pq.requires_grad_()
            optimizer = torch.optim.Adam([pq], lr=1e-4)
        elif method == '1+6':
            p = np.zeros((3,))
            q = np.zeros((3,))
            rpy = R.from_quat(avePoseO[3:]).as_euler('xyz', degrees=False)
            p = avePoseO[:3]
            q = rpy[::-1] # stored as yaw, pitch, roll
            p = torch.from_numpy(p)
            q = torch.from_numpy(q.copy())
            p.to(device)
            q.to(device)
            p.requires_grad_()
            q.requires_grad_()
            posOptimizer = torch.optim.Adam([p], lr=1e-2)
            rotOptimizer = torch.optim.Adam([q], lr=1e-5)
        pts = np.array(pts).reshape((-1, 3))
        pts = torch.from_numpy(pts)
        ptRays = np.array(ptRays).reshape((-1, 3))
        ptRays = torch.from_numpy(ptRays)
        ptRaysNorm = torch.linalg.norm(ptRays, dim=1).detach()
        obj = []
        poses = []
        ks = []
        # print(pts.shape, avePoseO[:3].shape)
        for i in range(1000):
            if method == '1DoF':
                disToPts = pts - (p0 + k*d0).reshape((-1, 3))
                disToPtsNorm = torch.linalg.norm(disToPts, dim=1)
                inner = torch.sum(d0 * disToPts, axis=1)
                gain = inner - torch.sum(disToPts * np.cos(torch.pi/4), axis=1)
                # gain = inner
                factor = torch.pow(disToPtsNorm / ptRaysNorm, 3)
                penalty = torch.where(factor >= 1, factor, 1)
                y = torch.sum(-gain/penalty)
                # y = torch.prod(-gain/penalty)
                y.backward()
                obj.append(y.detach().item())
                ks.append(k.detach().item())
                optimizer.step()
                poses.append((p0 + k*d0).detach().numpy())
            elif method == '6DoF':
                optimizer.zero_grad()
                disToPts = pts - pq[:3].reshape((-1, 3))
                disToPtsNorm = torch.linalg.norm(disToPts, dim=1)
                
                # http://msl.cs.uiuc.edu/planning/node102.html, roll, pitch, then yaw
                # front = torch.Tensor([0, 0, 1], dtype=torch.float)
                rotatedFrontX = torch.cos(pq[3]) * torch.sin(pq[4]) * torch.cos(pq[5]) + torch.sin(pq[3]) * torch.sin(pq[5])
                rotatedFrontY = torch.sin(pq[3]) * torch.sin(pq[4]) * torch.cos(pq[5]) - torch.cos(pq[3]) * torch.sin(pq[5])
                rotatedFrontZ = torch.cos(pq[4]) * torch.cos(pq[5])
                
                inner = disToPts[:, 0] * rotatedFrontX + disToPts[:, 1] * rotatedFrontY + disToPts[:, 2] * rotatedFrontZ
                gain = inner - torch.sum(disToPts * np.cos(torch.pi/6), axis=1)
                # gain = inner - torch.sum(disToPts * np.cos(torch.pi/4), axis=1)
                factor = torch.pow(disToPtsNorm / ptRaysNorm, 3)
                penalty = torch.where(factor >= 1, factor, 1)
                # y = torch.sum(-gain/penalty)
                y = torch.prod(-gain/penalty)
                
                y.backward()
                obj.append(y.detach().item())
                optimizer.step()
                poses.append(pq.detach().numpy())
            elif method == '1+6':
                posOptimizer.zero_grad()
                rotOptimizer.zero_grad()
                disToPts = pts - p.reshape((-1, 3))
                disToPtsNorm = torch.linalg.norm(disToPts, dim=1)
                
                # http://msl.cs.uiuc.edu/planning/node102.html, roll, pitch, then yaw
                # front = torch.Tensor([0, 0, 1], dtype=torch.float)
                rotatedFrontX = torch.cos(q[0]) * torch.sin(q[1]) * torch.cos(q[2]) + torch.sin(q[0]) * torch.sin(q[2])
                rotatedFrontY = torch.sin(q[0]) * torch.sin(q[1]) * torch.cos(q[2]) - torch.cos(q[0]) * torch.sin(q[2])
                rotatedFrontZ = torch.cos(q[1]) * torch.cos(q[2])
                
                inner = disToPts[:, 0] * rotatedFrontX + disToPts[:, 1] * rotatedFrontY + disToPts[:, 2] * rotatedFrontZ
                # gain = inner - torch.sum(disToPts * np.cos(torch.pi/6), axis=1)
                gain = inner - torch.sum(disToPts * np.cos(torch.pi/4), axis=1)
                factor = torch.pow(disToPtsNorm / ptRaysNorm, 3)
                penalty = torch.where(factor >= 1, factor, 1)
                # y = torch.sum(-gain/penalty)
                y = torch.sum(-gain/penalty)                    
                y.backward()
                
                obj.append(y.detach().item())
                if i % 2: # position
                    posOptimizer.step()
                else:
                    rotOptimizer.step()
                
                poses.append(np.concatenate([p.detach().numpy(), q.detach().numpy()]))
        
        obj = np.array(obj)
        
        if method == '1DoF':
            pqO = np.zeros((7,))
            pqO[:3] = poses[np.argmin(obj)]
            pqO[3:] = avePoseO[3:]
        elif method == '6DoF':
            pq = poses[np.argmin(obj)]
            pq7 = np.zeros((7,))
            pq7[:3] = pq[:3]
            pq7[3:] = R.from_euler('xyz', pq[3:][::-1], degrees=False).as_quat()
            pqO = pq7
        elif method == '1+6':
            pq = poses[np.argmin(obj)]
            pq7 = np.zeros((7,))
            pq7[:3] = pq[:3]
            pq7[3:] = R.from_euler('xyz', pq[3:][::-1], degrees=False).as_quat()
            pqO = pq7
        else:
            pqO = avePoseO
        
        print(f'ave pose = {avePoseO}')
        print(f'opt pose = {pqO}')
        
        if plts:
            plt.plot(obj)
            plt.savefig(str(oDir/f'{method}-{u}.png'))
            plt.clf()
            if method == '1DoF':
                plt.plot(ks, obj)
                plt.savefig(str(oDir/f'{method}_k-{u}.png'))
                plt.clf()
            
        
        # optimized pose
        eye, center, up = convertO3d7ToO3dEyeCenterUp(pqO)
        rays = cc.createRays(eye, center, up)
        ans = cc.castRays(rays)
        hit = ans['t_hit'].isfinite()
        points2 = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
        points2 = o3d.core.append(points2, [eye.astype(np.float32)], axis=0)
        pcds.append(o3d.t.geometry.PointCloud(points2).to_legacy())   
        rays = ccProxy.createRays(eye, center, up)
        ans = ccProxy.castRays(rays)
        hit = ans['t_hit'].isfinite()
        points2 = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
        points2 = o3d.core.append(points2, [eye.astype(np.float32)], axis=0)
        lines = [[points2.shape[0] - 1, i] for i in range(points2.shape[0] - 1)]
        colors = [[1, 0, 0] for i in range(len(lines))]
        lineSet = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points2.numpy()),
            lines=o3d.utility.Vector2iVector(lines),
        )
        lineSet.colors = o3d.utility.Vector3dVector(colors)
        lineSets.append(lineSet)
        lines = [[points2.shape[0] - 1, i] for i in range(points2.shape[0] - 1)]
        colors = [[1, 0, 0] for i in range(len(lines))]
        lineSet = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points2.numpy()),
            lines=o3d.utility.Vector2iVector(lines),
        )
        lineSet.colors = o3d.utility.Vector3dVector(colors)
        
        # AvePose
        avePoseO = convertUnityPoses7ToO3d7(avePoseU)
        eye, center, up = convertO3d7ToO3dEyeCenterUp(avePoseO)
        rays = ccProxy.createRays(eye, center, up)
        ans = ccProxy.castRays(rays)
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 1.00
        points = o3d.core.append(points, [eye.astype(np.float32)], axis=0)
        lines = [[points.shape[0] - 1, i] for i in range(points.shape[0] - 1)]
        colors = [[0, 1, 0] for i in range(len(lines))]
        lineSet = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points.numpy()),
            lines=o3d.utility.Vector2iVector(lines),
        )
        lineSet.colors = o3d.utility.Vector3dVector(colors)
        lineSets.append(lineSet)
    
    obj = o3d.io.read_triangle_mesh(str(Path('obj')/'FurnishedCabin_Demo.obj'))
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([obj, *pcds, origin, *lineSets])
