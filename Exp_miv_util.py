from pathlib import Path
import numpy as np
import ffmpeg
import cv2
import math
from scipy.spatial.transform import Rotation as R
import json
import re
from Common import *

UNITY_FPS = 50

def convertRaw2Depth16(w, h, fns, outYuvWithOutDim, f2d):
    '''
    * w: width
    * h: height
    * fns: filenames to convert to depth Yuv, must be in order
    * outYuvWithOutDim: output filename without _wxh.yuv suffix
    * f2d: function that converts values in fns to true depth

    Example:
    f2d = fDepthPlannarFactory(1000)
    fns = list((Path('Output')/'test0').glob('d_sv0_*.raw'))[100:120]
    fns = sorted(fns, key=lambda x: int(re.findall('d_sv0_(.*).raw', x.name)[0]))
    convertRaw2Depth16(1080, 720, fns, 'a_', f2d)
    '''
    allDepth = []
    for fn in fns:
        data = np.fromfile(fn, dtype=np.float32)
        data = data.reshape((h, w, 4))
        data = data[::-1, :, 0]
        data = f2d(data)
        allDepth.append(data)
    zmin, zmax = np.min(allDepth), np.max(allDepth)
    with open(f'{str(outYuvWithOutDim)}_{w}x{h}_yuv420p16le.yuv', mode='wb') as f:
        for depth in allDepth:
            depth_16bit = (((1/depth-1/zmax) / (1/zmin-1/zmax)) * 65535)
            depth_16bit = depth_16bit.astype(np.int16)
            depth_16bit = np.append(depth_16bit, np.full(int(depth_16bit.size/2), 32768, dtype=np.int16))
            f.write(depth_16bit.tobytes())
    return zmin, zmax

def convertPng2Tex(fns, outYuvWithOutDim):
    '''
    * args are same as convertRaw2Depth16
    # https://github.com/kkroening/ffmpeg-python/tree/master/examples
    Example:
    fns = list((Path('Output')/'test0').glob('rgb_sv0_*.png'))[100:120]
    fns = sorted(fns, key=lambda x: int(re.findall('rgb_sv0_(.*).png', x.name)[0]))
    convertPng2Tex(fns, 'b_')
    '''
    h, w = cv2.imread(str(fns[0])).shape[:2]
    with open(f'{outYuvWithOutDim}_{w}x{h}_yuv420p10le.yuv', mode='wb') as f:
        for fn in fns:
            out, _ = (
                ffmpeg
                .input(fn)
                .output('pipe:', format='rawvideo', pix_fmt='yuv420p10le')
                .run(capture_stdout=True)
            )
            f.write(out)

def fDepthPlannarFactory(far):
    '''
    far: far plane set by camera
    '''
    def f2d(data):
        return data * far
    return f2d

def truncatePose(inFns, outFns, start, end):
    '''
    * inFns: input csv, fmt=t,x,y,z,qx,qy,qz,qw
    * outFns: output csv, fmt=t,x,y,z,qx,qy,qz,qw
    assume fixed record interval, outputs [startFrame, endFrame)
    raise if any of inFns cannot complete [startFrame, endFrame)

    Example:
    outputDir = Path('TraceTemp')
    inFns = list(Path('Trace').glob('*.csv'))
    outFns = [str(outputDir/fn.name) for fn in inFns]
    truncatePose(inFns, outFns, 100, 120)
    '''
    poses = [np.loadtxt(fn, skiprows=1, delimiter=',') for fn in inFns]
    poses = [poses[i][start:end] for i in range(len(poses))]
    poses = np.array(poses)
    for pose, fn in zip(poses, outFns):
        np.savetxt(fn, pose, fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')
# def truncatePoseLastSampleFromDir(inDir, outDir, numSamples):
#     '''
#     * inDir: input directory, containing pose*.csv of fmt=t,x,y,z,qx,qy,qz,qw
#     * outDir: output directory, output selected samples of pose*.csv
#     raise if any of file that does not have enough samples
#     '''
#     inFns = inDir.glob('pose*.csv')
#     poses = [np.loadtxt(fn, skiprows=1, delimiter=',') for fn in inFns]
#     poses = [poses[i][-numSamples:] for i in range(len(poses))]
#     poses = np.array(poses)
#     for pose, fn in zip(poses, inFns):
#         np.savetxt(outDir/fn.name, pose, fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')
def truncatePoseDir2Dir(inDir: Path, outDir: Path, start, end):
    '''
    * inDir: input directory contains pose*.csv, fmt=t,x,y,z,qx,qy,qz,qw
    * outDir: same as above but for output
    * start: start index to select
    * end: end index to select (exclusive)
    '''
    inPoses = list(inDir.glob('pose*.csv'))
    inPoses = sorted(inPoses, key=lambda x: int(re.findall('pose(.*).csv', x.name)[0]))
    outDir.mkdir(exist_ok=True, parents=True)
    # (outDir/'TargetView').mkdir(exist_ok=True, parents=True)
    poseLens = []
    for inPose in inPoses:
        pose = np.loadtxt(inPose, skiprows=1, delimiter=',')
        pose = pose[start:end, ...]
        np.savetxt(str(outDir/inPose.name), pose, fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')
        poseLens.append(pose.shape[0])
    if np.unique(np.array(poseLens)).size > 1:
        print('Warning: not all pose has the same length')
        print(f'poseLen = {poseLens}')        

def truncatePoseDir2DirFolded(inDir: Path, outDir: Path, start, end, nFolds: int):
    '''
    * inDir: input directory contains pose*.csv, fmt=t,x,y,z,qx,qy,qz,qw
    * outDir: same as above but for output
    * start: start index to select
    * end: end index to select (exclusive)
    * nFolds: number of folds per pose trajectory
        each trajectory will be of length (end - start) / nFolds
    '''
    inPoses = list(inDir.glob('pose*.csv'))
    inPoses = sorted(inPoses, key=lambda x: int(re.findall('pose(.*).csv', x.name)[0]))
    outDir.mkdir(exist_ok=True, parents=True)
    # (outDir/'TargetView').mkdir(exist_ok=True, parents=True)
    poses = []
    poseLens = []
    for inPose in inPoses:
        pose = np.loadtxt(inPose, skiprows=1, delimiter=',')
        pose = pose[start:end, ...]
        poses.append(pose)
        # np.savetxt(str(outDir/inPose.name), pose, fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')
        poseLens.append(pose.shape[0])
    assert np.unique(np.array(poseLens)).size == 1
    print(f'poseLen = {poseLens}')
    poses = np.array(poses)
    assert poses.shape[1] % nFolds == 0
    szPerFold = poses.shape[1] // nFolds
    for i in range(poses.shape[0]):
        for f in range(nFolds):
            fold = poses[i, szPerFold*f:szPerFold*(f+1)]
            np.savetxt(str(outDir/f'pose{i*nFolds + f}.csv'), fold, fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')

# def convertUnityPoses7ToMIVCoord(unityPoses):
#     unityPoses = unityPoses.reshape((-1, 7))
#     # x, y, z, yaw, pitch, roll
#     ret = np.zeros((unityPoses.shape[0], 6))
    
#     ret[:, 0] = unityPoses[:, 2]
#     ret[:, 1] = -unityPoses[:, 0]
#     ret[:, 2] = unityPoses[:, 1]
    
#     q02 = R.from_quat(unityPoses[:, 3:])
      
#     # MIV's Roll, Pitch, then Yaw
#     rot = q02.as_euler('zxy', degrees=True)
#     ret[:, 3] = -rot[:, 2] # y
#     ret[:, 4] = rot[:, 1] # x
#     ret[:, 5] = -rot[:, 0] # z
    
#     return ret

def generateTMIVInputsStaticSceneFromDir(camPlaceDir, contentName, f2d):
    '''
        camPlaceDir: camera placement directory, in which there are
            sv*.csv, d_sv*_0.raw, rgb_sv*_0.png, cameraParam.json
        contentName: any
        f2d: function that converts *.raw to depth value used by TMIV
        ->outputs:
            miv.json
            sv*_texture_{w}x{h}_yuv420p10le.yuv
            sv*_depth_{w}x{h}_yuv420p16le.yuv
        return json object
    '''
    with open(camPlaceDir/'cameraParam.json') as f:
        camParam = json.load(f)
    svCsvs = list(camPlaceDir.glob('sv*.csv'))
    svCsvs = sorted(svCsvs, key=lambda x: int(re.findall('sv(.*).csv', x.name)[0]))
    # t is omitted
    svPoses = [np.loadtxt(fn, skiprows=1, delimiter=',').reshape((8,))[1:] for fn in svCsvs]
    allDepth = []
    h, w = camParam['height'], camParam['width']
    for svCsv in svCsvs:
        # each cam only shoot once
        data = np.fromfile(str(camPlaceDir/f'd_{svCsv.stem}_0.raw'), dtype=np.float32)
        data = data.reshape((h, w, 4))
        data = data[::-1, :, 0]
        data = f2d(data)
        allDepth.append(data)
    zmin, zmax = np.min(allDepth), np.max(allDepth)
    # convert depth yo yuv
    for svCsv, depth in zip(svCsvs, allDepth):
        with open(camPlaceDir/f'{svCsv.stem}_depth_{w}x{h}_yuv420p16le.yuv', mode='wb') as f:
            depth_16bit = (((1/depth-1/zmax) / (1/zmin-1/zmax)) * 65535)
            depth_16bit = depth_16bit.astype(np.int16)
            depth_16bit = np.append(depth_16bit, np.full(int(depth_16bit.size/2), 32768, dtype=np.int16))
            f.write(depth_16bit.tobytes())
    # convert png to yuv
    for svCsv in svCsvs:
        with open(camPlaceDir/f'{svCsv.stem}_texture_{w}x{h}_yuv420p10le.yuv', mode='wb') as f:
            out, _ = (
                ffmpeg
                .input(str(camPlaceDir/f'rgb_{svCsv.stem}_0.png'))
                .output('pipe:', format='rawvideo', pix_fmt='yuv420p10le')
                .run(capture_stdout=True)
            )
            f.write(out)
    # generate camera param json
    camera_parameter = {}
    camera_parameter['Version'] = '4.0'
    camera_parameter["BoundingBox_center"] = [0, 0, 0]
    camera_parameter["Fps"] = UNITY_FPS
    camera_parameter["Content_name"] = contentName
    camera_parameter["Frames_number"] = 1
    camera_parameter["lengthsInMeters"] = True
    camera_parameter["sourceCameraNames"] = [svCsv.stem for svCsv in svCsvs]
    camera_parameter["cameras"] = []
    for svCsv, svPose in zip(svCsvs, svPoses):
        camera = {}
        camera["BitDepthColor"] = 10
        camera["BitDepthDepth"] = 16
        camera["Name"] = svCsv.stem
        camera["Depth_range"] = [float(zmin), float(zmax)]
        camera["DepthColorSpace"] = "YUV420"
        camera["ColorSpace"] = "YUV420"
        MIV_camera_pose = convertUnityPoses7ToMIVCoord(svPose).reshape((-1,))
        camera["Position"] = list(MIV_camera_pose[:3])
        camera["Rotation"] = list(MIV_camera_pose[3:])
        camera["Resolution"] = [w, h]
        camera["Projection"] = "Perspective"
        camera["HasInvalidDepth"] = False
        camera["Depthmap"] = 1
        camera["Background"] = 0
        
        # F = w / (2 * tan(FOV/2))
        # Use horizontal Fov in calculation, vertical Fov is determined automatically by aspect ratio
        camera["Focal"] = [
            camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180)), camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180))
        ]
        # w / 2, h / 2
        camera["Principle_point"] = [
            camera["Resolution"][0]/2, camera["Resolution"][1]/2
        ]        
        camera_parameter["cameras"].append(camera)
    
    # @@ why ?
    viewport_parameter = camera_parameter["cameras"][0].copy()
    viewport_parameter["Name"] = "viewport"
    viewport_parameter["Position"] = [0.0, 0.0, 0.0]
    viewport_parameter["Rotation"] = [0.0, 0.0, 0.0]
    viewport_parameter["HasInvalidDepth"] = True
    camera_parameter["cameras"].append(viewport_parameter)
    with open(camPlaceDir/'miv.json', 'w') as f:
        json.dump(camera_parameter, f)

def generateTMIVUsersStaticSceneFromDir(userDir):
    '''
        userDir must have pose*.csv, rgb_pose*_*.png, cameraParam.json
        ->outputs:
            pose*_texture_{w}x{h}_yuv420p10le.yuv
            miv_pose*.csv
    '''
    userPoseFns = list(userDir.glob('pose*.csv'))
    userPoseFns = sorted(userPoseFns, key=lambda x: int(re.findall('pose(.*).csv', x.name)[0]))
    # t is omitted
    userPoses = [np.loadtxt(fn, skiprows=1, delimiter=',').reshape((-1, 8))[:, 1:] for fn in userPoseFns]
    with open(userDir/'cameraParam.json') as f:
        camParam = json.load(f)
    h, w = camParam['height'], camParam['width']
    for userPoseFn in userPoseFns: # for each user
        rgbs = list(userDir.glob(f'rgb_{userPoseFn.stem}_*.png'))
        rgbs = sorted(rgbs, key=lambda x: int(re.findall(f'rgb_{userPoseFn.stem}_(.*).png', x.name)[0]))
        with open(userDir/f'{userPoseFn.stem}_texture_{w}x{h}_yuv420p10le.yuv', mode='wb') as f:
            for rgb in rgbs:
                out, _ = (
                    ffmpeg
                    .input(str(rgb))
                    .output('pipe:', format='rawvideo', pix_fmt='yuv420p10le')
                    .run(capture_stdout=True)
                )
                f.write(out)
    # convert pose
    for userPose, userPoseFn in zip(userPoses, userPoseFns):
        mivPoses = convertUnityPoses7ToMIVCoord(userPose)
        np.savetxt(str(userDir/f'miv_{userPoseFn.stem}.csv'), mivPoses, fmt='%.4f', header='X,Y,Z,Yaw,Pitch,Roll', comments='', delimiter=',')
            
def generateTMIVInputsStaticSceneFromDirCandidates(cddDir, contentName, f2d):
    '''
        cddDir: camera placement directory, in which there are
            sv*.csv, d_sv*_*.raw, rgb_sv*_*.png, cameraParam.json
        contentName: any
        f2d: function that converts *.raw to depth value used by TMIV
        ->outputs:
            miv.json
            sv*_*_texture_{w}x{h}_yuv420p10le.yuv
            sv*_*_depth_{w}x{h}_yuv420p16le.yuv
        return json object
    '''
    with open(cddDir/'cameraParam.json') as f:
        camParam = json.load(f)
    svCsvs = list(cddDir.glob('sv*.csv'))
    svCsvs = sorted(svCsvs, key=lambda x: int(re.findall('sv(.*).csv', x.name)[0]))
    # t is omitted
    svPoses = [np.loadtxt(fn, skiprows=1, delimiter=',').reshape((-1, 8))[:, 1:] for fn in svCsvs]
    h, w = camParam['height'], camParam['width']
    for group, svCsv in enumerate(svCsvs):
        allDepth = []
        for i in range(len(list(cddDir.glob(f'd_{svCsv.stem}_*.raw')))):
            data = np.fromfile(str(cddDir/f'd_{svCsv.stem}_{i}.raw'), dtype=np.float32)
            data = data.reshape((h, w, 4))
            data = data[::-1, :, 0]
            data = f2d(data)
            allDepth.append(data)
        # convert depth yo yuv
        with open(cddDir/f'{svCsv.stem}_depth_{w}x{h}_yuv420p16le.yuv', mode='wb') as f:
            for i in range(len(list(cddDir.glob(f'd_{svCsv.stem}_*.raw')))):
                depth = allDepth[i]
                zmin, zmax = np.min(allDepth), np.max(allDepth)
                depth_16bit = (((1/depth-1/zmax) / (1/zmin-1/zmax)) * 65535)
                depth_16bit = depth_16bit.astype(np.int16)
                depth_16bit = np.append(depth_16bit, np.full(int(depth_16bit.size/2), 32768, dtype=np.int16))
                f.write(depth_16bit.tobytes())
        # convert png to yuv
        with open(cddDir/f'{svCsv.stem}_texture_{w}x{h}_yuv420p10le.yuv', mode='wb') as f:
            out, _ = (
                ffmpeg
                .input(str(cddDir/f'rgb_{svCsv.stem}_0.png'))
                .output('pipe:', format='rawvideo', pix_fmt='yuv420p10le')
                .run(capture_stdout=True)
            )
            f.write(out)
        # generate camera param json
        camera_parameter = {}
        camera_parameter['Version'] = '4.0'
        camera_parameter["BoundingBox_center"] = [0, 0, 0]
        camera_parameter["Fps"] = UNITY_FPS
        camera_parameter["Content_name"] = contentName
        camera_parameter["Frames_number"] = 1
        camera_parameter["lengthsInMeters"] = True
        camera_parameter["sourceCameraNames"] = [svCsv.stem for svCsv in svCsvs]
        camera_parameter["cameras"] = []
        for i, svPose in enumerate(svPoses[group]):
            camera = {}
            camera["BitDepthColor"] = 10
            camera["BitDepthDepth"] = 16
            camera["Name"] = f'{svCsv.stem}_{i}'
            camera["Depth_range"] = [float(zmin), float(zmax)]
            camera["DepthColorSpace"] = "YUV420"
            camera["ColorSpace"] = "YUV420"
            MIV_camera_pose = convertUnityPoses7ToMIVCoord(svPose).reshape((-1,))
            camera["Position"] = list(MIV_camera_pose[:3])
            camera["Rotation"] = list(MIV_camera_pose[3:])
            camera["Resolution"] = [w, h]
            camera["Projection"] = "Perspective"
            camera["HasInvalidDepth"] = False
            camera["Depthmap"] = 1
            camera["Background"] = 0
            
            # F = w / (2 * tan(FOV/2))
            # Use horizontal Fov in calculation, vertical Fov is determined automatically by aspect ratio
            camera["Focal"] = [
                camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180)), camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180))
            ]
            # w / 2, h / 2
            camera["Principle_point"] = [
                camera["Resolution"][0]/2, camera["Resolution"][1]/2
            ]        
            camera_parameter["cameras"].append(camera)
        
        # @@ why ?
        viewport_parameter = camera_parameter["cameras"][0].copy()
        viewport_parameter["Name"] = "viewport"
        viewport_parameter["Position"] = [0.0, 0.0, 0.0]
        viewport_parameter["Rotation"] = [0.0, 0.0, 0.0]
        viewport_parameter["HasInvalidDepth"] = True
        camera_parameter["cameras"].append(viewport_parameter)
        with open(cddDir/f'miv_{group}.json', 'w') as f:
            json.dump(camera_parameter, f)