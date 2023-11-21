import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json
from Common import *
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ContentCreator:
    '''
    Implementing a content creator
    
    RayCasting:
    http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html
    TriangleMesh:
    http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
    
    '''
    def __init__(self, settings) -> None:
        self.settings = dict(settings)
        if isinstance(self.settings['obj'], Path):
            self.mesh = o3d.io.read_triangle_mesh(str(self.settings['obj']))
        elif isinstance(self.settings['obj'], o3d.geometry.TriangleMesh):
            self.mesh = self.settings['obj']
        else:
            print('[CC] not recognized settings["obj"]')
            return
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        self.resetCost()
    @classmethod
    def GetDefaultSettings(cls) -> dict:
        return {
            'width': 960, # render width
            'height': 540, # render height
            'fov': 90,
            'renderD': 1, # cost of rendering depth
            'renderRGB': 1, # cost of rendering depth
            'obj': Path('.') # path to obj files or a mesh in open3d
        }
    def getSettings(self):
        return dict(self.settings)
    def resetCost(self):
        self.cost = {
            'renderD': 0,
            'renderRGB': 0,
        }
    def getCost(self):
        return dict(self.cost)
    def summary(self, outDir: Path):
        outDir.mkdir(exist_ok=True)
        with open(outDir/'ContentCreator.json', 'w') as f:
            self.settings['obj'] = str(self.settings['obj'])
            json.dump(self.settings, f)
    def createRays(self, eye, center, up):
        '''
        * eye, center, up are in O3d frame, created by convertO3d7ToO3dEyeCenterUp()
        '''
        rays = self.scene.create_rays_pinhole(
            fov_deg=self.settings['fov'],
            center=center, #  The point the camera is looking at with shape {3}
            eye=eye, # The position of the camera with shape {3}.
            up=up, # The up-vector with shape {3}
            width_px=self.settings['width'],
            height_px=self.settings['height'],
        )
        return rays
    def castRays(self, rays):
        '''
        * rays: created by self.createRays()
        '''
        return self.scene.cast_rays(rays)
    def render(self, pose, tp: str):
        '''
        * pose: unity frame (7,) or (-1, 7)
        (-1, 7) large scale rendering seems to be no faster than call (7,) one by one
        * tp: any keys in "ans" of http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html
        '''
        if len(pose.shape) == 1:
            p = convertUnityPoses7ToO3d7(pose).reshape((7,))
            # eye = p[:3]
            # center = np.array([0, 0, 1])
            # up = np.array([0, 1, 0])
            # r = R.from_quat(p[3:])
            # center = r.apply(center) + eye
            # up = r.apply(up)
            eye, center, up = convertO3d7ToO3dEyeCenterUp(p)
            # print(f'eye = {eye}, center={center}, up={up}')
            rays = self.createRays(eye, center, up)
            ans = self.scene.cast_rays(rays)
            ans = ans[tp].numpy()
            return ans[::-1, ::-1]
        else:
            assert len(pose.shape) == 2
            eyeCenterUps = [convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(pose[i]).reshape((7,))) for i in range(pose.shape[0])]
            rays = o3d.core.Tensor.zeros((pose.shape[0], self.settings['height'], self.settings['width'], 6))
            for i, (eye, center, up) in enumerate(eyeCenterUps):
                rays[i] = self.createRays(eye, center, up)
            # rays = o3d.core.Tensor([self.createRays(eye, center, up) for (eye, center, up) in eyeCenterUps])
            ans = self.scene.cast_rays(rays)
            ans = ans[tp].numpy()
            ans = ans[..., ::-1, ::-1]
            return ans
    def renderD(self, pose) -> np.array:
        '''
        return depth image from a pose
        * pose: (7,) unity frame
        '''
        if len(pose.shape) == 1:
            self.cost['renderD'] += 1
            return self.render(pose, 't_hit')
        else: # must be shape (-1, 7)
            assert len(pose.shape) == 2
            self.cost['renderD'] += pose.shape[0]
            return self.render(pose, 't_hit')
    def renderRGB(self, pose):
        '''
        return rgb image from a pose
        * pose: (7,) unity frame
        This function is simulation only, won't return rgb
        '''
        if len(pose.shape) == 1:
            self.cost['renderRGB'] += 1
        else: # must be shape (-1, 7)
            self.cost['renderRGB'] += pose.shape[0]
    def renderObjId(self, pose) -> np.array:
        '''
        * pose: unity frame
        # ! incorrect result from imported mesh
        '''
        return NotImplemented
        # return self.render(pose, 'geometry_ids')

def makeExtrapolateCoverageMap(d13: np.array, d23: np.array, thres=1e-1):
    '''
    * p1: (7,) pose at d1 in unity frame
    * d1: (H, W) depth map rendered at pose1
    * p2: (7,) pose at d2 in unity frame
    * d2: (H, W) depth map rendered at pose2
    * p3: (7,) or (-1, 7) poses to be extra/inter-polated in unity frame
    * d13: depth map using pose 1 to estimate 3
    * d23: depth map using pose 2 to estimate 3
    * ccSt: json, settings for the content creator
    * thres: threshold to reject false depth
    
    return lower bound coverage map
    '''
    # consistency check
    consistent = makeCoverageMap(d13, d23, thres)
    return np.logical_and.reduce([consistent, d13 != np.inf, d23 != np.inf])