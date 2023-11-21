#%%
'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 4.9542741775512695, 2.988436222076416, 5.3653368949890137 ],
			"boundingbox_min" : [ -3.5019974708557129, 0.12037336826324463, -0.79421579837799072 ],
			"field_of_view" : 60.0,
			"front" : [ 0.0090650185518678268, -0.063914004928435283, -0.99791423750373576 ],
			"lookat" : [ 1.0400519480137256, 1.152810501195682, 3.141982378576833 ],
			"up" : [ 0.00066924983229904036, 0.99795540588802756, -0.063910562221938083 ],
			"zoom" : 0.33999999999999969
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
from pathlib import Path
from Common import *
from PoseFeeder import *
from ContentCreator import *
show = True

dir = Path('disocculsion_gen_dir')
pfdSt = PoseFeeder.GetDefaultSettings()
pfdSt['size'] = 1
pfdSt['poseDir'] = dir
pfd = PoseFeeder(pfdSt)

ccSt = ContentCreator.GetDefaultSettings()
ccSt['width'] = ccSt['width'] // 1
ccSt['height'] = ccSt['height'] // 1
ccSt['obj'] = Path('obj')/'FurnishedCabin_Demo.obj'
cc = ContentCreator(ccSt)

for u, _poses in enumerate(pfd):
    poses = _poses
    break
p = poses[0, 0]

d = cc.renderD(p)
# plt.imshow(d)
# plt.show()


pp = convertUnityPoses7ToO3d7(p)
eye, center, up = convertO3d7ToO3dEyeCenterUp(pp)
rays = cc.createRays(eye, center, up)

mesh = makeMeshFromRaysDepth(rays.numpy(), d)

ans = cc.castRays(rays)
hit = ans['t_hit'].isfinite()
points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1)) * 0.95
pcd = o3d.t.geometry.PointCloud(points)
# Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

# mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
o3d.visualization.draw_geometries([pcd.to_legacy(), mesh],
    front=center - eye,
    lookat=center,
    up=up,
    zoom=0.7
)