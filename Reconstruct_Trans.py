#%%
'''
This transforms the reconstructed pcd to align with the ground truth mesh
(manually transformation for trying to calculate objective metrics for comparing two pcds)
'''
#%%
import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import copy
from scipy.spatial.transform import Rotation as R

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    o3d.visualization.draw_geometries([source_temp, target_temp, origin],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556]) 
def draw_registration_result_nt(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    # o3d.visualization.draw_geometries([source_temp, target_temp, origin])
    o3d.visualization.draw_geometries([source_temp, target_temp]) 
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray(np.eye(4).astype(np.float64))
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result
def refine_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.asarray(np.eye(4).astype(np.float64)),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.asarray(np.eye(4).astype(np.float64)),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result
def cabinTransform(source):
    tmat = np.eye(4).astype(np.float64)
    tmat[1, 3] = -3.5
    source.transform(tmat)

    mat = R.from_euler(seq='xy', angles=[150, -100], degrees=True).as_matrix()
    tmat = np.eye(4).astype(np.float64)
    tmat[:3, :3] = mat
    source.transform(tmat)

    tmat = np.eye(4).astype(np.float64)
    tmat[2, 3] = -4
    tmat[1, 3] = 0.5
    source.transform(tmat)

    # tmat = np.eye(4).astype(np.float64)
    # source.transform(tmat)

    tmat = np.eye(4).astype(np.float64)
    tmat[0, 0] = 0.53
    tmat[1, 1] = 0.6
    tmat[2, 2] = 0.55
    source.transform(tmat)
    return source
def roomSourceTransform(source):
    tmat = np.eye(4).astype(np.float64)
    tmat[1, 3] = -2.5
    source.transform(tmat)
    
    mat = R.from_euler(seq='xy', angles=[170, 0], degrees=True).as_matrix()
    tmat = np.eye(4).astype(np.float64)
    tmat[:3, :3] = mat
    source.transform(tmat)
    
    tmat = np.eye(4).astype(np.float64)
    tmat[1, 3] = 1
    source.transform(tmat)

    mat = R.from_euler(seq='yx', angles=[-45, -10], degrees=True).as_matrix()
    tmat = np.eye(4).astype(np.float64)
    tmat[:3, :3] = mat
    source.transform(tmat)
    
    tmat = np.eye(4).astype(np.float64)
    tmat[2, 3] = -20
    source.transform(tmat)

    # tmat = np.eye(4).astype(np.float64)
    # tmat[0, 0] = 0.53
    # tmat[1, 1] = 0.6
    # tmat[2, 2] = 0.55
    # source.transform(tmat)
    return source
def roomTargetTransform(source):
    '''
    {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 17.381772994995117, 11.710968017578125, 19.808254241943359 ],
			"boundingbox_min" : [ -12.257238388061523, -16.594343185424805, -11.840169906616211 ],
			"field_of_view" : 60.0,
			"front" : [ -0.77672530882175206, 0.62875368108186536, 0.036967596104671305 ],
			"lookat" : [ 2.886278102057525, -2.5870377584834139, 2.2685612426178068 ],
			"up" : [ 0.59395322661776961, 0.75073617513812518, -0.28916216891100899 ],
			"zoom" : 0.13999999999999979
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
    '''
    tmat = np.eye(4).astype(np.float64)
    tmat[0, 3] = 3.5
    tmat[2, 3] = -42.5
    source.transform(tmat)
    return source
def bigroomSourceTransform(source):   
    mat = R.from_euler(seq='xy', angles=[170, 0], degrees=True).as_matrix()
    tmat = np.eye(4).astype(np.float64)
    tmat[:3, :3] = mat
    source.transform(tmat)
    
    tmat = np.eye(4).astype(np.float64)
    tmat[1, 3] = 3
    source.transform(tmat)

    mat = R.from_euler(seq='yx', angles=[135, 0], degrees=True).as_matrix()
    tmat = np.eye(4).astype(np.float64)
    tmat[:3, :3] = mat
    source.transform(tmat)
    
    tmat = np.eye(4).astype(np.float64)
    tmat[1, 3] = 0.2
    tmat[2, 3] = -9
    tmat[1, 3] = 0.7
    source.transform(tmat)

    tmat = np.eye(4).astype(np.float64)
    tmat[0, 0] = 0.8
    tmat[1, 1] = 0.7
    tmat[2, 2] = 0.8
    source.transform(tmat)
    return source
def smallroomSourceTransform(source):
    '''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 7.6760750145383554, 5.6569976283135937, 8.0711750920690246 ],
			"boundingbox_min" : [ -8.4777271222716113, -3.3555307311584581, -19.548170649266268 ],
			"field_of_view" : 60.0,
			"front" : [ -0.80541016669925114, 0.56500484833773845, -0.17912002884193046 ],
			"lookat" : [ -0.24269408149950264, 1.2062166372693752, -0.43834462964476262 ],
			"up" : [ 0.57636332498165144, 0.81706845184127874, -0.01429904268792993 ],
			"zoom" : 0.080000000000000002
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
    '''
    mat = R.from_euler(seq='xz', angles=[-10, 180], degrees=True).as_matrix()
    tmat = np.eye(4).astype(np.float64)
    tmat[:3, :3] = mat
    source.transform(tmat)
    
    tmat = np.eye(4).astype(np.float64)
    tmat[1, 3] = 5
    source.transform(tmat)
    
    mat = R.from_euler(seq='y', angles=[-115], degrees=True).as_matrix()
    tmat = np.eye(4).astype(np.float64)
    tmat[:3, :3] = mat
    source.transform(tmat)

    
    tmat = np.eye(4).astype(np.float64)
    tmat[2, 3] = 5
    source.transform(tmat)

    tmat = np.eye(4).astype(np.float64)
    tmat[0, 0] = 0.5
    tmat[1, 1] = 0.5
    tmat[2, 2] = 0.5
    source.transform(tmat)
    
    tmat = np.eye(4).astype(np.float64)
    tmat[0, 3] = 1
    source.transform(tmat)
    return source
#%%
# ! Cabin
# * done
voxel_size = 0.05
source = 'Reconstruction\FurnishedCabin/fused.ply'
target = 'Scenes\FurnishedCabin\scene_NOGUARD.obj'
source = o3d.io.read_point_cloud(source)
source = cabinTransform(source)
nPoints = np.asarray(source.points).shape[0]
target = o3d.io.read_triangle_mesh(target)
target = target.sample_points_uniformly(number_of_points=nPoints)

# # ! heavy
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
# source, target, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
# draw_registration_result_nt(source, target)

# result_ransac = execute_global_registration(source, target,
#                                             source_fpfh, target_fpfh,
#                                             voxel_size)
# print(f'ransac: {result_ransac}')
# print(f'ransac: {result_ransac.transformation}')
# draw_registration_result(source, target, result_ransac.transformation)

# result_icp = refine_registration(source, target, voxel_size)
# print(result_icp)
# print(result_icp.transformation)
# draw_registration_result(source, target, result_icp.transformation)

T = np.array([
    [ 9.99999574e-01, -9.21549848e-04, -4.92008897e-05, -2.12293940e-04],
    [ 9.21563502e-04,  9.99999537e-01,  2.78225363e-04, -1.99550343e-02],
    [ 4.89444684e-05, -2.78270586e-04,  9.99999960e-01, -2.36114857e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
])
estSource = o3d.geometry.PointCloud(source)
estSource.transform(T)
draw_registration_result_nt(estSource, target)
'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 6.7063747187952991, 5.6010169982910165, 20.830090462741261 ],
			"boundingbox_min" : [ -26.231258932240774, -9.4805146361373431, -19.600803813614196 ],
			"field_of_view" : 60.0,
			"front" : [ 0.63715456081114974, 0.25395851894261889, 0.72769439759697363 ],
			"lookat" : [ 0.30324323924548596, 0.74868927111878603, -1.08837250137958 ],
			"up" : [ -0.17698826246681082, 0.96713461145759017, -0.18255354904727372 ],
			"zoom" : 0.080000000000000002
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
o3d.visualization.draw_geometries([estSource]) 
# o3d.io.write_point_cloud('Reconstruction\FurnishedCabin/fused_align.ply', estSource)
#%%
# * done
# ! bigroom
voxel_size = 0.05
source = 'Reconstruction\ScifiTraceBigroom/fused.ply'
target = 'Scenes\ScifiTraceBigroom\scene_NOGUARD.obj'
source = o3d.io.read_point_cloud(source)
source = bigroomSourceTransform(source)
nPoints = np.asarray(source.points).shape[0]
target = o3d.io.read_triangle_mesh(target)
target = target.sample_points_uniformly(number_of_points=nPoints)
target = roomTargetTransform(target)
draw_registration_result_nt(source, target)

# ! heavy
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
# source, target, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
# # draw_registration_result_nt(source, target)

# result_ransac = execute_global_registration(source, target,
#                                             source_fpfh, target_fpfh,
#                                             voxel_size)
# print(f'ransac: {result_ransac}')
# print(f'ransac: {result_ransac.transformation}')
# draw_registration_result(source, target, result_ransac.transformation)

# result_icp = refine_registration(source, target, voxel_size)
# print(result_icp)
# print(result_icp.transformation)
# draw_registration_result(source, target, result_icp.transformation)

# T = result_icp.transformation

T = np.array([
    [9.99999982e-01, -1.88660835e-04, -2.61045101e-05,  4.51935262e-03],
    [1.88652760e-04,  9.99999934e-01, -3.08960648e-04, -1.86900907e-02],
    [2.61627971e-05,  3.08955717e-04,  9.99999952e-01, -2.21994064e-03],
    [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
])
estSource = o3d.geometry.PointCloud(source)
estSource.transform(T)
'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 7.6760750145383554, 5.6569976283135937, 8.0711750920690246 ],
			"boundingbox_min" : [ -8.4777271222716113, -3.3555307311584581, -19.548170649266268 ],
			"field_of_view" : 60.0,
			"front" : [ -0.91769994431304291, 0.34669799062367751, -0.19397761599046015 ],
			"lookat" : [ 0.93031551107913879, 0.49992350211943726, -6.9064252271041457 ],
			"up" : [ 0.3345796287739633, 0.93775058817246348, 0.093168161899200422 ],
			"zoom" : 0.16
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
# draw_registration_result_nt(estSource, target)
o3d.visualization.draw_geometries([estSource]) 
# o3d.io.write_point_cloud('Reconstruction\ScifiTracelBigroom/fused_align.ply', estSource)
#%%
# * done
# ! smallroom
voxel_size = 0.05
source = 'Reconstruction\ScifiTraceSmallroom/fused.ply'
target = 'Scenes\ScifiTraceSmallroom\scene_NOGUARD.obj'
source = o3d.io.read_point_cloud(source)
source = smallroomSourceTransform(source)
nPoints = np.asarray(source.points).shape[0]
target = o3d.io.read_triangle_mesh(target)
target = target.sample_points_uniformly(number_of_points=nPoints)
target = roomTargetTransform(target)
draw_registration_result_nt(source, target)

# ! heavy
# # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
# source, target, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
# # # draw_registration_result_nt(source, target)

# # result_ransac = execute_global_registration(source, target,
# #                                             source_fpfh, target_fpfh,
# #                                             voxel_size)
# # print(f'ransac: {result_ransac}')
# # print(f'ransac: {result_ransac.transformation}')
# # draw_registration_result(source, target, result_ransac.transformation)

# result_icp = refine_registration(source, target, voxel_size)
# print(result_icp)
# print(result_icp.transformation)
# draw_registration_result(source, target, result_icp.transformation)

# T = result_icp.transformation

T = np.array([
    [ 9.99999748e-01, 3.73581453e-04, 6.03312089e-04, -2.31450218e-02],
    [-3.74192986e-04, 9.99999416e-01, 1.01383216e-03, -1.57795922e-02],
    [-6.02932988e-04,-1.01405766e-03, 9.99999304e-01, -5.76447243e-03],
    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,  1.00000000e+00],
])
estSource = o3d.geometry.PointCloud(source)
estSource.transform(T)
'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 7.6760750145383554, 5.6569976283135937, 8.0711750920690246 ],
			"boundingbox_min" : [ -8.4777271222716113, -3.3555307311584581, -19.548170649266268 ],
			"field_of_view" : 60.0,
			"front" : [ 0.69211573985799157, 0.31369172734354139, 0.65005638435219626 ],
			"lookat" : [ 1.2293405465250744, 0.70699569638419835, 2.2089347701519544 ],
			"up" : [ -0.27179032035176898, 0.94759714820216689, -0.16789778581685832 ],
			"zoom" : 0.12000000000000001
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
o3d.visualization.draw_geometries([estSource]) 
# o3d.io.write_point_cloud('Reconstruction\ScifiTracelSmallroom/fused_align.ply', estSource)
#%%
# ! both Room failed
# ! Place in the wrong place
# ! room level misplacement

# voxel_size = 0.05
# source = 'Reconstruction\ScifiTracelroom/fused.ply'
# target = 'Scenes\ScifiTraceBigroom\scene_NOGUARD.obj'
# source = o3d.io.read_point_cloud(source)
# source = roomSourceTransform(source)
# nPoints = np.asarray(source.points).shape[0]
# target = o3d.io.read_triangle_mesh(target)
# target = target.sample_points_uniformly(number_of_points=nPoints)
# target = roomTargetTransform(target)
# draw_registration_result_nt(source, target)

# ! heavy
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
# source, target, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
# draw_registration_result_nt(source, target)

# result_ransac = execute_global_registration(source, target,
#                                             source_fpfh, target_fpfh,
#                                             voxel_size)
# print(f'ransac: {result_ransac}')
# print(f'ransac: {result_ransac.transformation}')
# draw_registration_result(source, target, result_ransac.transformation)

# result_icp = refine_registration(source, target, voxel_size)
# print(result_icp)
# print(result_icp.transformation)
# draw_registration_result(source, target, result_icp.transformation)

# cabinT = np.array([
    
# ])
# estSource = o3d.geometry.PointCloud(source)
# estSource.transform(cabinT)
# draw_registration_result_nt(estSource, target)
# o3d.io.write_point_cloud('Reconstruction\ScifiTracelroom/fused_align.ply', estSource)