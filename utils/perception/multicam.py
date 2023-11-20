import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import pprint
import numpy as np
import cv2
import copy
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.calib_utils.solver import Solver
from utils.vision.rs_streamer import RealsenseStreamer, MarkSearch
import open3d as o3d
from polymetis import RobotInterface

GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01

def compute_pcd_offline(serial_nos, bgr_images, depth_images):
    if os.path.exists('cache/calib/transforms.npy'):
        transforms = np.load('cache/calib/transforms.npy', allow_pickle=True).item()
    if os.path.exists('cache/calib/icp_tf.npy'):
        icp_tf = np.load('cache/calib/icp_tf.npy', allow_pickle=True).item()
    else:
        icp_tf = None

    raw_points = []
    merged_points = []
    merged_colors = []

    icp_tfs = []
    cam_ids = []
    for idx, serial_no in enumerate(serial_nos):
        bgr_image, depth_img = bgr_images[serial_no], depth_images[serial_no]
        
        denoised_idxs = denoise(depth_img)

        K = np.load(f'cache/cam_K/{serial_no}.npy')
        tf = transforms[serial_no]['tcr']

        points_3d = deproject(depth_img, K, tf)
        raw_points.append(copy.deepcopy(points_3d))
        colors = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).reshape(points_3d.shape)/255.

        points_3d = points_3d[denoised_idxs]
        colors = colors[denoised_idxs]

        idxs = crop(points_3d)
        points_3d = points_3d[idxs]
        colors = colors[idxs]

        merged_points.append(points_3d)
        merged_colors.append(colors)

        if idx > 0:
            cam_ids.append(serial_no)
            if icp_tf is not None:
                icp_tfs.append(icp_tf[serial_no])

    pcd_merged = merge_pcls(merged_points, merged_colors, tfs=icp_tfs, cam_ids=cam_ids, visualize=True)
    return pcd_merged, raw_points

def deproject(depth_image, K, tf = np.eye(4), base_units=-3):
    depth_image = depth_image*(10**base_units) # convert mm to m (TODO)

    h,w = depth_image.shape
    row_indices = np.arange(h)
    col_indices = np.arange(w)
    pixel_grid = np.meshgrid(col_indices, row_indices)
    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
    depth_arr = np.tile(depth_image.flatten(), [3, 1])
    points_3d = depth_arr * np.linalg.inv(K).dot(pixels_homog)

    points_3d_transf = np.vstack((points_3d, np.ones([1,points_3d.shape[1]])))
    points_3d_transf = ((tf.dot(points_3d_transf)).T)[:, 0:3]

    return points_3d_transf

def project(robot_point, K, TRC, img_dims):
    width, height = img_dims
    xr,yr,zr = robot_point
    xc,yc,zc = TRC.dot(np.array([xr,yr,zr,1]))
    u,v,depth = K.dot(np.array([xc,yc,zc]))
    u /= depth
    v /= depth
    px = np.array([int(u), int(v)])
    return px

def transform_points(tf, points_3d):
    points_3d = points_3d.T
    points_3d_transf = np.vstack((points_3d, np.ones([1,points_3d.shape[1]])))
    points_3d_transf = ((tf.dot(points_3d_transf)).T)[:, 0:3]
    print(points_3d_transf.shape)
    return points_3d_transf

def crop(pcd):
    min_bound = [0.1,-0.35,0.10]
    max_bound = [1.1,0.3,0.5]

    idxs = np.logical_and(np.logical_and(
                  np.logical_and(pcd[:,0] > min_bound[0], pcd[:,0] < max_bound[0]),
                  np.logical_and(pcd[:,1] > min_bound[1], pcd[:,1] < max_bound[1])),
                  np.logical_and(pcd[:,2] > min_bound[2], pcd[:,2] < max_bound[2]))

    return idxs

def rescale_pcd(pcd, scale=1.0):
    pcd_temp = copy.deepcopy(pcd)
    points = np.asarray(pcd.points)
    new_points = points*scale
    pcd_temp.points = o3d.utility.Vector3dVector(new_points)
    return pcd_temp

def align_pcds(pcds, tfs=None, cam_ids=None, visualize=False):
    target_pcd = pcds[0]

    threshold = 0.02
    trans_init = np.eye(4)
    scale = 2.
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                                                 relative_rmse=0.000001,
                                                                 max_iteration=50)

    aligned_pcds = [target_pcd]

    target_pcd = rescale_pcd(target_pcd, scale=scale)

    for idx, source_pcd in enumerate(pcds[1:]):

        source_pcd = rescale_pcd(source_pcd, scale=scale)

        if tfs is None or not len(tfs):
            reg_p2p = o3d.pipelines.registration.registration_icp(
                            source_pcd, target_pcd, threshold, trans_init,
                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)

            tf = reg_p2p.transformation
            cam_id = cam_ids[idx]
            if os.path.exists('cache/calib/icp_tf.npy'):
                icp_tf = np.load('cache/calib/icp_tf.npy', allow_pickle=True).item()
            else:
                icp_tf = dict()
            icp_tf[cam_id] = tf 

            print('Saving', icp_tf)
            np.save('cache/calib/icp_tf.npy', icp_tf)
        else:
            tf = tfs[idx]

        source_pcd_transf = copy.deepcopy(source_pcd)
        source_pcd_transf.transform(tf)
        source_pcd_transf = rescale_pcd(source_pcd_transf, 1/scale)

        aligned_pcds.append(source_pcd_transf)

    return aligned_pcds

def merge_pcls(pcls, colors, tfs=None, cam_ids=None, origin=[0,0,0], visualize=True):
    pcds = []
    for pcl, color in zip(pcls, colors):
        # Check if pcl needs to be converted into array
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.colors = o3d.utility.Vector3dVector(color)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.2)
        pcd = pcd.select_by_index(ind)
        pcds.append(pcd)

    pcds = align_pcds(pcds, tfs=tfs, cam_ids=cam_ids, visualize=visualize)

    pcd_combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        pcd_combined += pcd

    if visualize:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=origin)
        pcds.append(mesh_frame)
        o3d.visualization.draw_geometries(pcds)

    return pcd_combined

def denoise(depth_img):
    max_val = np.amax(depth_img)
    min_val = np.amin(depth_img)
    normalized = depth_img - min_val / (max_val - min_val)
    normalized_vis = cv2.normalize(normalized, 0, 255, cv2.NORM_MINMAX)
    idxs = np.where(normalized_vis.ravel() > 0)[0]
    return idxs

class MultiCam:
    def __init__(self, serial_nos):
        self.cameras = []
        for serial_no in serial_nos:
            self.cameras.append(RealsenseStreamer(serial_no))

        self.transforms = None

        if os.path.exists('cache/calib/transforms.npy'):
            self.transforms = np.load('cache/calib/transforms.npy', allow_pickle=True).item()
        if os.path.exists('cache/calib/icp_tf.npy'):
            self.icp_tf = np.load('cache/calib/icp_tf.npy', allow_pickle=True).item()
        else:
            self.icp_tf = None

    def project_robot_waypoint(self, waypoint):
        waypoints_proj = {cam.serial_no:None for cam in self.cameras}
        for cam in self.cameras:
            tcr = self.transforms[cam.serial_no]['tcr']
            tf = np.linalg.inv(np.vstack((tcr, np.array([0,0,0,1]))))[:3]
            pixel = project(waypoint, cam.K, tf, (cam.width, cam.height))
            waypoints_proj[cam.serial_no] = pixel
        return waypoints_proj

    def take_bgr(self):
        bgr_images = {cam.serial_no:None for cam in self.cameras}

        for cam in self.cameras:
            bgr_image = cam.capture_rgb()
            bgr_images[cam.serial_no] = bgr_image         

        return bgr_images 

    def take_raw_bgrd(self):
        bgr_images = {cam.serial_no:None for cam in self.cameras}
        depth_images = {cam.serial_no:None for cam in self.cameras}
        for cam in self.cameras:
            for i in range(5):
                _, bgr_image, depth_frame, depth_img_vis = cam.capture_rgbd()
            # _, bgr_image, depth_frame, depth_img_vis = cam.capture_rgbd()

            bgr_images[cam.serial_no] = bgr_image            
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_images[cam.serial_no] = depth_img

        return bgr_images, depth_images

    def compute_pcd(self, bgr_images, depth_images):
        raw_points = []
        merged_points = []
        merged_colors = []

        icp_tfs = []
        cam_ids = []
        for idx, cam in enumerate(self.cameras):
            bgr_image, depth_img = bgr_images[cam.serial_no], depth_images[cam.serial_no]
            
            denoised_idxs = denoise(depth_img)

            tf = self.transforms[cam.serial_no]['tcr']

            points_3d = deproject(depth_img, cam.K, tf)
            raw_points.append(copy.deepcopy(points_3d))
            colors = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).reshape(points_3d.shape)/255.

            points_3d = points_3d[denoised_idxs]
            colors = colors[denoised_idxs]

            idxs = crop(points_3d)
            points_3d = points_3d[idxs]
            colors = colors[idxs]

            merged_points.append(points_3d)
            merged_colors.append(colors)

            if idx > 0:
                cam_ids.append(cam.serial_no)
                if self.icp_tf is not None:
                    icp_tfs.append(self.icp_tf[cam.serial_no])

        pcd_merged = merge_pcls(merged_points, merged_colors, tfs=icp_tfs, cam_ids=cam_ids, visualize=False)
        return pcd_merged, raw_points

    def take_bgrd(self, visualize=True):
        bgr_images = {cam.serial_no:None for cam in self.cameras}
        depth_images = {cam.serial_no:None for cam in self.cameras}
        raw_points = []
        merged_points = []
        merged_colors = []

        icp_tfs = []
        cam_ids = []
        for idx, cam in enumerate(self.cameras):
            for i in range(5):
                _, bgr_image, depth_frame, depth_img_vis = cam.capture_rgbd()
            # _, bgr_image, depth_frame, depth_img_vis = cam.capture_rgbd()

            bgr_images[cam.serial_no] = bgr_image            
            
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_images[cam.serial_no] = depth_img
            denoised_idxs = denoise(depth_img)

            tf = self.transforms[cam.serial_no]['tcr']

            points_3d = deproject(depth_img, cam.K, tf)
            raw_points.append(copy.deepcopy(points_3d))
            colors = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).reshape(points_3d.shape)/255.

            points_3d = points_3d[denoised_idxs]
            colors = colors[denoised_idxs]

            idxs = crop(points_3d)
            points_3d = points_3d[idxs]
            colors = colors[idxs]

            merged_points.append(points_3d)
            merged_colors.append(colors)

            if idx > 0:
                cam_ids.append(cam.serial_no)
                if self.icp_tf is not None:
                    icp_tfs.append(self.icp_tf[cam.serial_no])
            
        pcd_merged = merge_pcls(merged_points, merged_colors, tfs=icp_tfs, cam_ids=cam_ids, visualize=visualize)

        return bgr_images, pcd_merged, raw_points, depth_images

    def calibrate_back_cam(self, robot=None):
        if not os.path.exists('cache/calib'):
            os.mkdir('cache/calib')
            curr_calib = {}
        else:
            curr_calib = np.load('cache/calib/transforms.npy', allow_pickle=True).item()

        self.marker_search = MarkSearch()

        self.solver = Solver()

        def gen_calib_waypoints(start_pos):
            waypoints = []
            for i in np.linspace(-0.05,0.05,3):
                for j in np.linspace(-0.1,0.1,3):
                    for k in np.linspace(-0.20,0.0,3):
                        waypoints.append(start_pos + torch.Tensor([i,j,k]))
            return waypoints
    

        if robot is None:
            robot = RobotInterface(
                ip_address="172.16.0.1",
                enforce_version=False
            )
        robot.go_home()

        # Get ee pose
        ee_pos, ee_quat = robot.pose_ee()
        ee_euler = R.from_quat(ee_quat).as_euler('xyz', degrees=True)

        waypoints = gen_calib_waypoints(ee_pos)

        calib_quats = []
        z_offsets = [-180]
        for z_off in z_offsets:
            calib_euler = ee_euler + np.array([-35,35,z_off])
            calib_quat = R.from_euler('xyz', calib_euler, degrees=True).as_quat()
            calib_quats.append(calib_quat)

        waypoints_rob = []
        waypoints_cam = {c.serial_no:[] for c in self.cameras}

        state_log = robot.set_ee_pose(
            position=ee_pos, orientation=calib_quat
        )

        for waypoint in waypoints:
            successful_waypoint = True

            intermed_waypoints = {}
            for idx, cam in enumerate(self.cameras):
                calib_quat = calib_quats[idx]
                state_log = robot.set_ee_pose(
                    position=waypoint, orientation=calib_quat, time_to_go=2.0
                )

                _, rgb_image, depth_frame, depth_img = cam.capture_rgbd()
                (u,v), vis = self.marker_search.find_marker(rgb_image)
                if u is None:
                    successful_waypoint = False
                    break

                waypoint_cam = cam.deproject((u,v), depth_frame)
                intermed_waypoints[cam.serial_no] = waypoint_cam

            if successful_waypoint:
                waypoints_rob.append([waypoint[0], waypoint[1], waypoint[2]])
                for k in intermed_waypoints:
                    waypoints_cam[k].append(intermed_waypoints[k])

        pprint.pprint(waypoints_cam)
        pprint.pprint(waypoints_rob)

        transforms = {}

        waypoints_rob = np.array(waypoints_rob)

        for cam in self.cameras:
            waypoints_cam_curr = waypoints_cam[cam.serial_no]
            waypoints_cam_curr = np.array(waypoints_cam_curr)
            trc, tcr = self.solver.solve_transforms(waypoints_rob, waypoints_cam_curr)
            transforms[cam.serial_no] = {'trc': trc, 'tcr':tcr}
            

        curr_calib.update(transforms)
        np.save('cache/calib/transforms.npy', curr_calib)

    def calibrate_front_cams(self, robot=None):
        if not os.path.exists('cache/calib'):
            os.mkdir('cache/calib')
            curr_calib = {}
        else:
            curr_calib = np.load('cache/calib/transforms.npy', allow_pickle=True).item()

        self.marker_search = MarkSearch()

        self.solver = Solver()

        def gen_calib_waypoints(start_pos):
            waypoints = []
            for i in np.linspace(-0.15,0.05,3):
                for j in np.linspace(-0.1,0.1,3):
                    for k in np.linspace(-0.20,0.0,3):
                        waypoints.append(start_pos + torch.Tensor([i,j,k]))
            return waypoints
    

        if robot is None:
            robot = RobotInterface(
                ip_address="172.16.0.1",
                enforce_version=False
            )
        robot.go_home()

        # Get ee pose
        ee_pos, ee_quat = robot.pose_ee()
        ee_euler = R.from_quat(ee_quat).as_euler('xyz', degrees=True)

        waypoints = gen_calib_waypoints(ee_pos)

        calib_quats = []
        z_offsets = [-60,70]
        for z_off in z_offsets:
            calib_euler = ee_euler + np.array([-20,20,z_off])
            calib_quat = R.from_euler('xyz', calib_euler, degrees=True).as_quat()
            calib_quats.append(calib_quat)

        waypoints_rob = []
        waypoints_cam = {c.serial_no:[] for c in self.cameras}

        for waypoint in waypoints:
            successful_waypoint = True

            intermed_waypoints = {}
            for idx, cam in enumerate(self.cameras):
                calib_quat = calib_quats[idx]
                state_log = robot.set_ee_pose(
                    position=waypoint, orientation=calib_quat, time_to_go=2.0
                )

                _, rgb_image, depth_frame, depth_img = cam.capture_rgbd()
                (u,v), vis = self.marker_search.find_marker(rgb_image)
                if u is None:
                    successful_waypoint = False
                    break

                waypoint_cam = cam.deproject((u,v), depth_frame)
                intermed_waypoints[cam.serial_no] = waypoint_cam

            if successful_waypoint:
                waypoints_rob.append([waypoint[0], waypoint[1], waypoint[2]])
                for k in intermed_waypoints:
                    waypoints_cam[k].append(intermed_waypoints[k])

        pprint.pprint(waypoints_cam)
        pprint.pprint(waypoints_rob)

        transforms = {}

        waypoints_rob = np.array(waypoints_rob)

        for cam in self.cameras:
            waypoints_cam_curr = waypoints_cam[cam.serial_no]
            waypoints_cam_curr = np.array(waypoints_cam_curr)
            trc, tcr = self.solver.solve_transforms(waypoints_rob, waypoints_cam_curr)
            transforms[cam.serial_no] = {'trc': trc, 'tcr':tcr}

        curr_calib.update(transforms)

        np.save('cache/calib/transforms.npy', curr_calib)

    # For action primitive
    def robot_fingertip_pos_to_ee(self, fingertip_pos, ee_quat):
        HOME_QUAT = np.array([ 0.9367,  0.3474, -0.0088, -0.0433])
        FINGERTIP_OFFSET = np.array([0,0,-0.095])
        home_euler = R.from_quat(HOME_QUAT).as_euler('zyx', degrees=True)

        ee_euler = R.from_quat(ee_quat).as_euler('zyx', degrees=True)

        offset_euler = ee_euler - home_euler

        fingertip_offset_euler = offset_euler * [1,-1,1]
        fingertip_transf = R.from_euler('zyx', fingertip_offset_euler, degrees=True)
        fingertip_offset = fingertip_transf.as_matrix() @ FINGERTIP_OFFSET
        fingertip_offset[2] -= FINGERTIP_OFFSET[2]

        ee_pos = fingertip_pos - fingertip_offset
        return ee_pos

    def project_merged_pcd_to_image(self, camera_idx, bbox, merged_pcd):
        # Step 1: Load the merged 3D point cloud
        cam = self.cameras[camera_idx]
        merged_point_cloud = np.asarray(merged_pcd.points)
        # Step 2: Load the image from camera A
        _, bgr_image, depth_frame, depth_img_vis = cam.capture_rgbd()
        depth_img = np.asanyarray(depth_frame.get_data())
        print(bgr_image.shape)
        # Step 3: Get the coordinates of the bounding box in the image
        # For example, (u1, v1) and (u2, v2) are the top-left and bottom-right corner coordinates
        v1, u1, v2, u2 = bbox
        bounding_box_2d = (u1, v1, u2, v2)

        # Step 4: Project the 2D bounding box onto the 3D point cloud
        def project_2d_to_3d(u, v, camera_matrix, depth):
            inv_camera_matrix = np.linalg.inv(camera_matrix)
            ray_direction = np.dot(inv_camera_matrix, [u, v, 1])
            ray_direction /= np.linalg.norm(ray_direction)
            ray_direction *= depth
            return ray_direction

        # Get the depth values at the corners of the bounding box in image A
        depth_image_A = depth_img  # Depth image from camera A
        depth_u1_v1 = depth_image_A[v1, u1]
        depth_u2_v2 = depth_image_A[v2, u2]

        # Project the bounding box corners into 3D rays
        camera_matrix_A = cam.K  # Intrinsic camera matrix (3x3) for camera A
        ray_u1_v1 = project_2d_to_3d(u1, v1, camera_matrix_A, depth_u1_v1)
        ray_u2_v2 = project_2d_to_3d(u2, v2, camera_matrix_A, depth_u2_v2)

        # Step 5: Find the 3D points within the projected bounding box
        def find_points_within_bounding_box(point_cloud, ray_start, ray_end):
            # Calculate the direction and length of the ray
            ray_direction = ray_end - ray_start
            ray_length = np.linalg.norm(ray_direction)
            ray_direction /= ray_length

            # Calculate the dot product of the points with the ray direction
            dot_products = np.dot(point_cloud - ray_start, ray_direction)

            # Find the indices of points that are within the bounding box
            indices_within_bounding_box = np.where((dot_products >= 0) & (dot_products <= ray_length))[0]

            # Get the 3D points within the bounding box
            points_within_bounding_box = point_cloud[indices_within_bounding_box]

            return points_within_bounding_box

        # Usage:
        # Apply the above steps to get the 3D points within the bounding box
        points_within_bounding_box = find_points_within_bounding_box(merged_point_cloud, ray_u1_v1, ray_u2_v2)
        return points_within_bounding_box
