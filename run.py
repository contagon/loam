#!/usr/bin/env python
import argparse
import os
import glob
import csv
from collections import OrderedDict

import numpy as np
from gtsam import Pose3, Rot3
import open3d as o3d
import loam
import gtsam

"""
########     ###    ########   ######  #### ##    ##  ######   
##     ##   ## ##   ##     ## ##    ##  ##  ###   ## ##    ##  
##     ##  ##   ##  ##     ## ##        ##  ####  ## ##        
########  ##     ## ########   ######   ##  ## ## ## ##   #### 
##        ######### ##   ##         ##  ##  ##  #### ##    ##  
##        ##     ## ##    ##  ##    ##  ##  ##   ### ##    ##  
##        ##     ## ##     ##  ######  #### ##    ##  ######   
"""


def to_sec(sec, nsec):
    """
    Converts a rostime int format (seconds, nano seconds) into a float time (seconds)
    """
    return float(sec) + (nsec * 1e-9)


def parse_keyframes(dataset_dir, keyframe_rate):
    """
    Iterate over all of the scans, mark keyframes based on keyframe_rate
    returns map[(sec, nsec)] -> file path to pcd
    """
    scan_dir = os.path.join(dataset_dir, "scans")
    keyframe_delta = 1.0 / keyframe_rate

    keyframes = OrderedDict()
    last_kf_time = None
    for scan_file in sorted(glob.glob(os.path.join(scan_dir, "*.pcd"))):
        # Format XXX_sec_nsec.pcd
        scan_name = os.path.basename(scan_file)
        _, sec, nsec = scan_name.split(".")[0].split("_")
        sec, nsec = int(sec), int(nsec)
        scan_time = to_sec(sec, nsec)

        if (last_kf_time is None) or ((scan_time - last_kf_time) > keyframe_delta):
            keyframes[(sec, nsec)] = scan_file
            last_kf_time = scan_time
    return keyframes


def parse_groundtruth(dataset_dir):
    """
    Parses all groundtruth poses from the gt_poses.csv file in the dataset
    returns list of [((sec, nsec), pose), ...]
    """
    gt_poses = []
    with open(os.path.join(dataset_dir, "gt_poses.csv")) as csvfile:
        for line in csv.reader(filter(lambda row: row[0] != "#", csvfile)):
            sec, nsec, x, y, z, qx, qy, qz, qw = line
            r = Rot3.Quaternion(float(qw), float(qx), float(qy), float(qz))
            t = np.array([float(x), float(y), float(z)])
            pose = Pose3(r, t)
            gt_poses.append(((int(sec), int(nsec)), pose))
    return gt_poses


def filter_gt_keyframes(gt_poses, keyframes):
    """
    Filters the gt_poses extracting the pose for each keyframe
    returns list of [((sec, nsec), pose), ...]
    """
    keyframe_gt_poses = []
    keyframes_with_gt = set()
    for ts, pose in gt_poses:
        if ts in keyframes.keys():
            keyframe_gt_poses.append((ts, pose))
            keyframes_with_gt.add(ts)

    for ts, _ in keyframes.items():
        if ts not in keyframes_with_gt:
            raise Exception(
                "Keyframe at ({}, {}) missing groundtruth".format(ts[0], ts[1])
            )
    return keyframe_gt_poses


def read_ouster_cloud(file, lidar_params):
    """
    Reads and formats an ouster pointcloud
    """
    # Read the pointcloud from file
    pcd = o3d.io.read_point_cloud(file)
    # Convert to numpy type
    pcd_as_array = np.asarray(pcd.points)
    # Reorganize to row-major
    pcd_as_array = np.concatenate(
        [
            pcd_as_array[i :: lidar_params.scan_lines]
            for i in range(lidar_params.scan_lines)
        ]
    )
    return list(pcd_as_array)


"""
##     ##    ###    #### ##    ## 
###   ###   ## ##    ##  ###   ## 
#### ####  ##   ##   ##  ####  ## 
## ### ## ##     ##  ##  ## ## ## 
##     ## #########  ##  ##  #### 
##     ## ##     ##  ##  ##   ### 
##     ## ##     ## #### ##    ## 
"""


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="The directory containing the dataset to evaluate",
    )
    parser.add_argument(
        "--keyframe_rate",
        "-kr",
        type=float,
        help="The rate of keyframes",
    )
    return parser.parse_args()


def main():
    args = handle_args()
    print("Parsing Keyframes... ")
    keyframes = parse_keyframes(args.dataset_dir, args.keyframe_rate)
    print("Parsing Groundtruth...")
    gt_poses = parse_groundtruth(args.dataset_dir)
    print("Filtering keyframe groundtruth... ")
    keyframe_gt_poses = filter_gt_keyframes(gt_poses, keyframes)
    print("Starting LOAM... ")

    odom_pose = gtsam.Pose3()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    target_pcd = None
    source_pcd = None

    lidar_params = loam.LidarParams(64, 1024, 1.0, 120.0)
    feat_params = loam.FeatureExtractionParams()
    feat_params.edge_feat_threshold = 10000
    feat_params.planar_feat_threshold = 1.0
    for i in range(0, len(keyframe_gt_poses)):
        # Get info about this pose
        stamp_i, pose_i = keyframe_gt_poses[i]
        pcd_i = read_ouster_cloud(keyframes[stamp_i], lidar_params)

        # Get info about the next pose
        stamp_ip1, pose_ip1 = keyframe_gt_poses[i + 1]
        pcd_ip1 = read_ouster_cloud(keyframes[stamp_ip1], lidar_params)

        # Extract the features
        feat_i = loam.extractFeatures(pcd_i, lidar_params, feat_params)
        feat_ip1 = loam.extractFeatures(pcd_ip1, lidar_params, feat_params)

        print(len(feat_i.edge_points), len(feat_i.planar_points))
        # feat_i.edge_points = []
        # feat_ip1.edge_points = []

        i_T_ip1 = loam.registerFeatures(
            source=feat_ip1,
            target=feat_i,
            target_T_source_init=loam.Pose3d.Identity(),
        )

        rel_pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(
                i_T_ip1.rotation.w(),
                i_T_ip1.rotation.x(),
                i_T_ip1.rotation.y(),
                i_T_ip1.rotation.z(),
            ),
            i_T_ip1.translation,
        )
        print(rel_pose)
        quit()

        # Visualize
        if target_pcd is None:
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(pcd_i)
            target_pcd.paint_uniform_color(np.array([0, 0, 1]))
            target_pcd.transform(odom_pose.matrix())

            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(pcd_ip1)
            source_pcd.paint_uniform_color(np.array([0, 1, 0]))
            source_pcd.transform(odom_pose.compose(rel_pose).matrix())
            vis.add_geometry(target_pcd)
            vis.add_geometry(source_pcd)
        else:
            target_pcd.points = o3d.utility.Vector3dVector(pcd_i)
            target_pcd.paint_uniform_color(np.array([0, 0, 1]))
            target_pcd.transform(odom_pose.matrix())

            source_pcd.points = o3d.utility.Vector3dVector(pcd_ip1)
            source_pcd.paint_uniform_color(np.array([0, 0, 1]))
            source_pcd.transform(odom_pose.compose(rel_pose).matrix())

        vis.update_geometry(target_pcd)
        vis.update_geometry(source_pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("temp_figs/%04d.png" % i)

        print(f"finished iteration {i}")

        odom_pose = odom_pose.compose(rel_pose)


if __name__ == "__main__":
    main()
