#!/usr/bin/env python
import argparse
import os
import glob
import csv
from collections import OrderedDict

import numpy as np
from gtsam import Pose3, Rot3
import open3d as o3d
import rerun as rr
import loam
import gtsam
import random

"""
########     ###    ########   ######  #### ##    ##  ######   
##     ##   ## ##   ##     ## ##    ##  ##  ###   ## ##    ##  
##     ##  ##   ##  ##     ## ##        ##  ####  ## ##        
########  ##     ## ########   ######   ##  ## ## ## ##   #### 
##        ######### ##   ##         ##  ##  ##  #### ##    ##  
##        ##     ## ##    ##  ##    ##  ##  ##   ### ##    ##  
##        ##     ## ##     ##  ######  #### ##    ##  ######   
"""


def gtsam2rr(pose: gtsam.Pose3) -> rr.Transform3D:
    return rr.Transform3D(
        mat3x3=pose.rotation().matrix(),
        translation=pose.translation(),
    )


def ate(gt, sol):
    error = 0
    for i in range(len(gt)):
        error += np.linalg.norm(gt[i].translation() - sol[i].translation())
    return error / len(gt)


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
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for line in reader:
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the results",
    )
    parser.add_argument(
        "--percent_edge",
        type=float,
        help="Edge feature % to keep",
        default=1.0,
    )
    parser.add_argument(
        "--percent_planar",
        type=float,
        help="Planar feature % to keep",
        default=1.0,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Number of keyframes to process",
    )
    return parser.parse_args()


def main():
    args = handle_args()

    if args.verbose:
        print("Parsing Keyframes... ")
    keyframes = parse_keyframes(args.dataset_dir, args.keyframe_rate)
    if args.verbose:
        print("Parsing Groundtruth...")
    gt_poses = parse_groundtruth(args.dataset_dir)
    if args.verbose:
        print("Filtering keyframe groundtruth... ")
    keyframe_gt_poses = filter_gt_keyframes(gt_poses, keyframes)
    if args.verbose:
        print("Starting LOAM... ")

    odom_pose = gtsam.Pose3()
    gt = []
    sol = []
    num_edges = []
    num_planar = []

    gt_T_lidar = gtsam.Pose3(
        gtsam.Rot3(0.38268, 0, 0, 0.92388), np.array([-0.08425, -0.025, 0.050188])
    )

    for i in range(len(keyframe_gt_poses)):
        stamp, frame = keyframe_gt_poses[i]
        frame = frame.compose(gt_T_lidar)
        keyframe_gt_poses[i] = (stamp, frame)

    if args.visualize:
        rr.init("loam", spawn=False)
        rr.connect("0.0.0.0:9876")

    lidar_params = loam.LidarParams(64, 1024, 1.0, 120.0)
    feat_params = loam.FeatureExtractionParams()
    feat_params.neighbor_points = 4
    feat_params.number_sectors = 6
    feat_params.max_edge_feats_per_sector = 10
    feat_params.max_planar_feats_per_sector = 50

    feat_params.edge_feat_threshold = 50.0
    feat_params.planar_feat_threshold = 1.0

    feat_params.occlusion_thresh = 0.9
    feat_params.parallel_thresh = 0.01

    reg_params = loam.RegistrationParams()
    reg_params.max_iterations = 80

    length = len(keyframe_gt_poses)
    if args.length and args.length < length:
        length = args.length

    for i in range(0, length):
        # Get info about this pose
        stamp_i, pose_i = keyframe_gt_poses[i]
        pcd_i = read_ouster_cloud(keyframes[stamp_i], lidar_params)

        if i == 0:
            odom_pose = pose_i

        # Get info about the next pose
        stamp_ip1, pose_ip1 = keyframe_gt_poses[i + 1]
        # Move the ground truth pose to the lidar frame
        pcd_ip1 = read_ouster_cloud(keyframes[stamp_ip1], lidar_params)

        # Extract the features
        feat_i = loam.extractFeatures(pcd_i, lidar_params, feat_params)
        feat_ip1 = loam.extractFeatures(pcd_ip1, lidar_params, feat_params)

        # Downsample features
        # edges
        edges = feat_i.edge_points
        feat_i.edge_points = random.sample(edges, int(len(edges) * args.percent_edge))
        edges = feat_ip1.edge_points
        feat_ip1.edge_points = random.sample(edges, int(len(edges) * args.percent_edge))
        # planar
        planar = feat_i.planar_points
        feat_i.planar_points = random.sample(
            planar, int(len(planar) * args.percent_planar)
        )
        planar = feat_ip1.planar_points
        feat_ip1.planar_points = random.sample(
            planar, int(len(planar) * args.percent_planar)
        )

        detail = loam.RegistrationDetail()
        i_T_ip1 = loam.registerFeatures(
            source=feat_ip1,
            target=feat_i,
            target_T_source_init=loam.Pose3d.Identity(),
            detail=detail,
            params=reg_params,
        )
        result = ""
        if detail.termination_type == loam.RegistrationTerminationType.CONVERGED:
            result = "CONVERGED"
        elif detail.termination_type == loam.RegistrationTerminationType.MAX_ITER:
            result = "MAX_ITER"

        if args.verbose:
            print(
                f"iter {i}, edges {len(feat_i.edge_points)}, planar {len(feat_i.planar_points)}, result: {result}"
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

        # Visualize
        if args.visualize and i % 1 == 0:
            # Send in the source map
            rr.set_time_seconds("loam_time", seconds=to_sec(*stamp_i))
            rr.log("source", gtsam2rr(odom_pose))
            edges = np.array(feat_i.edge_points)
            rr.log("source/edges", rr.Points3D(edges, colors=[[255, 0, 0]]))
            planar = np.array(feat_i.planar_points)
            rr.log("source/planar", rr.Points3D(planar, colors=[[0, 255, 0]]))

            # Send in the target map
            temp = odom_pose.compose(rel_pose)
            rr.log("target", gtsam2rr(temp))
            edges = np.array(feat_ip1.edge_points)
            rr.log("target/edges", rr.Points3D(edges, colors=[[100, 0, 0]]))
            planar = np.array(feat_ip1.planar_points)
            rr.log("target/planar", rr.Points3D(planar, colors=[[0, 100, 0]]))

            rr.log("ground_truth", gtsam2rr(pose_ip1))

        odom_pose = odom_pose.compose(rel_pose)

        gt.append(pose_ip1)
        sol.append(odom_pose)
        num_edges.append(len(feat_i.edge_points))
        num_planar.append(len(feat_i.planar_points))

    # print(
    #     f"Length: {length}, ERROR: {ate(gt, sol)}, ThreshEdge: {args.threshold_edge}, ThreshPlanar: {args.threshold_planar}, AvgEdges: {np.mean(num_edges)}, AvgPlanar: {np.mean(num_planar)}"
    # )
    print(
        length,
        ate(gt, sol),
        args.percent_edge,
        args.percent_planar,
        np.mean(num_edges),
        np.mean(num_planar),
    )


if __name__ == "__main__":
    main()
