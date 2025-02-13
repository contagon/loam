#pragma once

#include "loam/deskew.h"
#include "loam/geometry.h"

namespace loam {
template <template <typename> class Accessor, typename PointType, template <typename> class Alloc>
std::vector<Eigen::Vector3d> deskewConstantVelocity(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                                    const std::vector<double>& rel_time_stamps, Eigen::Vector3d vel_rot,
                                                    Eigen::Vector3d vel_trans) {
  std::vector<Eigen::Vector3d> transformed_points;
  transformed_points.reserve(input_scan.size());
  for (size_t i = 0; i < input_scan.size(); i++) {
    Eigen::Vector3d pt = pointToEigen<Accessor>(input_scan[i]);

    // Short circuit the zero case
    if (pt.x() == 0.0 && pt.y() == 0.0 && pt.z() == 0.0) {
      transformed_points.push_back(Eigen::Vector3d::Zero());
    } else {
      double stamp = rel_time_stamps[i];
      Eigen::Quaterniond rot(Eigen::AngleAxisd(stamp * vel_rot.norm(), vel_rot.normalized()));
      Pose3d pose(rot, stamp * vel_trans);

      Eigen::Vector3d pt_transformed = pose.act(pt);
      transformed_points.push_back(pt_transformed);
    }
  }

  return transformed_points;
}

// TODO: This is SO3 x R3 interpolation, do I need full SE3 to get the screw action?
template <template <typename> class Accessor, typename PointType, template <typename> class Alloc>
std::vector<Eigen::Vector3d> deskewInterpolate(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                               const std::vector<double>& rel_time_stamps, Pose3d pose_start,
                                               Pose3d pose_end) {
  // Guess the time period from the stamps
  double max_time = *std::max_element(rel_time_stamps.begin(), rel_time_stamps.end());
  if (max_time == 0) {
    throw std::runtime_error("All time stamps are zero");
  }

  // Compute velocities from the two poses
  Eigen::Vector3d vel_trans = (pose_end.translation - pose_start.translation) / max_time;
  // TODO: Verify I did this in the right order
  Eigen::AngleAxisd rot_diff = Eigen::AngleAxisd(pose_start.rotation.inverse() * pose_end.rotation);
  Eigen::Vector3d vel_rot = rot_diff.angle() * rot_diff.axis() / max_time;

  return deskewConstantVelocity<Accessor, PointType, Alloc>(input_scan, rel_time_stamps, vel_rot, vel_trans);
}

template <template <typename> class Accessor, typename PointType, template <typename> class Alloc>
std::vector<Eigen::Vector3d> deskewImu(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                       const std::vector<double>& point_time_stamps, const std::vector<Pose3d> poses,
                                       const std::vector<double>& imu_time_stamps) {
  std::vector<Eigen::Vector3d> transformed_points;
  transformed_points.reserve(input_scan.size());
  for (size_t i = 0; i < input_scan.size(); i++) {
    Eigen::Vector3d pt = pointToEigen<Accessor>(input_scan[i]);

    // Short circuit the zero case
    if (pt.x() == 0.0 && pt.y() == 0.0 && pt.z() == 0.0) {
      transformed_points.push_back(Eigen::Vector3d::Zero());
    } else {
      double stamp = point_time_stamps[i];
      Pose3d pose = interpolateList(poses, imu_time_stamps, stamp);

      Eigen::Vector3d pt_transformed = pose.act(pt);
      transformed_points.push_back(pt_transformed);
    }
  }

  return transformed_points;
}
}  // namespace loam