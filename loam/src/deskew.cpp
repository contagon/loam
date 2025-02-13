#include "loam/deskew.h"

namespace loam {
Pose3d interpolate(const Pose3d pose_start, const Pose3d pose_end, const double ratio) {
  Eigen::Vector3d delta_trans = ratio * (pose_end.translation - pose_start.translation);
  Eigen::AngleAxisd rot_diff = Eigen::AngleAxisd(pose_start.rotation.inverse() * pose_end.rotation);
  Eigen::Vector3d delta_rot = ratio * rot_diff.angle() * rot_diff.axis();

  Eigen::Vector3d trans = pose_start.translation + delta_trans;
  Eigen::Quaterniond rot = pose_start.rotation * Eigen::AngleAxisd(delta_rot.norm(), delta_rot.normalized());
  return Pose3d(rot, trans);
}

Pose3d interpolateList(const std::vector<Pose3d>& poses, const std::vector<double>& stamps, const double point_stamp) {
  size_t end = 0;
  while (end < stamps.size() && stamps[end] < point_stamp) {
    end++;
  }

  if (end == 0) {
    return poses.front();
  } else if (end == stamps.size()) {
    return poses.back();
  }

  size_t start = end - 1;
  const double ratio = (point_stamp - stamps[start]) / (stamps[end] - stamps[start]);

  // Sanity check our ratio
  if (ratio < 0.0 || ratio > 1.0) {
    throw std::runtime_error("Ratio is out of bounds");
  }

  return interpolate(poses[start], poses[end], ratio);
}
}  // namespace loam