#pragma once
#include <Eigen/Dense>

#include "loam/common.h"
#include "loam/geometry.h"

namespace loam {
template <template <typename> class Accessor = FieldAccessor, typename PointType, template <typename> class Alloc>
std::vector<Eigen::Vector3d> deskewConstantVelocity(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                                    const std::vector<double>& rel_time_stamps, Eigen::Vector3d vel_rot,
                                                    Eigen::Vector3d vel_trans);

template <template <typename> class Accessor = FieldAccessor, typename PointType, template <typename> class Alloc>
std::vector<Eigen::Vector3d> deskewInterpolate(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                               const std::vector<double>& rel_time_stamps, Pose3d pose_start,
                                               Pose3d pose_end);
}  // namespace loam

#include "loam/deskew-inl.h"