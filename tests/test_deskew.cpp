#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "loam/deskew.h"
#include "loam/geometry.h"

TEST(TestLoamDeskew, InterpolateFunc) {
  Eigen::Vector3d xi1(0.1, 0.2, 0.3);
  Eigen::Quaterniond q1(Eigen::AngleAxisd(xi1.norm(), xi1.normalized()));
  loam::Pose3d pose1(q1, Eigen::Vector3d(1.0, 2.0, 3.0));

  Eigen::Vector3d xi2(-0.4, 0.5, 0.6);
  Eigen::Quaterniond q2(Eigen::AngleAxisd(xi2.norm(), xi2.normalized()));
  loam::Pose3d pose2(q2, Eigen::Vector3d(-3.0, -2.0, 1.0));

  auto start = loam::interpolate(pose1, pose2, 0.0);
  ASSERT_TRUE(pose1.translation.isApprox(start.translation));
  ASSERT_TRUE(pose1.rotation.isApprox(start.rotation));

  auto end = loam::interpolate(pose1, pose2, 1.0);
  ASSERT_TRUE(pose2.translation.isApprox(end.translation));
  ASSERT_TRUE(pose2.rotation.isApprox(end.rotation));
};

TEST(TestLoamDeskew, ConstantVelocity) {
  std::vector<Eigen::Vector3d> pts;
  std::vector<double> stamps;
  pts.push_back(Eigen::Vector3d(1, 0, 0));
  stamps.push_back(0.0);

  Eigen::Vector3d vel_rot(0, 0, 0);
  Eigen::Vector3d vel_trans(1, 0, 0);

  // Verify the deskew does nothing to 0
  auto transformed = loam::deskewConstantVelocity<loam::ParenAccessor>(pts, stamps, vel_rot, vel_trans);
  ASSERT_TRUE(transformed[0].isApprox(pts[0]));

  // Verify the deskew moves the point
  stamps[0] = 1.0;
  transformed = loam::deskewConstantVelocity<loam::ParenAccessor>(pts, stamps, vel_rot, vel_trans);
  ASSERT_TRUE(transformed[0].isApprox(Eigen::Vector3d(2, 0, 0)));

  // Verify the deskew rotates the point
  vel_rot = Eigen::Vector3d(0, 0, M_PI / 2);
  vel_trans = Eigen::Vector3d(0, 0, 0);
  transformed = loam::deskewConstantVelocity<loam::ParenAccessor>(pts, stamps, vel_rot, vel_trans);
  ASSERT_TRUE(transformed[0].isApprox(Eigen::Vector3d(0, 1, 0)));
}

TEST(TestLoamDeskew, Interpolate) {
  std::vector<Eigen::Vector3d> pts;
  std::vector<double> stamps;
  pts.push_back(Eigen::Vector3d(1, 0, 0));
  stamps.push_back(0.0);

  loam::Pose3d start = loam::Pose3d::Identity();
  loam::Pose3d end = loam::Pose3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1.0, 0, 0));

  // Verify the deskew does nothing to 0
  EXPECT_THROW(loam::deskewInterpolate<loam::ParenAccessor>(pts, stamps, start, end), std::runtime_error);

  // Verify the deskew moves the point
  stamps[0] = 1.0;
  auto transformed = loam::deskewInterpolate<loam::ParenAccessor>(pts, stamps, start, end);
  ASSERT_TRUE(transformed[0].isApprox(Eigen::Vector3d(2, 0, 0)));

  // Verify the deskew rotates the point
  end.translation = Eigen::Vector3d::Zero();
  end.rotation = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ());
  transformed = loam::deskewInterpolate<loam::ParenAccessor>(pts, stamps, start, end);
  ASSERT_TRUE(transformed[0].isApprox(Eigen::Vector3d(0, 1, 0)))
      << transformed[0].transpose() << " != " << Eigen::Vector3d(0, 1, 0).transpose();
}