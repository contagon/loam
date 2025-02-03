#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "loam/deskew.h"
#include "loam/geometry.h"

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