/** @brief Implementation of LOAM Features Module
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include "loam/registration.h"

namespace loam {

/*********************************************************************************************************************/
template <template <typename> class Accessor, typename PointType, template <typename> class Alloc>
Pose3d registerFeatures(const LoamFeatures<PointType, Alloc>& source, const LoamFeatures<PointType, Alloc>& target,
                        const Pose3d& target_T_source_init, const RegistrationParams& params,
                        std::shared_ptr<RegistrationDetail> detail) {
  // Convert features to eigen once here to avoid repeated conversions later
  LoamFeatures<Eigen::Vector3d> source_eig = features_internal::featuresToEigen<Accessor>(source);
  LoamFeatures<Eigen::Vector3d> target_eig = features_internal::featuresToEigen<Accessor>(target);

  // Compute a KDtree for the target features: 20 leaf nodes is approx optimal given nanoflann's documentation
  kdtree_internal::KDTreeDataAdaptor target_edge_adaptor(target_eig.edge_points);
  kdtree_internal::KDTree target_edge_kdtree(3, target_edge_adaptor, kdtree_internal::KDTreeParams(20));
  kdtree_internal::KDTreeDataAdaptor target_plane_adaptor(target_eig.planar_points);
  kdtree_internal::KDTree target_plane_kdtree(3, target_plane_adaptor, kdtree_internal::KDTreeParams(20));
  kdtree_internal::KDTreeDataAdaptor target_point_adaptor(target_eig.point_points);
  kdtree_internal::KDTree target_point_kdtree(3, target_point_adaptor, kdtree_internal::KDTreeParams(20));

  // TODO: Is this computing anything? Should we offload it?
  kdtree_internal::KDTreeDataAdaptor source_plane_adaptor(source_eig.planar_points);
  kdtree_internal::KDTree source_plane_kdtree(3, source_plane_adaptor, kdtree_internal::KDTreeParams(20));

  // Setup the parameters of the optimization (the relative pose)
  Pose3d target_T_source_est(target_T_source_init);
  RegistrationDetail::TerminationType termination_type = RegistrationDetail::TerminationType::MAX_ITER;
  for (size_t iter_count = 0; iter_count < params.max_iterations; iter_count++) {
    // Define the Ceres Problem
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // Define the parameters of the system and their respective manifolds
    // Note we are estimating a relative update between from the current target estimate to the true target frame
    Pose3d estimate_update;  // next_target_T_prev_target
    problem.AddParameterBlock(estimate_update.rotation.coeffs().data(), 4, new ceres::QuaternionManifold());
    problem.AddParameterBlock(estimate_update.translation.data(), 3, new ceres::EuclideanManifold<3>());

    // Compute Associations and accumulate the optimization problem [WARN: Mutates "problem"]
    auto edge_assoc = registration_internal::associateEdges(params, source_eig, target_eig, target_edge_kdtree,
                                                            target_T_source_est, estimate_update, problem);

    // TODO: Make separate plane-plane option here
    // TODO: Compute all source normals here to avoid recomputing them
    auto plane_assoc =
        registration_internal::associatePlanes(params, source_eig, target_eig, source_plane_kdtree, target_plane_kdtree,
                                               target_T_source_est, estimate_update, problem);
    auto point_assoc = registration_internal::associatePoints(params, source_eig, target_eig, target_point_kdtree,
                                                              target_T_source_est, estimate_update, problem);

    if (edge_assoc.size() + plane_assoc.size() + point_assoc.size() < params.min_associations) {
      termination_type = RegistrationDetail::TerminationType::INSUFFICIENT_ASSOCIATIONS;
      break;
    }

    // Solve the problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 4;
    options.num_threads = 1;
    // options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // If configured Fill out the detail
    if (detail) {
      detail->iteration_info.emplace_back(target_T_source_est, edge_assoc, plane_assoc, estimate_update);
    }

    // Update the solution
    // The update is computed as next_target_T_prev_target so compose it on the left side
    target_T_source_est = estimate_update.compose(target_T_source_est);

    // Check for convergence - defined as when the computed update is sufficiently small
    const double angle_change = estimate_update.rotation.angularDistance(Eigen::Quaterniond::Identity());
    const double position_change = estimate_update.translation.norm();
    if (angle_change < params.rotation_convergence_thresh && position_change < params.position_convergence_thresh) {
      termination_type = RegistrationDetail::TerminationType::CONVERGED;
      break;
    }
  }
  // Fill out termination reason in detail if it exists
  if (detail) detail->termination_type = termination_type;
  return target_T_source_est;
}

/**
 * #### ##    ## ######## ######## ########  ##    ##    ###    ##
 *  ##  ###   ##    ##    ##       ##     ## ###   ##   ## ##   ##
 *  ##  ####  ##    ##    ##       ##     ## ####  ##  ##   ##  ##
 *  ##  ## ## ##    ##    ######   ########  ## ## ## ##     ## ##
 *  ##  ##  ####    ##    ##       ##   ##   ##  #### ######### ##
 *  ##  ##   ###    ##    ##       ##    ##  ##   ### ##     ## ##
 * #### ##    ##    ##    ######## ##     ## ##    ## ##     ## ########
 */
/*********************************************************************************************************************/
namespace registration_internal {

template <typename T>
bool EdgeCostFunction::operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const {
  Eigen::Map<const Eigen::Quaternion<T>> t_R_s(t_R_s_ptr);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_p_s(t_p_s_ptr);

  // Transform the point into the target frame given the current estimate
  const Eigen::Matrix<T, 3, 1> target_pt_ = t_R_s * source_pt_.cast<T>() + t_p_s;

  // Compute the loss
  residuals_ptr[0] = geometry_internal::pointToLineDistance<T>(target_pt_, line_.a.cast<T>(), line_.b.cast<T>());
  return true;
}

/*********************************************************************************************************************/
template <typename T>
bool PlaneCostFunction::operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const {
  Eigen::Map<const Eigen::Quaternion<T>> t_R_s(t_R_s_ptr);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_p_s(t_p_s_ptr);

  // Transform the point into the target frame given the current estimate
  const Eigen::Matrix<T, 3, 1> target_pt_ = t_R_s * source_pt_.cast<T>() + t_p_s;

  // Compute the loss
  residuals_ptr[0] = geometry_internal::pointToPlaneDistance<T>(target_pt_, plane_.normal.cast<T>(), T(plane_.d));
  return true;
}

/*********************************************************************************************************************/
template <typename T>
bool PseudoPlaneCostFunction::operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const {
  Eigen::Map<const Eigen::Quaternion<T>> t_R_s(t_R_s_ptr);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_p_s(t_p_s_ptr);

  // Transform the point into the target frame given the current estimate
  const Eigen::Matrix<T, 3, 1> target_pt_ = t_R_s * source_pt_.cast<T>() + t_p_s;

  // Compute the pseudo plane projection
  const Eigen::Matrix<T, 3, 1> v = origin_.cast<T>() - target_pt_;
  const Eigen::Matrix<T, 3, 1> v_proj = sqrt_projector_.cast<T>() * v;

  // Copy the loss
  residuals_ptr[0] = v_proj[0];
  residuals_ptr[1] = v_proj[1];
  residuals_ptr[2] = v_proj[2];

  return true;
}

/*********************************************************************************************************************/
template <typename T>
bool PlanePlaneCostFunction::operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const {
  Eigen::Map<const Eigen::Quaternion<T>> t_R_s(t_R_s_ptr);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_p_s(t_p_s_ptr);

  // Transform the point into the target frame given the current estimate
  const Eigen::Matrix<T, 3, 1> est_target_pt_ = t_R_s * source_pt_.cast<T>() + t_p_s;
  const Eigen::Matrix<T, 3, 3> est_target_sqrt_projector =
      t_R_s * source_sqrt_projector_.cast<T>();  // TODO: * t_R_s.transpose();

  const Eigen::Matrix<T, 3, 3> sqrt_projector = target_sqrt_projector_.cast<T>() + est_target_sqrt_projector;

  // Compute the pseudo plane projection
  const Eigen::Matrix<T, 3, 1> v = target_pt_.cast<T>() - est_target_pt_;
  const Eigen::Matrix<T, 3, 1> v_proj = sqrt_projector * v;

  // Copy the loss
  residuals_ptr[0] = v_proj[0];
  residuals_ptr[1] = v_proj[1];
  residuals_ptr[2] = v_proj[2];

  return true;
}

/*********************************************************************************************************************/
template <typename T>
bool PointCostFunction::operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const {
  Eigen::Map<const Eigen::Quaternion<T>> t_R_s(t_R_s_ptr);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_p_s(t_p_s_ptr);
  const Eigen::Matrix<T, 3, 1> target_pt = target_pt_.cast<T>();

  // Transform the point into the target frame given the current estimate
  const Eigen::Matrix<T, 3, 1> target_pt_est_ = t_R_s * source_pt_.cast<T>() + t_p_s;

  // Compute the loss
  residuals_ptr[0] = target_pt[0] - target_pt_est_[0];
  residuals_ptr[1] = target_pt[1] - target_pt_est_[1];
  residuals_ptr[2] = target_pt[2] - target_pt_est_[2];
  return true;
}

}  // namespace registration_internal
}  // namespace loam