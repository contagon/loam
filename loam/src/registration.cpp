/** @brief Implementation of LOAM Features Module
 * @author Dan McGann
 * @date Mar 2024
 */

#include "loam/registration.h"

namespace loam {

/**
 * #### ##    ## ######## ######## ########  ##    ##    ###    ##
 *  ##  ###   ##    ##    ##       ##     ## ###   ##   ## ##   ##
 *  ##  ####  ##    ##    ##       ##     ## ####  ##  ##   ##  ##
 *  ##  ## ## ##    ##    ######   ########  ## ## ## ##     ## ##
 *  ##  ##  ####    ##    ##       ##   ##   ##  #### ######### ##
 *  ##  ##   ###    ##    ##       ##    ##  ##   ### ##     ## ##
 * #### ##    ##    ##    ######## ##     ## ##    ## ##     ## ########
 */

namespace registration_internal {

/*********************************************************************************************************************/
std::vector<std::pair<size_t, size_t>> associateEdges(const RegistrationParams& params,
                                                      const LoamFeatures<Eigen::Vector3d>& source_eig,
                                                      const LoamFeatures<Eigen::Vector3d>& target_eig,
                                                      const kdtree_internal::KDTree& target_edge_kdtree,
                                                      const Pose3d& target_T_source_est, Pose3d& estimate_update,
                                                      ceres::Problem& problem) {
  std::vector<std::pair<size_t, size_t>> edge_associations;
  edge_associations.reserve(source_eig.edge_points.size());
  // Compute Edge Associations
  for (size_t source_idx = 0; source_idx < source_eig.edge_points.size(); source_idx++) {
    // Transform the point into the target frame using the current solution
    const Eigen::Vector3d point_tgt = target_T_source_est.act(source_eig.edge_points.at(source_idx));

    // Associate the query point with target points
    std::vector<size_t> neighbor_edge_idxes = kdtree_internal::knnSearch(
        target_edge_kdtree, point_tgt, params.num_edge_neighbors, params.max_edge_neighbor_dist);
    if (neighbor_edge_idxes.size() < params.min_line_fit_points) continue;  // GUARD: Insufficient Matches

    // Accumulate the points into a matrix
    Eigen::MatrixXd neighbor_edge_points = Eigen::MatrixXd::Zero(neighbor_edge_idxes.size(), 3);
    for (size_t i = 0; i < neighbor_edge_idxes.size(); i++) {
      neighbor_edge_points.row(i) = target_eig.edge_points.at(neighbor_edge_idxes[i]);
    }

    // Fit the Line to these points
    auto [line, condition_number] = geometry_internal::fitLine(neighbor_edge_points);
    if (condition_number < params.min_line_condition_number) continue;  // GUARD: Edge points not co-linear

    // Construct the cost function and add it to the problem
    // Note the point has already been transformed by the current estimate
    // Therefore WRT the edge cost function class
    //     - The source frame = current estimate of the target frame
    //     - The target frame = the "true" target frame
    problem.AddResidualBlock(EdgeCostFunction::Create(point_tgt, line), new ceres::HuberLoss(1.0),
                             estimate_update.rotation.coeffs().data(), estimate_update.translation.data());
    // Accumulate the association
    edge_associations.emplace_back(source_idx, neighbor_edge_idxes.front());
  }
  return edge_associations;
}

/*********************************************************************************************************************/
std::vector<std::pair<size_t, size_t>> associatePlanes(const RegistrationParams& params,
                                                       const LoamFeatures<Eigen::Vector3d>& source_eig,
                                                       const LoamFeatures<Eigen::Vector3d>& target_eig,
                                                       const kdtree_internal::KDTree& source_plane_kdtree,
                                                       const kdtree_internal::KDTree& target_plane_kdtree,
                                                       const Pose3d& target_T_source_est, Pose3d& estimate_update,
                                                       ceres::Problem& problem) {
  std::vector<std::pair<size_t, size_t>> plane_associations;
  plane_associations.reserve(source_eig.planar_points.size());
  for (size_t source_idx = 0; source_idx < source_eig.planar_points.size(); source_idx++) {
    // Transform the point into the target frame using the current solution
    const Eigen::Vector3d query = source_eig.planar_points.at(source_idx);
    const Eigen::Vector3d point_tgt = target_T_source_est.act(query);

    // Associate the query point with target points
    std::vector<size_t> target_plane_idxes = kdtree_internal::knnSearch(
        target_plane_kdtree, point_tgt, params.num_plane_neighbors, params.max_plane_neighbor_dist);
    if (target_plane_idxes.size() < params.min_plane_fit_points) continue;  // GUARD: Insufficient Matches

    // Accumulate the points into a matrix
    Eigen::MatrixXd target_plane_points = Eigen::MatrixXd::Zero(target_plane_idxes.size(), 3);
    for (size_t i = 0; i < target_plane_idxes.size(); i++) {
      target_plane_points.row(i) = target_eig.planar_points.at(target_plane_idxes[i]);
    }

    // Fit the Plane to these points
    auto [target_plane, avg_dist] = geometry_internal::fitPlane(target_plane_points);
    if (avg_dist > params.max_avg_point_plane_dist) continue;  // GUARD: Plane points not co-planar

    // Construct the cost function and add it to the problem
    // Note the point has already been transformed by the current estimate
    // Therefore WRT the plane cost function class
    //     - The source frame = the "true" source fame
    //     - The est target frame = the current estimate of the target frame -> all source points should end in this
    //     - The target frame = the "true" target frame
    ceres::CostFunction* cost = nullptr;

    // Pseudo - Planar - Potentially also constrain the other two directions
    if (params.planar_version == RegistrationParams::PlanarVersion::PSEUDO_PLANAR) {
      cost = PseudoPlaneCostFunction::Create(point_tgt, target_plane_points.row(0), target_plane.normal,
                                             params.pseudo_plane_normal_epsilon);
    }

    // True Planar - Constrain only the normal direction
    else if (params.planar_version == RegistrationParams::PlanarVersion::TRUE_PLANAR) {
      cost = PlaneCostFunction::Create(point_tgt, target_plane);
    }

    // Plane - Plane - Two way constrain planes
    else if (params.planar_version == RegistrationParams::PlanarVersion::PLANE_PLANE) {
      // Associate the query point with target points (Done in the true source frame)
      std::vector<size_t> source_plane_idxes = kdtree_internal::knnSearch(
          source_plane_kdtree, query, params.num_plane_neighbors, params.max_plane_neighbor_dist);
      if (source_plane_idxes.size() < params.min_plane_fit_points) continue;  // GUARD: Insufficient Matches

      // Accumulate the points into a matrix and transform them into the est target frame
      Eigen::MatrixXd source_plane_points = Eigen::MatrixXd::Zero(source_plane_idxes.size(), 3);
      for (size_t i = 0; i < source_plane_idxes.size(); i++) {
        // TODO: Transform fitted plane instead of the points?
        Eigen::Vector3d source_pt = source_eig.planar_points.at(source_plane_idxes[i]);
        source_plane_points.row(i) = target_T_source_est.act(source_pt);
      }

      // Fit the Plane to these points (will be in est target frame)
      auto [source_plane, avg_dist] = geometry_internal::fitPlane(source_plane_points);
      if (avg_dist > params.max_avg_point_plane_dist) continue;  // GUARD: Plane points not co-planar

      cost = PlanePlaneCostFunction::Create(point_tgt, target_plane_points.row(0), source_plane.normal,
                                            target_plane.normal, params.pseudo_plane_normal_epsilon);
    } else {
      throw std::runtime_error("Unknown Planar Version");
    }

    problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), estimate_update.rotation.coeffs().data(),
                             estimate_update.translation.data());
    // Accumulate the association
    plane_associations.emplace_back(source_idx, target_plane_idxes.front());
  }
  return plane_associations;
}

/*********************************************************************************************************************/
std::vector<std::pair<size_t, size_t>> associatePoints(const RegistrationParams& params,
                                                       const LoamFeatures<Eigen::Vector3d>& source_eig,
                                                       const LoamFeatures<Eigen::Vector3d>& target_eig,
                                                       const kdtree_internal::KDTree& target_point_kdtree,
                                                       const Pose3d& target_T_source_est, Pose3d& estimate_update,
                                                       ceres::Problem& problem) {
  std::vector<std::pair<size_t, size_t>> point_associations;
  point_associations.reserve(source_eig.point_points.size());
  for (size_t source_idx = 0; source_idx < source_eig.point_points.size(); source_idx++) {
    // Transform the point into the target frame using the current solution
    const Eigen::Vector3d point_tgt = target_T_source_est.act(source_eig.point_points.at(source_idx));

    // Associate the query point with target points
    std::vector<size_t> neighbor_point_idxes =
        kdtree_internal::knnSearch(target_point_kdtree, point_tgt, 1, params.max_point_neighbor_dist);
    if (neighbor_point_idxes.size() < 1) continue;  // GUARD: Insufficient Matches

    Eigen::Vector3d point = target_eig.point_points.at(neighbor_point_idxes.front());

    // Construct the cost function and add it to the problem
    // Note the point has already been transformed by the current estimate
    // Therefore WRT the point cost function class
    //     - The source frame = current estimate of the target frame
    //     - The target frame = the "true" target frame
    problem.AddResidualBlock(PointCostFunction::Create(point_tgt, point), new ceres::HuberLoss(1.0),
                             estimate_update.rotation.coeffs().data(), estimate_update.translation.data());
    // Accumulate the association
    point_associations.emplace_back(source_idx, neighbor_point_idxes.front());
  }
  return point_associations;
}

}  // namespace registration_internal
}  // namespace loam