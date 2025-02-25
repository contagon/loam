/** @brief Implementation of LOAM Features Module
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include "loam/features.h"

namespace loam {

/*********************************************************************************************************************/
template <template <typename> class Accessor, typename PointType, template <typename> class Alloc>
LoamFeatures<PointType, Alloc> extractFeatures(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                               const LidarParams& lidar_params, const FeatureExtractionParams& params) {
  validateLidarScan(input_scan, lidar_params);
  // Initialize Results
  LoamFeatures<PointType, Alloc> out_features;
  // Compute the number of points in each sector given the parameters
  const size_t points_per_sector = lidar_params.points_per_line / params.number_sectors;

  // Step 1: Compute mask of valid points
  std::vector<bool> valid_mask = computeValidPoints<Accessor>(input_scan, lidar_params, params);

  // Step 2: Compute Curvature for all points this is used in the extraction of both planar and edge feature points
  std::vector<PointCurvature> curvature = computeCurvature<Accessor>(input_scan, lidar_params, params, valid_mask);

  /// Step 3: Detect features in each sector of each scan line
  for (size_t scan_line_idx = 0; scan_line_idx < lidar_params.scan_lines; scan_line_idx++) {
    // Independently detect features in each sector of this scan_line
    for (size_t sector_idx = 0; sector_idx < params.number_sectors; sector_idx++) {
      // Get the point index of the sector start and sector end
      const size_t sector_start_pt = (scan_line_idx * lidar_params.points_per_line) + (sector_idx * points_per_sector);
      // Special case for end point as we add any reminder points to the last sector
      const size_t sector_end_pt = (sector_idx == params.number_sectors - 1)
                                       ? ((scan_line_idx + 1) * lidar_params.points_per_line)
                                       : sector_start_pt + points_per_sector;

      // Sort the points within the sector based on curvature
      std::sort(curvature.begin() + sector_start_pt, curvature.begin() + sector_end_pt, curvatureComparator);

      // Search largest to smallest curvature [i.e. edge features] WARN: Mutates out_features + valid_mask
      features_internal::extractSectorEdgeFeatures(sector_start_pt, sector_end_pt, input_scan, curvature, params,
                                                   out_features, valid_mask);
      // Search smallest to largest [i.e. planar features] WARN: Mutates out_features + valid_mask
      features_internal::extractSectorPlanarFeatures(sector_start_pt, sector_end_pt, input_scan, curvature, params,
                                                     out_features, valid_mask);
      // Search smallest to largest [i.e. point features] WARN: Mutates out_features + valid_mask
      features_internal::extractSectorPointFeatures(sector_start_pt, sector_end_pt, input_scan, curvature, params,
                                                    out_features, valid_mask);
      // Fill in the scan indices for the edge points
      while (out_features.edge_points.size() > out_features.edge_scan_indices.size()) {
        out_features.edge_scan_indices.push_back(scan_line_idx);
      }
      while (out_features.planar_points.size() > out_features.planar_scan_indices.size()) {
        out_features.planar_scan_indices.push_back(scan_line_idx);
      }
      while (out_features.point_points.size() > out_features.point_scan_indices.size()) {
        out_features.point_scan_indices.push_back(scan_line_idx);
      }

    }  // end sector search
  }  // end scan line search
  return out_features;
}

/*********************************************************************************************************************/
template <template <typename> class Accessor, typename PointType, template <typename> class Alloc>
std::vector<PointCurvature> computeCurvature(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                             const LidarParams& lidar_params, const FeatureExtractionParams& params,
                                             const std::vector<bool>& valid_mask) {
  validateLidarScan(input_scan, lidar_params);
  // Allocate vector (with zeros) to store curvature
  std::vector<PointCurvature> curvature;
  curvature.reserve(input_scan.size());

  // Convert all valid points to eigen for use in kdtree
  std::vector<Eigen::Vector3d> eigen_points;
  for (size_t idx = 0; idx < input_scan.size(); idx++) {
    if (valid_mask.size() == 0 || valid_mask[idx]) {
      eigen_points.push_back(pointToEigen<Accessor>(input_scan[idx]));
    }
  }

  // Make kdtree for lookup
  kdtree_internal::KDTreeDataAdaptor adaptor(eigen_points);
  kdtree_internal::KDTree kdtree(3, adaptor, kdtree_internal::KDTreeParams(20));

  // Structured search (search over each scan line individually over all points [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < lidar_params.scan_lines; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < lidar_params.points_per_line; line_pt_idx++) {
      const size_t idx = (scan_line_idx * lidar_params.points_per_line) + line_pt_idx;
      // If point is on the edge of the scan line record invalid curvature [-1]
      if (line_pt_idx < params.neighbor_points ||
          line_pt_idx >= lidar_params.points_per_line - params.neighbor_points) {
        curvature.push_back(PointCurvature(idx, -1));
      }
      // If a point is all zeros, it's invalid, also report invalid curvature
      // Quick test shows that this doesn't impact anything later on -> these points will be filtered out anyways
      else if (valid_mask.size() > 0 && !valid_mask[idx]) {
        curvature.push_back(PointCurvature(idx, -1));
      }
      // If not an edge point compute the curvature
      else {
        // LOAM based curvature
        if (params.curvature_type == FeatureExtractionParams::Curvature::LOAM) {
          // Initialize with the difference term
          double dx = -(2.0 * params.neighbor_points) * Accessor<PointType>::x(input_scan[idx]);
          double dy = -(2.0 * params.neighbor_points) * Accessor<PointType>::y(input_scan[idx]);
          double dz = -(2.0 * params.neighbor_points) * Accessor<PointType>::z(input_scan[idx]);
          // Iterate over neighbors and accumulate
          for (size_t n = 1; n <= params.neighbor_points; n++) {
            dx = dx + Accessor<PointType>::x(input_scan[idx - n]) + Accessor<PointType>::x(input_scan[idx + n]);
            dy = dy + Accessor<PointType>::y(input_scan[idx - n]) + Accessor<PointType>::y(input_scan[idx + n]);
            dz = dz + Accessor<PointType>::z(input_scan[idx - n]) + Accessor<PointType>::z(input_scan[idx + n]);
          }
          curvature.push_back(PointCurvature(idx, dx * dx + dy * dy + dz * dz));
        }

        // Eigenvalue based curvature
        else if (params.curvature_type == FeatureExtractionParams::Curvature::EIGEN) {
          // Prep to compute covariance
          Eigen::Vector3d center = pointToEigen<Accessor>(input_scan[idx]);
          Eigen::Matrix<double, Eigen::Dynamic, 3> neighbors(2 * params.neighbor_points, 3);

          for (size_t n = 1; n <= params.neighbor_points; n++) {
            neighbors.row(2 * n - 2) = pointToEigen<Accessor>(input_scan[idx - n]) - center;
            neighbors.row(2 * n - 1) = pointToEigen<Accessor>(input_scan[idx + n]) - center;
          }

          // Compute covariance matrix and extract smallest eigenvalue
          Eigen::Matrix3d cov = (neighbors.transpose() * neighbors);
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> b(cov);
          auto abs = b.eigenvalues().cwiseAbs();
          double c = abs[1];
          curvature.push_back(PointCurvature(idx, c));
        }

        // Eigenvalue based curvature with a nearest neighbor search
        else if (params.curvature_type == FeatureExtractionParams::Curvature::EIGEN_NN) {
          // Find nearest neighbors within a distance
          const Eigen::Vector3d center(pointToEigen<Accessor>(input_scan[idx]));
          std::vector<size_t> neighbors =
              kdtree_internal::knnSearch(kdtree, center, 2 * params.neighbor_points + 1, params.max_neighbor_distance);

          // Skip first found point (it's the center)
          Eigen::Matrix<double, Eigen::Dynamic, 3> neighbor_points(neighbors.size() - 1, 3);
          for (size_t n = 1; n < neighbors.size(); n++) {
            neighbor_points.row(n - 1) = eigen_points.at(neighbors[n]) - center;
          }

          // Compute covariance matrix and extract smallest eigenvalue
          Eigen::Matrix3d cov = (neighbor_points.transpose() * neighbor_points);
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> b(cov);
          double c = b.eigenvalues()[0];
          curvature.push_back(PointCurvature(idx, c));
        }
      }
    }
  }
  return curvature;
}

/*********************************************************************************************************************/
template <template <typename> class Accessor, typename PointType, template <typename> class Alloc>
std::vector<bool> computeValidPoints(const std::vector<PointType, Alloc<PointType>>& input_scan,
                                     const LidarParams& lidar_params, const FeatureExtractionParams& params) {
  validateLidarScan(input_scan, lidar_params);
  // Allocate mask (with valid flags)
  std::vector<bool> mask(input_scan.size(), true);

  // Structured search (search over each scan line individually over all points [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < lidar_params.scan_lines; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < lidar_params.points_per_line; line_pt_idx++) {
      const size_t idx = (scan_line_idx * lidar_params.points_per_line) + line_pt_idx;

      // CHECK 1: Due to edge effects, the first and last neighbor_points points of each scan line are invalid
      if (features_internal::markEdgesInvalid(idx, line_pt_idx, lidar_params, params, mask)) continue;

      // Get the current point and its two neighbors
      const PointType prev_point = input_scan.at(idx - 1);
      const PointType point = input_scan.at(idx);
      const PointType next_point = input_scan[idx + 1];

      // Compute the range of each point
      const double point_range = pointRange<Accessor>(point);
      const double next_point_range = pointRange<Accessor>(next_point);
      const double prev_point_range = pointRange<Accessor>(prev_point);

      // CHECK 2: Is the point in the valid range of the LiDAR
      if (features_internal::markOutOfRangeInvalid(idx, point_range, lidar_params, params, mask)) continue;
      // CHECK 3: Occlusions
      if (features_internal::markOccludedInvalid(idx, point_range, next_point_range, params, mask)) continue;
      // CHECK 4: Check if the point is on a plane nearly parallel to the LiDAR Beam (no continue b/c last )
      features_internal::markParallelInvalid(idx, prev_point_range, point_range, next_point_range, params, mask);
    }  // end line point search
  }  // end scan line search
  return mask;
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
namespace features_internal {
template <typename PointType, template <typename> class Alloc>
void extractSectorEdgeFeatures(const size_t& sector_start_point, const size_t& sector_end_point,
                               const std::vector<PointType, Alloc<PointType>>& input_scan,
                               const std::vector<PointCurvature>& curvature, const FeatureExtractionParams& params,
                               LoamFeatures<PointType, Alloc>& out_features, std::vector<bool>& valid_mask) {
  size_t num_sector_edge_features = 0;
  for (size_t sorted_curv_idx_p1 = sector_end_point; sorted_curv_idx_p1 > sector_start_point; sorted_curv_idx_p1--) {
    const PointCurvature curv = curvature[sorted_curv_idx_p1 - 1];  // subtraction as loop cannot go negative

    if (valid_mask[curv.index] && curv.curvature > params.edge_feat_threshold) {
      out_features.edge_points.push_back(input_scan.at(curv.index));  // Add to edge points
      for (size_t n = 0; n < params.neighbor_points; n++) {           // update mask
        valid_mask[curv.index + n] = false;
        valid_mask[curv.index - n] = false;
      }
      num_sector_edge_features++;
    }
    // Early exit if we have found enough features
    if (num_sector_edge_features > params.max_edge_feats_per_sector) break;
  }
}

/*********************************************************************************************************************/
template <typename PointType, template <typename> class Alloc>
void extractSectorPlanarFeatures(const size_t& sector_start_point, const size_t& sector_end_point,
                                 const std::vector<PointType, Alloc<PointType>>& input_scan,
                                 const std::vector<PointCurvature>& curvature, const FeatureExtractionParams& params,
                                 LoamFeatures<PointType, Alloc>& out_features, std::vector<bool>& valid_mask) {
  size_t num_sector_planar_features = 0;
  for (size_t sorted_curv_idx = sector_start_point; sorted_curv_idx < sector_end_point; sorted_curv_idx++) {
    const PointCurvature curv = curvature[sorted_curv_idx];
    if (valid_mask[curv.index] && curv.curvature < params.planar_feat_threshold) {
      out_features.planar_points.push_back(input_scan.at(curv.index));  // Add to edge points
      for (size_t n = 0; n < params.neighbor_points; n++) {             // update mask
        valid_mask[curv.index + n] = false;
        valid_mask[curv.index - n] = false;
      }
      num_sector_planar_features++;
    }
    // Early exit if we have found enough features
    if (num_sector_planar_features > params.max_planar_feats_per_sector) break;

  }  // end feature search in sector
}

/*********************************************************************************************************************/
template <typename PointType, template <typename> class Alloc>
void extractSectorPointFeatures(const size_t& sector_start_point, const size_t& sector_end_point,
                                const std::vector<PointType, Alloc<PointType>>& input_scan,
                                const std::vector<PointCurvature>& curvature, const FeatureExtractionParams& params,
                                LoamFeatures<PointType, Alloc>& out_features, std::vector<bool>& valid_mask) {
  size_t num_sector_point_features = 0;

  // Figure out how many we may have
  std::vector<size_t> unused_points;
  for (size_t idx = sector_start_point; idx < sector_end_point; idx++) {
    if (valid_mask[idx]) {
      unused_points.push_back(idx);
    }
  }

  // By what factor do we have too many?
  size_t factor = 1 + unused_points.size() / params.max_point_feats_per_sector;
  // Do "factor" number of passes over the points until we get enough
  // This should help spread them out evenly
  for (size_t offset = 0; offset < factor; offset++) {
    for (size_t unused_idx = offset; unused_idx < unused_points.size(); unused_idx += factor) {
      const size_t idx = unused_points[unused_idx];
      if (valid_mask[idx]) {
        out_features.point_points.push_back(input_scan.at(idx));  // Add to points
        for (size_t n = 0; n < params.neighbor_points; n++) {     // update mask
          // TODO: make sure we don't get in the way of adjacent sector edge/feature points?
          valid_mask[idx + n] = false;
          valid_mask[idx - n] = false;
        }
        num_sector_point_features++;
      }
      // Early exit if we have found enough features
      if (num_sector_point_features > params.max_point_feats_per_sector) break;
    }
  }
}

}  // namespace features_internal

}  // namespace loam
