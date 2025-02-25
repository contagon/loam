#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loam/features.h"
#include "loam/loam.h"

// Use short form as it appears a lot
namespace py = pybind11;

PYBIND11_MODULE(loam_python, m) {
  py::module gtsam = py::module::import("numpy");

  /**
   *  ######   #######  ##     ## ##     ##  #######  ##    ##
   * ##    ## ##     ## ###   ### ###   ### ##     ## ###   ##
   * ##       ##     ## #### #### #### #### ##     ## ####  ##
   * ##       ##     ## ## ### ## ## ### ## ##     ## ## ## ##
   * ##       ##     ## ##     ## ##     ## ##     ## ##  ####
   * ##    ## ##     ## ##     ## ##     ## ##     ## ##   ###
   *  ######   #######  ##     ## ##     ##  #######  ##    ##
   */

  py::class_<loam::LidarParams>(m, "LidarParams")
      .def(py::init<size_t, size_t, double, double>(),  // Constructor
           py::kw_only(), py::arg("scan_lines"), py::arg("points_per_line"), py::arg("min_range"), py::arg("max_range"))
      .def_readonly("scan_lines", &loam::LidarParams::scan_lines)
      .def_readonly("points_per_line", &loam::LidarParams::points_per_line)
      .def_readonly("min_range", &loam::LidarParams::min_range)
      .def_readonly("max_range", &loam::LidarParams::max_range);

  /**
   *  ######   ########  #######  ##     ## ######## ######## ########  ##    ##
   * ##    ##  ##       ##     ## ###   ### ##          ##    ##     ##  ##  ##
   * ##        ##       ##     ## #### #### ##          ##    ##     ##   ####
   * ##   #### ######   ##     ## ## ### ## ######      ##    ########     ##
   * ##    ##  ##       ##     ## ##     ## ##          ##    ##   ##      ##
   * ##    ##  ##       ##     ## ##     ## ##          ##    ##    ##     ##
   *  ######   ########  #######  ##     ## ########    ##    ##     ##    ##
   */
  py::class_<Eigen::Quaterniond>(m, "Quaterniond")
      .def(py::init<double, double, double, double>(),  // Constructor
           py::kw_only(), py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
      .def("w", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::w)
      .def("x", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::x)
      .def("y", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::y)
      .def("z", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::z);

  py::class_<loam::Pose3d>(m, "Pose3d")
      .def(py::init<Eigen::Quaterniond, Eigen::Vector3d>(),  // Constructor
           py::arg("rotation"), py::arg("translation"))
      .def_static("Identity", &loam::Pose3d::Identity)
      .def("inverse", &loam::Pose3d::inverse)
      .def("compose", &loam::Pose3d::compose, py::arg("other"))
      .def("act", &loam::Pose3d::act, py::arg("point"))
      .def_readwrite("rotation", &loam::Pose3d::rotation)
      .def_readwrite("translation", &loam::Pose3d::translation);

  // ------------------------- Deskew ------------------------- //
  m.def("deskewConstantVelocity", &loam::deskewConstantVelocity<loam::AtAccessor, py::array_t<double>, std::allocator>,
        py::arg("input_scan"), py::arg("rel_time_stamps"), py::arg("vel_rot"), py::arg("vel_trans"));

  m.def("deskewImu", &loam::deskewImu<loam::AtAccessor, py::array_t<double>, std::allocator>,  //
        py::arg("input_scan"), py::arg("point_time_stamps"), py::arg("poses"), py::arg("imu_time_stamps"));

  /**
   * ######## ########    ###    ######## ##     ## ########  ########  ######
   * ##       ##         ## ##      ##    ##     ## ##     ## ##       ##    ##
   * ##       ##        ##   ##     ##    ##     ## ##     ## ##       ##
   * ######   ######   ##     ##    ##    ##     ## ########  ######    ######
   * ##       ##       #########    ##    ##     ## ##   ##   ##             ##
   * ##       ##       ##     ##    ##    ##     ## ##    ##  ##       ##    ##
   * ##       ######## ##     ##    ##     #######  ##     ## ########  ######
   */

  py::enum_<loam::FeatureExtractionParams::Curvature>(m, "Curvature")
      .value("LOAM", loam::FeatureExtractionParams::Curvature::LOAM)
      .value("EIGEN", loam::FeatureExtractionParams::Curvature::EIGEN)
      .value("EIGEN_NN", loam::FeatureExtractionParams::Curvature::EIGEN_NN)
      .export_values();

  py::class_<loam::FeatureExtractionParams>(m, "FeatureExtractionParams")
      .def(py::init<>())  // Constructor
      .def_readwrite("neighbor_points", &loam::FeatureExtractionParams::neighbor_points)
      .def_readwrite("number_sectors", &loam::FeatureExtractionParams::number_sectors)
      .def_readwrite("max_edge_feats_per_sector", &loam::FeatureExtractionParams::max_edge_feats_per_sector)
      .def_readwrite("max_planar_feats_per_sector", &loam::FeatureExtractionParams::max_planar_feats_per_sector)
      .def_readwrite("max_point_feats_per_sector", &loam::FeatureExtractionParams::max_point_feats_per_sector)
      .def_readwrite("edge_feat_threshold", &loam::FeatureExtractionParams::edge_feat_threshold)
      .def_readwrite("planar_feat_threshold", &loam::FeatureExtractionParams::planar_feat_threshold)
      .def_readwrite("occlusion_thresh", &loam::FeatureExtractionParams::occlusion_thresh)
      .def_readwrite("parallel_thresh", &loam::FeatureExtractionParams::parallel_thresh)
      .def_readwrite("max_neighbor_distance", &loam::FeatureExtractionParams::max_neighbor_distance)
      .def_readwrite("curvature_type", &loam::FeatureExtractionParams::curvature_type);

  py::class_<loam::LoamFeatures<py::array_t<double>>>(m, "LoamFeatures")
      .def(py::init<>())  // Constructor
      .def_readwrite("edge_points", &loam::LoamFeatures<py::array_t<double>>::edge_points)
      .def_readwrite("edge_scan_indices", &loam::LoamFeatures<py::array_t<double>>::edge_scan_indices)
      .def_readwrite("planar_points", &loam::LoamFeatures<py::array_t<double>>::planar_points)
      .def_readwrite("planar_scan_indices", &loam::LoamFeatures<py::array_t<double>>::planar_scan_indices)
      .def_readwrite("point_points", &loam::LoamFeatures<py::array_t<double>>::point_points)
      .def_readwrite("point_scan_indices", &loam::LoamFeatures<py::array_t<double>>::point_scan_indices);

  m.def("extractFeatures", &loam::extractFeatures<loam::AtAccessor, py::array_t<double>, std::allocator>,
        py::arg("input_scan"), py::arg("lidar_params"), py::arg("params") = loam::FeatureExtractionParams());

  m.def("computeCurvature", &loam::computeCurvature<loam::AtAccessor, py::array_t<double>, std::allocator>,  //
        py::arg("input_scan"), py::arg("lidar_params"), py::arg("params") = loam::FeatureExtractionParams(),
        py::arg("valid_mask") = std::vector<bool>());

  m.def("computeValidPoints", &loam::computeValidPoints<loam::AtAccessor, py::array_t<double>, std::allocator>,  //
        py::arg("input_scan"), py::arg("lidar_params"), py::arg("params") = loam::FeatureExtractionParams());

  /**
   * ########  ########  ######   ####  ######  ######## ########     ###    ######## ####  #######  ##    ##
   * ##     ## ##       ##    ##   ##  ##    ##    ##    ##     ##   ## ##      ##     ##  ##     ## ###   ##
   * ##     ## ##       ##         ##  ##          ##    ##     ##  ##   ##     ##     ##  ##     ## ####  ##
   * ########  ######   ##   ####  ##   ######     ##    ########  ##     ##    ##     ##  ##     ## ## ## ##
   * ##   ##   ##       ##    ##   ##        ##    ##    ##   ##   #########    ##     ##  ##     ## ##  ####
   * ##    ##  ##       ##    ##   ##  ##    ##    ##    ##    ##  ##     ##    ##     ##  ##     ## ##   ###
   * ##     ## ########  ######   ####  ######     ##    ##     ## ##     ##    ##    ####  #######  ##    ##
   */

  py::class_<loam::RegistrationParams>(m, "RegistrationParams")
      .def(py::init<>())  // Constructor
      .def_readwrite("num_edge_neighbors", &loam::RegistrationParams::num_edge_neighbors)
      .def_readwrite("max_edge_neighbor_dist", &loam::RegistrationParams::max_edge_neighbor_dist)
      .def_readwrite("min_line_fit_points", &loam::RegistrationParams::min_line_fit_points)
      .def_readwrite("min_line_condition_number", &loam::RegistrationParams::min_line_condition_number)
      .def_readwrite("num_plane_neighbors", &loam::RegistrationParams::num_plane_neighbors)
      .def_readwrite("max_plane_neighbor_dist", &loam::RegistrationParams::max_plane_neighbor_dist)
      .def_readwrite("min_plane_fit_points", &loam::RegistrationParams::min_plane_fit_points)
      .def_readwrite("pseudo_plane_normal_epsilon", &loam::RegistrationParams::pseudo_plane_normal_epsilon)
      .def_readwrite("use_plane_to_plane", &loam::RegistrationParams::use_plane_to_plane)
      .def_readwrite("max_avg_point_plane_dist", &loam::RegistrationParams::max_avg_point_plane_dist)
      .def_readwrite("max_iterations", &loam::RegistrationParams::max_iterations)
      .def_readwrite("rotation_convergence_thresh", &loam::RegistrationParams::rotation_convergence_thresh)
      .def_readwrite("position_convergence_thresh", &loam::RegistrationParams::position_convergence_thresh)
      .def_readwrite("min_associations", &loam::RegistrationParams::min_associations);

  py::class_<loam::RegistrationDetail::IterationInfo>(m, "RegistrationIterationInfo")
      .def(py::init<const loam::Pose3d, const std::vector<std::pair<size_t, size_t>>,  // Constructor
                    const std::vector<std::pair<size_t, size_t>>, const loam::Pose3d>(),
           py::arg("target_T_source_init"), py::arg("edge_associations"), py::arg("plane_associations"),
           py::arg("estimate_update"))
      .def_readwrite("target_T_source_init", &loam::RegistrationDetail::IterationInfo::target_T_source_init)
      .def_readwrite("edge_associations", &loam::RegistrationDetail::IterationInfo::edge_associations)
      .def_readwrite("plane_associations", &loam::RegistrationDetail::IterationInfo::plane_associations)
      .def_readwrite("estimate_update", &loam::RegistrationDetail::IterationInfo::estimate_update);

  py::enum_<loam::RegistrationDetail::TerminationType>(m, "RegistrationTerminationType")
      .value("CONVERGED", loam::RegistrationDetail::TerminationType::CONVERGED)
      .value("MAX_ITER", loam::RegistrationDetail::TerminationType::MAX_ITER)
      .value("INSUFFICIENT_ASSOCIATIONS", loam::RegistrationDetail::TerminationType::INSUFFICIENT_ASSOCIATIONS)
      .export_values();

  // Note: Change the default holder to shared pointer so we can pass it by shared pointer in registerFeatures
  py::class_<loam::RegistrationDetail, std::shared_ptr<loam::RegistrationDetail>>(m, "RegistrationDetail")
      .def(py::init<>())  // Constructor
      .def_readwrite("iteration_info", &loam::RegistrationDetail::iteration_info)
      .def_readwrite("termination_type", &loam::RegistrationDetail::termination_type);

  m.def("registerFeatures", &loam::registerFeatures<loam::AtAccessor, py::array_t<double>, std::allocator>,  //
        py::arg("source"), py::arg("target"), py::arg("target_T_source_init"),
        py::arg("params") = loam::RegistrationParams(),
        py::arg("detail") = std::shared_ptr<loam::RegistrationDetail>());
}