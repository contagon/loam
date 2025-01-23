from __future__ import annotations
import numpy
import typing
__all__ = ['CONVERGED', 'FeatureExtractionParams', 'INSUFFICIENT_ASSOCIATIONS', 'LidarParams', 'LoamFeatures', 'MAX_ITER', 'Pose3d', 'Quaterniond', 'RegistrationDetail', 'RegistrationIterationInfo', 'RegistrationParams', 'RegistrationTerminationType', 'computeCurvature', 'computeValidPoints', 'extractFeatures', 'registerFeatures']
class FeatureExtractionParams:
    edge_feat_threshold: float
    max_edge_feats_per_sector: int
    max_planar_feats_per_sector: int
    neighbor_points: int
    number_sectors: int
    occlusion_thresh: float
    parallel_thresh: float
    planar_feat_threshold: float
    def __init__(self) -> None:
        ...
class LidarParams:
    def __init__(self, *, scan_lines: int, points_per_line: int, min_range: float, max_range: float) -> None:
        ...
    @property
    def max_range(self) -> float:
        ...
    @property
    def min_range(self) -> float:
        ...
    @property
    def points_per_line(self) -> int:
        ...
    @property
    def scan_lines(self) -> int:
        ...
class LoamFeatures:
    edge_points: list[numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]]
    planar_points: list[numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]]
    def __init__(self) -> None:
        ...
class Pose3d:
    rotation: Quaterniond
    translation: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]
    @staticmethod
    def Identity() -> Pose3d:
        ...
    def __init__(self, rotation: Quaterniond, translation: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    def act(self, point: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    def compose(self, other: Pose3d) -> Pose3d:
        ...
    def inverse(self) -> Pose3d:
        ...
class Quaterniond:
    def __init__(self, *, w: float, x: float, y: float, z: float) -> None:
        ...
    def w(self) -> float:
        ...
    def x(self) -> float:
        ...
    def y(self) -> float:
        ...
    def z(self) -> float:
        ...
class RegistrationDetail:
    iteration_info: list[RegistrationIterationInfo]
    termination_type: RegistrationTerminationType
    def __init__(self) -> None:
        ...
class RegistrationIterationInfo:
    edge_associations: list[tuple[int, int]]
    estimate_update: Pose3d
    plane_associations: list[tuple[int, int]]
    target_T_source_init: Pose3d
    def __init__(self, target_T_source_init: Pose3d, edge_associations: list[tuple[int, int]], plane_associations: list[tuple[int, int]], estimate_update: Pose3d) -> None:
        ...
class RegistrationParams:
    max_avg_point_plane_dist: float
    max_edge_neighbor_dist: float
    max_iterations: int
    max_plane_neighbor_dist: float
    min_associations: int
    min_line_condition_number: float
    min_line_fit_points: int
    min_plane_fit_points: int
    num_edge_neighbors: int
    num_plane_neighbors: int
    position_convergence_thresh: float
    rotation_convergence_thresh: float
    def __init__(self) -> None:
        ...
class RegistrationTerminationType:
    """
    Members:
    
      CONVERGED
    
      MAX_ITER
    
      INSUFFICIENT_ASSOCIATIONS
    """
    CONVERGED: typing.ClassVar[RegistrationTerminationType]  # value = <RegistrationTerminationType.CONVERGED: 0>
    INSUFFICIENT_ASSOCIATIONS: typing.ClassVar[RegistrationTerminationType]  # value = <RegistrationTerminationType.INSUFFICIENT_ASSOCIATIONS: 2>
    MAX_ITER: typing.ClassVar[RegistrationTerminationType]  # value = <RegistrationTerminationType.MAX_ITER: 1>
    __members__: typing.ClassVar[dict[str, RegistrationTerminationType]]  # value = {'CONVERGED': <RegistrationTerminationType.CONVERGED: 0>, 'MAX_ITER': <RegistrationTerminationType.MAX_ITER: 1>, 'INSUFFICIENT_ASSOCIATIONS': <RegistrationTerminationType.INSUFFICIENT_ASSOCIATIONS: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def computeCurvature(input_scan: list[numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]], lidar_params: LidarParams, params: FeatureExtractionParams = ...) -> list[...]:
    ...
def computeValidPoints(input_scan: list[numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]], lidar_params: LidarParams, params: FeatureExtractionParams = ...) -> list[bool]:
    ...
def extractFeatures(input_scan: list[numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]], lidar_params: LidarParams, params: FeatureExtractionParams = ...) -> LoamFeatures:
    ...
def registerFeatures(source: LoamFeatures, target: LoamFeatures, target_T_source_init: Pose3d, params: RegistrationParams = ..., detail: RegistrationDetail = None) -> Pose3d:
    ...
CONVERGED: RegistrationTerminationType  # value = <RegistrationTerminationType.CONVERGED: 0>
INSUFFICIENT_ASSOCIATIONS: RegistrationTerminationType  # value = <RegistrationTerminationType.INSUFFICIENT_ASSOCIATIONS: 2>
MAX_ITER: RegistrationTerminationType  # value = <RegistrationTerminationType.MAX_ITER: 1>
