"""Canonical camera and disjoint Search/VS geometry APIs (metres and bits)."""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CameraConfiguration:
    f_m: float = 0.035
    image_width_m: float = 0.0156
    image_length_m: float = 0.0235

    @property
    def b1(self):
        return 2 * self.f_m / self.image_width_m


IMAGE_PLANE_WIDTH_M = 0.0156
IMAGE_PLANE_LENGTH_M = 0.0235
SEARCH_FOCAL_LENGTH_M = 0.0175
VS_FOCAL_LENGTH_M = 0.035
SEARCH_CAMERA = CameraConfiguration(
    f_m=SEARCH_FOCAL_LENGTH_M,
    image_width_m=IMAGE_PLANE_WIDTH_M,
    image_length_m=IMAGE_PLANE_LENGTH_M,
)
VS_CAMERA = CameraConfiguration(
    f_m=VS_FOCAL_LENGTH_M,
    image_width_m=IMAGE_PLANE_WIDTH_M,
    image_length_m=IMAGE_PLANE_LENGTH_M,
)
# Compatibility alias for callers that historically imported the VS camera.
# Production Search geometry never reads this alias.
CAMERA = VS_CAMERA
DEFAULT_ROI_RADIUS_M = 80.0
VS_PACKET_MAX_BITS = 31_600.0
VS_QUALITY_WEIGHT = 0.8
VS_PROXIMITY_WEIGHT = 0.2
VISUAL_SENSING_CONTRACT_VERSION = (
    "task-selected-search-17p5mm-vs-35mm-valid-capture-v3"
)
DISTANCE_EPSILON_M = 1e-9
SINGULAR_RAY_EPSILON = 1e-8


def visual_sensing_metadata():
    return {
        "version": VISUAL_SENSING_CONTRACT_VERSION,
        "camera": asdict(VS_CAMERA),
        "camera_compatibility_alias": "camera is the legacy VS-camera alias",
        "shared_image_plane": {
            "image_width_m": IMAGE_PLANE_WIDTH_M,
            "image_length_m": IMAGE_PLANE_LENGTH_M,
        },
        "search_camera": asdict(SEARCH_CAMERA),
        "vs_camera": asdict(VS_CAMERA),
        "default_roi_radius_m": DEFAULT_ROI_RADIUS_M,
        "roi_radius_source": "scenario target object",
        "search_model": (
            "nadir rectangle; undiscovered ROI-center-inclusive detection; "
            "Search contributors excluding permanent GS gateway"
        ),
        "vs_model": "oblique camera aimed at assigned ROI",
        "camera_mode_selection": (
            "Search task -> Search camera; FOV or FOV+COM -> VS camera; "
            "COM-only, Hovering and permanent GS gateway -> no sensing"
        ),
        "camera_mode_state": (
            "derived from current task types; no independent per-step transition"
        ),
        "minimum_resolution_hard_constraint": {
            "search": False,
            "vs": False,
        },
        "coverage_model": "analytic circle-convex-oblique-polygon intersection / ROI area",
        "image_quantity": "ROI area / same oblique footprint area; raw I may exceed 1",
        "packet_max_bits": VS_PACKET_MAX_BITS,
        "packet_size": "packet_max_bits * min(max(I,0),1)",
        "quality_weight": VS_QUALITY_WEIGHT,
        "proximity_weight": VS_PROXIMITY_WEIGHT,
        "pair_score": "0.8 * coverage * min(I,1) + 0.2 * G",
        "proximity": "min(1,b1*relative_altitude/(horizontal_distance+1e-12))",
        "geometry_validity": "d2D <= b1*relative_altitude; positive altitude; all corner rays downward",
        "b1": VS_CAMERA.b1,
        "distance_epsilon_m": DISTANCE_EPSILON_M,
        "singular_ray_epsilon": SINGULAR_RAY_EPSILON,
        "assignment_eligibility": "independent of sensing_valid_now; existing target/role/energy constraints",
        "packet_generation": "assigned VS generates only while sensing_valid_now is true; no coverage threshold",
        "invalid_sensing": "I=c=Q=0; no packets or deferred rate credit; fractional credit cleared",
        "rate_credit_lifecycle": "per UAV/ROI/task; cleared on invalid sensing, removal, reassignment and episode reset",
        "capture_snapshot": "physical size, coverage, raw image quantity, ROI identity and task identity",
    }


@dataclass(frozen=True)
class SearchFootprint:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def contains(self, x, y):
        return bool(math.isfinite(x) and math.isfinite(y)
                    and self.xmin - DISTANCE_EPSILON_M <= x <= self.xmax + DISTANCE_EPSILON_M
                    and self.ymin - DISTANCE_EPSILON_M <= y <= self.ymax + DISTANCE_EPSILON_M)


def search_footprint(position, ground_z=0.0, camera=SEARCH_CAMERA):
    """Fixed nadir footprint, independent of any candidate ROI or VS pose."""
    x, y, z = map(float, position)
    altitude = z - float(ground_z)
    if not all(map(math.isfinite, (x, y, altitude))) or altitude <= 0:
        return None
    width = altitude * camera.image_width_m / camera.f_m
    height = altitude * camera.image_length_m / camera.f_m
    return SearchFootprint(x-width/2, x+width/2, y-height/2, y+height/2)


def _cross(a, b):
    return float(a[0]*b[1] - a[1]*b[0])


def circle_polygon_intersection_area(polygon, center, radius):
    """Deterministic edge integration: circle crossings split triangles/sectors.

    Coordinates are normalized to a unit circle before integrating each edge.
    This supports either winding and uses no bitmap or stochastic sampling.
    """
    vertices = np.asarray(polygon, dtype=float)
    radius = float(radius)
    if (vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3
            or not np.isfinite(vertices).all() or not np.isfinite(center).all()
            or not math.isfinite(radius) or radius <= 0):
        return 0.0
    vertices = (vertices - np.asarray(center, dtype=float)) / radius
    pieces = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        delta = b - a
        length2 = float(delta @ delta)
        if length2 <= 1e-30:
            continue
        closest_t = -float(a @ delta) / length2
        closest = a + closest_t * delta
        height2 = 1 - float(closest @ closest)
        cuts = [0.0, 1.0]
        if height2 > 0:
            half = math.sqrt(height2 / length2)
            cuts.extend(t for t in (closest_t-half, closest_t+half) if 0 < t < 1)
        cuts.sort()
        for t0, t1 in zip(cuts, cuts[1:]):
            p, q = a+t0*delta, a+t1*delta
            middle = (p+q)/2
            pieces.append((_cross(p, q) if float(middle @ middle) < 1.0
                           else math.atan2(_cross(p, q), float(p @ q))) / 2)
    return min(max(abs(math.fsum(pieces))*radius**2, 0.0), math.pi*radius**2)


@dataclass(frozen=True)
class VSGeometry:
    horizontal_distance: float
    relative_altitude: float
    b1: float
    model_range_valid: bool
    sensing_valid_now: bool
    polygon: tuple
    footprint_area: float
    image_quantity: float
    coverage_ratio: float
    proximity: float
    diagnostics: dict

    @property
    def quality(self):
        return self.coverage_ratio * min(self.image_quantity, 1.0)

    @property
    def pair_score(self):
        return VS_QUALITY_WEIGHT*self.quality + VS_PROXIMITY_WEIGHT*self.proximity


def vs_geometry(
    uav_position,
    roi_position,
    radius=DEFAULT_ROI_RADIUS_M,
    camera=VS_CAMERA,
):
    """Aim at ROI and project four camera rays onto its ground plane.

    Sensor width is along the tilt plane, height cross-track. Nadir uses +x
    bearing deterministically. The inclusive range boundary has a horizontal
    corner ray: model_range_valid is then true but sensing_valid_now is false.
    """
    uav, roi = np.asarray(uav_position, dtype=float), np.asarray(roi_position, dtype=float)
    radius = float(radius)
    distance = altitude = proximity = 0.0
    range_valid = False

    def invalid(reason):
        return VSGeometry(distance, altitude, camera.b1, range_valid, False,
                          (), 0.0, 0.0, 0.0, proximity, {"reason": reason})

    if uav.shape != (3,) or roi.shape != (3,) or not np.isfinite((uav, roi)).all():
        return invalid("non_finite_position")
    altitude = float(uav[2]-roi[2])
    delta = roi[:2]-uav[:2]
    distance = math.hypot(*delta)
    if not math.isfinite(altitude) or not math.isfinite(distance) or altitude <= 0:
        altitude = altitude if math.isfinite(altitude) else 0.0
        distance = distance if math.isfinite(distance) else 0.0
        return invalid("invalid_relative_altitude_or_distance")
    limit = camera.b1*altitude
    proximity = 1.0 if distance <= limit else min(1.0, limit/(distance+1e-12))
    range_valid = distance <= limit + DISTANCE_EPSILON_M
    if not math.isfinite(radius) or radius <= 0:
        return invalid("invalid_roi_radius")
    if not range_valid:
        return invalid("outside_oblique_model_range")
    bearing = delta/distance if distance > 1e-12 else np.array([1.0, 0.0])
    slant = math.hypot(distance, altitude)
    sin_t, cos_t = distance/slant, altitude/slant
    forward = np.array([sin_t*bearing[0], sin_t*bearing[1], -cos_t])
    sensor_x = np.array([cos_t*bearing[0], cos_t*bearing[1], sin_t])
    sensor_y = np.array([-bearing[1], bearing[0], 0.0])
    vertices, margins = [], []
    for sx, sy in ((-1,-1), (1,-1), (1,1), (-1,1)):
        ray = (forward + sx*camera.image_width_m/(2*camera.f_m)*sensor_x
               + sy*camera.image_length_m/(2*camera.f_m)*sensor_y)
        downward = -float(ray[2])
        if downward <= SINGULAR_RAY_EPSILON:
            return invalid("singular_or_non_intersecting_corner_ray")
        margins.append(downward)
        vertices.append(-delta + altitude/downward*ray[:2])
    vertices = np.asarray(vertices)
    if not np.isfinite(vertices).all():
        return invalid("non_finite_polygon")
    turns = [_cross(vertices[(i+1)%4]-vertices[i],
                    vertices[(i+2)%4]-vertices[(i+1)%4]) for i in range(4)]
    area = math.fsum(_cross(a,b) for a,b in zip(vertices, np.roll(vertices,-1,axis=0)))/2
    if not math.isfinite(area) or area <= 1e-12 or min(turns) <= 0:
        return invalid("degenerate_or_non_convex_polygon")
    roi_area = math.pi*radius**2
    quantity = roi_area/area
    coverage = circle_polygon_intersection_area(vertices, (0,0), radius)/roi_area
    polygon = tuple(tuple(map(float, point+roi[:2])) for point in vertices)
    if not np.isfinite(polygon).all() or not math.isfinite(quantity):
        return invalid("non_finite_geometry")
    return VSGeometry(distance, altitude, camera.b1, True, True, polygon, area,
                      quantity, min(max(coverage,0.0),1.0), proximity,
                      {"reason": "valid", "minimum_downward_ray": min(margins),
                       "bearing_radians": math.atan2(bearing[1],bearing[0])})
