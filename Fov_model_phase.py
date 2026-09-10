"""Compatibility facade over the canonical task-selected sensing modes."""
import math
from visual_sensing import (
    DEFAULT_ROI_RADIUS_M,
    VS_CAMERA,
    search_footprint,
    vs_geometry,
)


class FovModel:
    def __init__(self, z_u=100.0, gamma_g=DEFAULT_ROI_RADIUS_M):
        self.z_u, self.gamma_g = z_u, gamma_g
        self.f = VS_CAMERA.f_m
        self.wl = VS_CAMERA.image_width_m
        self.il = VS_CAMERA.image_length_m

    def calculate_fov_single(self, x_u, y_u, z_u, x_g, y_g, z_g):
        geometry = vs_geometry((x_u,y_u,z_u), (x_g,y_g,z_g), self.gamma_g)
        return geometry.image_quantity, math.hypot(geometry.horizontal_distance, geometry.relative_altitude)

    def get_ground_fov_size(self, z_u):
        footprint = search_footprint((0,0,z_u))
        return (footprint.width, footprint.height) if footprint else (0.0,0.0)
