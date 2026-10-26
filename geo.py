"""Local flat-earth coordinates centered on the Metal Shop, in meters:
x = east, y = up, z = north. Fine at the ~60 NM scale this app covers
(JOAN JET's basemap uses the same approximation, in NM)."""
import math

M_PER_NM = 1852.0
M_PER_DEG_LAT = 60.0 * M_PER_NM
FT = 0.3048
KT = M_PER_NM / 3600.0  # knots -> m/s
FPM = FT / 60.0  # feet/minute -> m/s


class Projection:
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))

    def to_xz(self, lat, lon):
        return (lon - self.lon0) * self.m_per_deg_lon, (lat - self.lat0) * M_PER_DEG_LAT


def wrap_deg(a):
    """-> a in (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a
