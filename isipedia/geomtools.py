import numpy as np
from skimage.measure import find_contours
import rasterio
import rasterio.mask
import shapely.ops
from shapely.geometry import LineString, Point, MultiPoint, Polygon, MultiLineString, GeometryCollection, LinearRing, MultiPolygon
from icedata.tools import coords_to_gdal_transform

def coords_to_gdal_transform(x, y):
    ni, nj = y.size, x.size
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    return rasterio.Affine(dx, 0, x[0]-dx/2, 0, dy, y[0]-dy/2)


def find_contours2(coords, values, level, **kw):
    """Like find_contours, but return line in coordinate values
    """
    contours = find_contours(values, level, **kw)
    contours2 = []
    x, y = coords
    dx, dy = x[1] - x[0], y[1]-y[0]
    for c in contours:
        ii, jj = np.array(c).T
        xx, yy = x[0]+jj*dx, y[0]+ii*dy
        line = np.array((xx, yy)).T
        contours2.append(line)
    return contours2


def mask_to_polygon(coords, mask, tol=None, minarea=0):
    x, y = coords
    mask2 = mask + 0.
    mask2[:,[0,-1]] = 0  # close all contours
    mask2[[0,-1],:] = 0
    contours = find_contours2(coords, mask2, level=0.5)
    
    rings = [LinearRing(c) for c in contours]
    
    if tol is not None:
        rings = [r.simplify(tol) for r in rings]   
        
    exteriors = [Polygon(p) for p in rings if p.is_ccw]
    interiors = [Polygon(p) for p in rings if not p.is_ccw]
    
    if minarea:
        exteriors = [p for p in exteriors if p.area > minarea]
        interiors = [p for p in interiors if p.area > minarea]

    mpoly = shapely.ops.unary_union(exteriors)
    return mpoly.symmetric_difference(MultiPolygon(interiors))        


def polygon_to_mask(geom, coords, all_touched=False):
    """return a numpy mask array which is True when it intersects with geometry

    all_touched : boolean, optional
        If True, all pixels touched by geometries will be burned in.  If
        false, only pixels whose center is within the polygon or that
        are selected by Bresenham's line algorithm will be burned in.
    """
    geoms = getattr(geom, 'geoms', [geom])
    shape = coords[1].size, coords[0].size
    transform = coords_to_gdal_transform(*coords)
    return rasterio.mask.geometry_mask(geoms, shape, transform, invert=True, all_touched=all_touched)
