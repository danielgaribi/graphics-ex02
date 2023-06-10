import numpy as np
import bisect

DOESNT_INTERSECT = -1

def find_sphere_intersect(ray, sphere):
    # acording to ray_casting_presentation page 7 (Geometric Method)
    L = sphere.position - ray.origin_position
    t_ca = np.dot(L, ray.direction)
    
    if t_ca < 0:
        return DOESNT_INTERSECT
    
    d_squared = np.dot(L, L) - t_ca ** 2
    if d_squared > sphere.radius ** 2:
        return DOESNT_INTERSECT
    
    t_hc = np.sqrt(sphere.radius ** 2 - d_squared)
    t = t_ca - t_hc
    return t


def find_plane_intersect(ray, plane):
    # acording to ray_casting_presentation page 9

    # if the dot product is 0, the ray is parallel to the plane (N orthogonal to V)
    if np.dot(ray.direction, plane.normal) == 0:
        return DOESNT_INTERSECT
    
    return -1 * (np.dot(ray.origin_position, plane.normal) + plane.offset) / np.dot(ray.direction, plane.normal)


def find_cube_intersect(ray, cube):
    # acording to http://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/03raytracing1.pdf
    # not acording to https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
    cube_3_axis_min_position = cube.position - cube.scale / 2
    cube_3_axis_max_position = cube.position + cube.scale / 2

    t_x_min = (cube_3_axis_min_position[0] - ray.origin_position[0]) / ray.direction[0]
    t_x_max = (cube_3_axis_max_position[0] - ray.origin_position[0]) / ray.direction[0]
    t_x_enter = min(t_x_min, t_x_max)
    t_x_exit = max(t_x_min, t_x_max)

    t_y_min = (cube_3_axis_min_position[1] - ray.origin_position[1]) / ray.direction[1]
    t_y_max = (cube_3_axis_max_position[1] - ray.origin_position[1]) / ray.direction[1]
    t_y_enter = min(t_y_min, t_y_max)
    t_y_exit = max(t_y_min, t_y_max)

    t_enter = max(t_x_enter, t_y_enter)
    t_exit = min(t_x_exit, t_y_exit)

    if t_enter > t_exit or t_exit < 0:
        return DOESNT_INTERSECT

    t_z_min = (cube_3_axis_min_position[2] - ray.origin_position[2]) / ray.direction[2]
    t_z_max = (cube_3_axis_max_position[2] - ray.origin_position[2]) / ray.direction[2]
    t_z_enter = min(t_z_min, t_z_max)
    t_z_exit = max(t_z_min, t_z_max)

    t_enter = max(t_enter, t_z_enter)
    t_exit = min(t_exit, t_z_exit)

    if t_enter > t_exit or t_exit < 0:
        return DOESNT_INTERSECT
    
    return t_enter

# Return a tuple of (surface, dist) of all objects intersacting the ray as a sorted array by distance (first is the closest to the origin of the ray)
def find_intersection(object_array, ray):
    intersercions = []
    
    for obj in object_array:
        if obj.type == "Sphere":
            dist = find_sphere_intersect(ray, obj)
        elif obj.type == "InfinitePlane":
            dist = find_plane_intersect(ray, obj)
        elif obj.type == "Cube":
            dist = find_cube_intersect(ray, obj)
        else:
            raise ValueError("Unknown object type: {}".format(obj.type))
        
        if dist != DOESNT_INTERSECT:
            bisect.insort(intersercions, (obj, dist), key=lambda x: x[1])
    
    return intersercions