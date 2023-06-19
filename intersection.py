import numpy as np


EPSILON = 10 ** -9
DOESNT_INTERSECT = -1

def find_sphere_intersect(sphere, ray):
    # acording to ray_casting_presentation page 7 (Geometric Method)
    L = sphere.position - ray.origin_position
    t_ca = np.dot(L, ray.direction)
    
    if t_ca < 0:
        return DOESNT_INTERSECT
    
    d_squared = np.dot(L, L) - np.power(t_ca, 2)
    r_squeared = np.power(sphere.radius, 2)
    if d_squared > r_squeared:
        return DOESNT_INTERSECT
    
    t_hc = np.sqrt(r_squeared - d_squared)
    return t_ca - t_hc


def find_plane_intersect(plane, ray):
    # acording to ray_casting_presentation page 9

    # if the dot product is 0, the ray is parallel to the plane (N orthogonal to V)
    dot_product = np.dot(ray.direction, plane.normal)
    if dot_product == 0:
        return DOESNT_INTERSECT
    
    return (plane.offset - np.dot(ray.origin_position, plane.normal)) / dot_product


def find_cube_intersect(cube, ray):
    # acording to http://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/03raytracing1.pdf
    # not acording to https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
    cube_3_axis_min_position = cube.position - cube.scale / 2
    cube_3_axis_max_position = cube.position + cube.scale / 2

    if ray.direction[0] == 0:
        ray.direction[0] = EPSILON
    if ray.direction[1] == 0:
        ray.direction[1] = EPSILON
    if ray.direction[2] == 0:
        ray.direction[2] = EPSILON

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
    intersections = []
    for obj in object_array:
        dist = obj.find_intersection(obj, ray)

        if dist != DOESNT_INTERSECT:
            intersections.append((obj, dist))
    
    intersections.sort(key=lambda x: x[1])  # Sort intersections by distance
    return intersections

def is_ray_hit(object_array, ray, max_dist, prior_object):
    if prior_object is not None:
        dist = prior_object.find_intersection(prior_object, ray)
                
        if dist != DOESNT_INTERSECT and dist < max_dist:
            return True, prior_object
    
    for obj in object_array:
        dist = obj.find_intersection(obj, ray)
        
        if dist != DOESNT_INTERSECT and dist < max_dist:
            return True, obj
    
    return False, prior_object

def update_find_intersection_func_for_all_objects(object_array):
    for obj in object_array:
        if obj.__class__.__name__ == "Sphere":
            obj.find_intersection = find_sphere_intersect
        elif obj.__class__.__name__ == "InfinitePlane":
            obj.find_intersection = find_plane_intersect
        elif obj.__class__.__name__ == "Cube":
            obj.find_intersection = find_cube_intersect
        else:
            raise ValueError("Unknown object type: {}".format(obj.type))