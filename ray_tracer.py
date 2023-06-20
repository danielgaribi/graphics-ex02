import argparse
from PIL import Image
import numpy as np
import random

# TODO: debug
from tqdm import tqdm

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

from intersection import find_intersection, is_ray_hit, update_find_intersection_func_for_all_objects
from screen import Screen
from ray import construct_ray_through_pixel, construct_ray

OBJECT_IDX  = 0
DIST_IDX    = 1

EPSILON = 10 ** -9

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects

def save_image(image_array, output_image):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(output_image)

def update_get_normal_func_for_all_objects(object_array):
    for obj in object_array:
        if obj.__class__.__name__ == "Sphere":
            obj.get_noraml = lambda object, intersection_cord: (intersection_cord - object.position) / np.linalg.norm(intersection_cord - object.position)
        elif obj.__class__.__name__ == "InfinitePlane":
            obj.get_noraml = lambda object, intersection_cord: object.normal
        elif obj.__class__.__name__ == "Cube":
            obj.get_noraml = lambda object, intersection_cord: np.array(compute_cube_normal(object, intersection_cord))
        else:
            raise ValueError("Unknown object type: {}".format(obj.type))

def compute_cube_normal(cube, intersection_cord):
    dist_from_center_to_edge = cube.scale / 2
    # Intersection is on the upper x-parallel plane
    if   abs((intersection_cord[0] - cube.position[0]) - dist_from_center_to_edge) < EPSILON:
        return [1, 0, 0]
    # Intersection is on the lower x-parallel plane
    elif abs((intersection_cord[0] - cube.position[0]) - dist_from_center_to_edge) < EPSILON:
        return [-1, 0, 0]
    # Intersection is on the upper y-parallel plane
    elif abs((intersection_cord[1] - cube.position[1]) - dist_from_center_to_edge) < EPSILON:
        return [0, 1, 0]
    # Intersection is on the lower y-parallel plane
    elif abs((intersection_cord[1] - cube.position[1]) - dist_from_center_to_edge) < EPSILON:
        return [0, -1, 0]
    # Intersection is on the upper z-parallel plane
    elif abs((intersection_cord[2] - cube.position[2]) - dist_from_center_to_edge) < EPSILON:
        return [0, 0, 1]
    # Intersection is on the lower z-parallel plane
    else:
        return [0, 0, -1]

def compute_intensity(scene_settings, light, intersection_coord, surface_obj, object_array):
    N = int(scene_settings.root_number_shadow_rays)

    # Construct a light ray
    center_light_ray, _ = construct_ray(light.position, end_point=intersection_coord)

    # Find a plane perpendicular to the ray
    width_vec = np.cross(center_light_ray.direction, np.array([1, 0, 0]))
    if (width_vec == 0).all():
        width_vec = np.cross(center_light_ray.direction, np.array([0, 1, 0]))
    width_vec /= np.linalg.norm(width_vec)

    height_vec = np.cross(center_light_ray.direction, width_vec) 
    height_vec /= np.linalg.norm(height_vec)

    # Define a rectangle on the plane centered at the light source
    rect_bottom_left = light.position - (light.radius / 2) * width_vec - (light.radius / 2) * height_vec

    # Divide the rectangle into a grid of N x N cells
    cell_size = light.radius / N

    # Re-sizing of width_vec and height_vec with step size of cell_size
    width_vec *= cell_size
    height_vec *= cell_size

    # Initialize a variable to count rays hitting the surface
    rays_hit = 0
    prior_obj = None
    for i in range(N):
        for j in range(N):
            # Generate a random point inside each cell
            cell_position = rect_bottom_left + (i + random.random()) * width_vec + (j + random.random()) * height_vec

            # Construct a shadow ray from the random point to the intersection coordinate
            grid_cell_ray, max_dist = construct_ray(cell_position, end_point=intersection_coord)

            # Find the objects the shadow ray intersects with
            is_hit_other_objects, prior_obj = is_ray_hit(object_array, grid_cell_ray, max_dist - EPSILON, prior_obj)

            if not is_hit_other_objects:
                rays_hit += 1
            

    light_rays_ratio = rays_hit / (N ** 2)
    light_intensity = 1 - light.shadow_intensity + light.shadow_intensity * light_rays_ratio
    return light_intensity

def compute_reflection_direction(vec1, normal):
    teta = np.dot(vec1, normal)
    vec2 = vec1 - 2 * teta * normal
    return vec2 / np.linalg.norm(vec2)

# Acording to ray_casting_presentation page 42
def compute_diffuse_color(light, light_intensity, intersection_cord, normal):
    L_vec = light.position - intersection_cord
    L_vec = L_vec / np.linalg.norm(L_vec)

    N_dot_L = np.dot(normal, L_vec)    
    if (N_dot_L < 0):
        return np.zeros(3, dtype=float)
    
    return np.array(light.color) * N_dot_L * light_intensity

# Acording to ray_casting_presentation page 45
def compute_specular_color(light, light_intensity, cam_pos, intersection_cord, normal, shininess):
    L_vec = intersection_cord - light.position
    L_vec = L_vec / np.linalg.norm(L_vec)

    R_vec = compute_reflection_direction(L_vec, normal)
    
    V_vec = cam_pos - intersection_cord
    V_vec = V_vec / np.linalg.norm(V_vec)

    V_dot_R = np.dot(V_vec, R_vec)
    if (V_dot_R < 0):
        return np.zeros(3, dtype=float)
    
    return np.array(light.color) * np.power(V_dot_R, shininess) * light_intensity * light.specular_intensity
    
def copmute_surface_color(scene_settings, ray, cam_pos, surfaces, surface_idx, object_array, material_array, light_array, recursion_level):
    curr_surface_obj, curr_surface_dist = surfaces[surface_idx]
    
    intersection_cord = ray.get_postion(curr_surface_dist)
    normal = curr_surface_obj.get_noraml(curr_surface_obj, intersection_cord)

    # Get surface's material
    curr_material = material_array[curr_surface_obj.material_index - 1] # -1 because material index start from 1 TODO: check if needed

    bg_color         = np.array(scene_settings.background_color)
    diffuse_color    = np.zeros(3, dtype=float)
    specular_color   = np.zeros(3, dtype=float)

    for light in light_array:
        light_intensity = compute_intensity(scene_settings, light, intersection_cord, curr_surface_obj, object_array)
        diffuse_color  += compute_diffuse_color(light, light_intensity, intersection_cord, normal)
        specular_color += compute_specular_color(light, light_intensity, cam_pos, intersection_cord, normal, curr_material.shininess)
        
    reflaction_direction = compute_reflection_direction(ray.direction, normal)
    reflaction_ray, _ = construct_ray(intersection_cord, direction=reflaction_direction)
    # TODO: check if needed - Advance EPSILON to avoid intersection with the same object
    # reflaction_ray = construct_ray(reflaction_ray.get_postion(EPSILON), intersection_cord + reflaction_direction)

    reflection_color = compute_pixel_color(scene_settings, reflaction_ray, object_array, material_array, light_array, cam_pos, recursion_level + 1)

    if ((curr_material.transparency > 0.0) and (surface_idx + 1 < len(surfaces))):
        # TODO: change recursion_level to zero? [https://moodle.tau.ac.il/mod/forum/discuss.php?d=98419]
        bg_color *= copmute_surface_color(scene_settings, ray, cam_pos, surfaces, surface_idx + 1, object_array, material_array, light_array, 0)

    diffuse_color    *= curr_material.diffuse_color
    specular_color   *= curr_material.specular_color
    reflection_color *= curr_material.reflection_color
    
    output_color = bg_color * curr_material.transparency + (diffuse_color + specular_color) * (1 - curr_material.transparency) + reflection_color
    return output_color

def compute_pixel_color(scene_settings, ray, object_array, material_array, light_array, cam_pos, recursion_level):
    if (recursion_level == scene_settings.max_recursions): 
        return np.array(scene_settings.background_color)
    
    surfaces = find_intersection(object_array, ray)
    if (len(surfaces) == 0):
        output_color = scene_settings.background_color
    else:
        output_color = copmute_surface_color(scene_settings, ray, cam_pos, surfaces, 0, object_array, material_array, light_array, recursion_level)
    
    return np.array(output_color)

def objects_to_numpy(camera, object_array, light_array, scene_settings):
    camera.position = np.array(camera.position)
    camera.look_at = np.array(camera.look_at)
    camera.up_vector = np.array(camera.up_vector)

    for obj in object_array:
        if obj.__class__.__name__ == "Sphere":
            obj.position = np.array(obj.position)
        elif obj.__class__.__name__ == "InfinitePlane":
            obj.normal = np.array(obj.normal)
        elif obj.__class__.__name__ == "Cube":
            obj.position = np.array(obj.position)
        else:
            raise ValueError("Unknown object type: {}".format(obj.type))
    
    update_find_intersection_func_for_all_objects(object_array)
    update_get_normal_func_for_all_objects(object_array)

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Divide objects into materials, lights, and other objects
    material_array = []
    light_array = []
    object_array = []
    for obj in objects:
        if isinstance(obj, Material):
            material_array.append(obj)
        elif isinstance(obj, Light):
            light_array.append(obj)
        else:
            object_array.append(obj)

    objects_to_numpy(camera, object_array, light_array, scene_settings)

    img_width = args.width
    img_height = args.width
    screen = Screen(camera, img_width, img_height)

    image_array = np.zeros((img_height, img_width, 3), dtype=float)
    for w in tqdm(range(img_width), desc="width"):
        for h in tqdm(range(img_height), desc="height", leave=False):
            ray = construct_ray_through_pixel(screen, h, w)
            output_color = compute_pixel_color(scene_settings, ray, object_array, material_array, light_array, camera.position, 0)
            image_array[h, w] = np.clip(output_color * 255, 0, 255)  # Clip color values to [0, 255]

    # Save the output image
    save_image(image_array, args.output_image)

if __name__ == '__main__':
    main()