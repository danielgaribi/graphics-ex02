import argparse
from PIL import Image
import numpy as np
import random

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

from intersection import find_intersection
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

def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")

# return the normal of the surface - normalized
def compute_surface_normal(surface_obj, intersection_cord):
    if (surface_obj.__class__.__name__ == "Sphere"):
        normal = (intersection_cord - surface_obj.position) / np.linalg.norm(intersection_cord - surface_obj.position)
    elif (surface_obj.__class__.__name__ == "InfinitePlane"):
        normal = surface_obj.normal
    elif (surface_obj.__class__.__name__ == "InfinitePlane"):
        # TODO: compute cube normal
        normal = [1,1,1]
    else: 
        raise ValueError("Unknown object type: {}".format(obj.type))

    return np.array(normal)

# TODO: double check 
def compute_intensity(scene_settings, light, intersection_coord, surface_obj, object_array):
    N = int(scene_settings.root_number_shadow_rays)

    # Construct a light ray
    center_light_ray = construct_ray(intersection_coord, light.position)

    # Find a plane perpendicular to the ray
    width_vec = np.cross(center_light_ray.direction, np.array([1, 0, 0]))
    if (width_vec == 0).all():
        width_vec = np.cross(center_light_ray.direction, np.array([0, 1, 0]))
    width_vec /= np.linalg.norm(width_vec)

    height_vec = np.cross(center_light_ray.direction, width_vec)

    # Define a rectangle on the plane centered at the light source
    rect_bottom_left = light.position - (light.radius / 2) * width_vec - (light.radius / 2) * height_vec

    # Divide the rectangle into a grid of N x N cells
    cell_size = light.radius / N

    # Re-sizing of width_vec and height_vec with step size of cell_size
    width_vec *= cell_size
    height_vec *= cell_size

    # Initialize a variable to count rays hitting the surface
    rays_hit = 0
    for i in range(N):
        for j in range(N):
            # Generate a random point inside each cell
            rand_x = random.random()
            rand_y = random.random()
            cell_position = rect_bottom_left + (i + rand_x) * width_vec + (j + rand_y) * height_vec

            # Construct a shadow ray from the random point to the intersection coordinate
            grid_cell_ray = construct_ray(intersection_coord, cell_position)

            # Find the objects the shadow ray intersects with
            surface_dist = find_intersection(object_array, grid_cell_ray)

            # TODO: is needed?
            if (len(surface_dist) == 0):
                continue

            first_surface_dist = surface_dist[0]
            if (np.linalg.norm(grid_cell_ray.get_postion(first_surface_dist[DIST_IDX]) - intersection_coord) < EPSILON):
                rays_hit += 1
            

    light_rays_ratio = float(rays_hit) / float(N * N)
    light_intensity = 1 * (1 - light.shadow_intensity) + (light.shadow_intensity * light_rays_ratio)
    return light_intensity


def compute_reflection_direction(vec1, normal):
    teta = np.dot(vec1, normal)
    vec2 = vec1 - 2 * teta * normal
    vec2 = vec2 / np.linalg.norm(vec2)
    return vec2

# Acording to ray_casting_presentation page 42
def compute_diffuse_color(light, light_intensity, intersection_cord, normal):
    L_vec = light.position - intersection_cord
    L_vec = L_vec / np.linalg.norm(L_vec)

    N_dot_L = np.dot(normal, L_vec)

    # TODO: debug 
    # print(f"L_vec: {L_vec}, normal: {normal}")
    # print(f"N_dot_L: {N_dot_L}")
    
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
    curr_surface_obj  = surfaces[surface_idx][OBJECT_IDX]
    curr_surface_dist = surfaces[surface_idx][DIST_IDX]
    
    intersection_cord = ray.get_postion(curr_surface_dist)
    normal = compute_surface_normal(curr_surface_obj, intersection_cord)

    # TODO: debug 
    print(f"ray.origin_position: {ray.origin_position}, ray.direction: {ray.direction}")
    print(f"curr_surface_obj: {curr_surface_obj}, curr_surface_dist: {curr_surface_dist}")
    print(f"intersection_cord: {intersection_cord}, normal: {normal}")

    # Get surface's material
    curr_material = material_array[curr_surface_obj.material_index - 1] # -1 because material index start from 1 TODO: check if needed

    bg_color         = np.array(scene_settings.background_color)
    diffuse_color    = np.zeros(3, dtype=float)
    specular_color   = np.zeros(3, dtype=float)

    for light in light_array:
        light_intensity = compute_intensity(scene_settings, light, intersection_cord, curr_surface_obj, object_array)
        diffuse_color  += compute_diffuse_color(light, light_intensity, intersection_cord, normal)
        specular_color += compute_specular_color(light, light_intensity, cam_pos, intersection_cord, normal, curr_material.shininess)
        
        # TODO: debug 
        # print(f"light - position: {light.position}, color: {light.color}, specular_intensity: {light.specular_intensity}, shadow_intensity: {light.shadow_intensity}, radius: {light.radius}")
        # print(f"light_intensity: {light_intensity}")
        # print(f"diffuse_color (light): {diffuse_color}")

    reflaction_direction = compute_reflection_direction(ray.direction, normal)
    reflaction_ray = construct_ray(intersection_cord, reflaction_direction)
    # TODO: check if needed - Advance EPSILON to avoid intersection with the same object
    # reflaction_ray = construct_ray(reflaction_ray.get_postion(EPSILON), reflaction_direction)

    reflection_color = compute_pixel_color(scene_settings, reflaction_ray, object_array, material_array, light_array, cam_pos, recursion_level + 1)

    if ((curr_material.transparency > 0.0) and (surface_idx + 1 < len(surfaces))):
        bg_color *= copmute_surface_color(scene_settings, ray, cam_pos, surfaces, surface_idx + 1, object_array, material_array, light_array, recursion_level + 1)

    diffuse_color    *= curr_material.diffuse_color    
    specular_color   *= curr_material.specular_color
    reflection_color *= curr_material.reflection_color

    # TODO: debug 
    # print(f"curr_material.diffuse_color: {curr_material.diffuse_color}")
    # print(f"diffuse_color (add material factor): {diffuse_color}")
    # print(f"specular_color: {specular_color}")
    # print(f"reflection_color: {reflection_color}")

    output_color = bg_color * curr_material.transparency + (diffuse_color + specular_color) * (1 - curr_material.transparency) + reflection_color
    # TODO: check if needed - Add clipping if above 1
    # print(f"output_color[{output_color}] = bg_color[{bg_color}] * curr_material.transparency[{curr_material.transparency}] \n\t+ (diffuse_color[{diffuse_color}] + specular_color[{specular_color}]) * \n\t(1 - curr_material.transparency[{curr_material.transparency}]) + reflection_color[{reflection_color}]\n\n")

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

def objects_to_numpy(camera, object_array):
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

    objects_to_numpy(camera, object_array)

    img_width = args.width
    img_height = args.width
    screen = Screen(camera, img_width, img_height)

    image_array = np.zeros((img_height, img_width, 3), dtype=float)
    for w in range(img_width):
        for h in range(img_height):
            ray = construct_ray_through_pixel(screen, h, w)
            output_color = compute_pixel_color(scene_settings, ray, object_array, material_array, light_array, camera.position, 0)
            image_array[h, w] = np.clip(output_color * 255, 0, 255)  # Clip color values to [0, 255]
            
            # TODO: debug 
            if (not np.all(image_array[h, w] == 255)):
                print(f"output: [h({h}),w({w})] = {image_array[h, w]}\n")

    # TODO: use arg?
    output_image_path = args.output_image

    # Save the output image
    save_image(image_array)

if __name__ == '__main__':
    main()