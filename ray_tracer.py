import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

from intersection import find_intersection
from screen import Screen
from ray import construct_ray_through_pixel

OBJECT_IDX  = 0
DIST_IDX    = 1

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
def compute_surface_normal(surface_obj, surface_dist):
    pass 

def compute_intencity():
    pass

def compute_reflection_ray(vec1, normal):
    teta = np.dot(vec1, normal)
    vec2 = vec1 - 2 * teta * normal
    vec2 = vec2 / np.linalg.norm(vec2)
    return vec2

def compute_diffuse_color(light, light_intensity, intersection_cord, normal):
    L_vec = light.position - intersection_cord
    L_vec = L_vec / np.linalg.norm(L_vec)
    N_dot_L = np.dot(normal, L_vec)
    
    if (N_dot_L < 0 ):
        return np.zeros(3, dtype=float)
    
    return light.color * N_dot_L * light_intensity

def compute_specular_color(light, light_intensity, cam_pos, intersection_cord, normal, shininess):
    L_vec = light.position - intersection_cord
    L_vec = L_vec / np.linalg.norm(L_vec)

    R_vec = compute_reflection_ray(L_vec, normal)
    
    V_vec = cam_pos - intersection_cord
    V_vec = V_vec / np.linalg.norm(V_vec)

    V_dot_R = np.dot(V_vec, R_vec)
    if (V_dot_R < 0):
        return np.zeros(3, dtype=float)
    
    return light.color * np.power(V_dot_R, shininess) * light_intensity * light.specular_intensity
    
def copmute_surface_color(scene_settings, ray, cam_pos, surfaces, surface_idx, material_array, light_array, recursion_level):
    curr_surface_obj    = surfaces[surface_idx][OBJECT_IDX]
    curr_surface_dist   = surfaces[surface_idx][DIST_IDX]
    
    intersection_cord = ray.get_postion(curr_surface_dist)
    normal = compute_surface_normal(curr_surface_obj, curr_surface_dist)

    # Get surface's material
    curr_material = material_array[curr_surface_obj.material_index]

    bg_color         = np.array(scene_settings.background_color)
    diffuse_color    = np.zeros(3, dtype=float)
    specular_color   = np.zeros(3, dtype=float)
    reflection_color = np.zeros(3, dtype=float)

    for light in light_array:
        light_intensity = compute_intencity()
        diffuse_color += compute_diffuse_color(light, light_intensity, intersection_cord, normal)
        specular_color += compute_specular_color(light, light_intensity, cam_pos, intersection_cord, normal, curr_material.shininess)

    if (recursion_level < scene_settings.max_recursions):
        reflaction_ray = compute_reflection_ray(ray, normal)
        reflection_color = compute_pixel_color(scene_settings, reflaction_ray, surfaces, material_array, light_array, cam_pos, recursion_level + 1)

    if ((curr_material.transparency > 0.0) and (surface_idx + 1 < len(surfaces))):
        bg_color *= copmute_surface_color(scene_settings, ray, cam_pos, surfaces, surface_idx + 1, material_array, light_array, 0)

    diffuse_color    *= curr_material.diffuse_color
    specular_color   *= curr_material.specular_color
    reflection_color *= curr_material.reflection_color

    return bg_color * curr_material.transparency + (diffuse_color + specular_color) * (1 - curr_material.transparency) + reflection_color

def compute_pixel_color(scene_settings, ray, object_array, material_array, light_array, cam_pos, recursion_level):
    surfaces = find_intersection(object_array, ray)
    if (len(surfaces) == 0):
        output_color = scene_settings.background_color
    else:
        output_color = copmute_surface_color(scene_settings, ray, cam_pos, surfaces, 0, material_array, light_array, recursion_level)
    
    return np.array(output_color)

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

    # TODO - remove - basic init
    img_width = 500
    img_height = 500
    screen = Screen(camera, img_width, img_height)

    image_array = np.zeros((img_height, img_width, 3), dtype=float)
    for w in range(img_width):
        for h in range(img_height):
            ray = construct_ray_through_pixel(screen, h, w)
            image_array[h, w] = compute_pixel_color(scene_settings, ray, object_array, material_array, light_array, camera.position, 0)

    # Save the output image
    save_image(image_array)

if __name__ == '__main__':
    main()
