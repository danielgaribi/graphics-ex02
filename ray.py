import numpy as np

class Ray:
    def __init__(self, origin_position, direction):
        self.origin_position = origin_position
        self.direction = direction

    def get_postion(self, distance):
        return self.origin_position + distance * self.direction
    
def construct_ray_through_pixel(screen, h, w): 
    origin_position = screen.camera.position
    pixel_position = screen.get_pix_location(h, w)
    return construct_ray(origin_position, pixel_position)

def construct_ray(start_point, end_point):
    direction = end_point - start_point
    direction /= np.linalg.norm(direction)
    return Ray(start_point, direction)
