import numpy as np

class Screen:
    def __init__(self, camera, img_pix_width, img_pix_height):
        self.camera = camera
        self.img_pix_width = img_pix_width
        self.img_pix_height = img_pix_height

        self.pixel_width = camera.screen_width / img_pix_width
        self.width = camera.screen_width
        self.height = self.pixel_width * img_pix_height

        # TODO: Garibi - Runtime error
        self.through_vector = (camera.look_at - camera.position) / np.linalg.norm(camera.look_at - camera.position)
        self.right_vector = np.cross(camera.up_vector, self.through_vector) / np.linalg.norm(np.cross(camera.up_vector, self.through_vector))
        self.up_vector = np.cross(self.through_vector, self.right_vector)

        self.middle_pixel_position = camera.position + self.through_vector * camera.screen_distance

        self.top_left_pixel_position = self.middle_pixel_position +\
            self.up_vector * (self.width / 2) - self.right_vector * (self.height / 2)
        
    def get_pix_location(self, h, w):
        return self.top_left_pixel_position\
                - h * self.pixel_width * self.up_vector \
                + w * self.pixel_width * self.right_vector