import cv2

class CV_API:
    
    def __init__(self):
        pass
    
    def load_image(self, file_path):
        """Load an image from file."""
        return cv2.imread(file_path)
    
    def save_image(self, image, file_path):
        """Save an image to file."""
        cv2.imwrite(file_path, image)
    
    def resize_image(self, image, width=None, height=None, keep_aspect_ratio=True):
        """Resize an image."""
        if keep_aspect_ratio:
            h, w, _ = image.shape
            if width is None:
                width = int(w * height / h)
            elif height is None:
                height = int(h * width / w)
            else:
                aspect_ratio = w / h
                if width / height > aspect_ratio:
                    width = int(height * aspect_ratio)
                else:
                    height = int(width / aspect_ratio)
        return cv2.resize(image, (width, height))
    
    def crop_image(self, image, x, y, width, height):
        """Crop a rectangular region from an image."""
        return image[y:y+height, x:x+width]
    
    def rotate_image(self, image, angle):
        """Rotate an image by the specified angle in degrees."""
        h, w, _ = image.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h))
