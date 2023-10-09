"""
This is a super simple script to create PCV renders from height maps
Based on
Duguet, Florent & Girardeau-Montaut, Daniel. (2004). Rendu en Portion de Ciel Visible de Gros Nuages de Points 3D.
"""

import cv2
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from numba import jit
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.colors as mcol

SCALE_FACTOR = 100
IMAGE_PATH = 'dtm.tif'  # Replace with path to image


class ImageViewer(QMainWindow):
    def __init__(self, image):
        super().__init__()

        self.image = image

        # Create the main layout
        layout = QVBoxLayout()

        # Create the matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Create the sliders and their labels
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_min.setMinimum(int(self.image.min() * SCALE_FACTOR))
        self.slider_min.setMaximum(int(self.image.max() * SCALE_FACTOR))
        self.slider_min.setValue(int(self.image.min() * SCALE_FACTOR))
        self.slider_min.valueChanged.connect(self.update_image)
        layout.addWidget(QLabel("Min Value"))
        layout.addWidget(self.slider_min)
        self.min_value_label = QLabel()
        layout.addWidget(self.min_value_label)

        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setMinimum(int(self.image.min() * SCALE_FACTOR))
        self.slider_max.setMaximum(int(self.image.max() * SCALE_FACTOR))
        self.slider_max.setValue(int(self.image.max() * SCALE_FACTOR))
        self.slider_max.valueChanged.connect(self.update_image)
        layout.addWidget(QLabel("Max Value"))
        layout.addWidget(self.slider_max)
        self.max_value_label = QLabel()
        layout.addWidget(self.max_value_label)

        # Set the layout to the main window
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Compute gradient
        gradient_y, gradient_x = np.gradient(image)
        self.magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        self.threshold = np.percentile(self.magnitude, 85)
        self.max_gradient_zones = self.magnitude > self.threshold

        self.update_image()

    def update_image(self):
        # Get the min and max values from the sliders and rescale them
        vmin = self.slider_min.value() / SCALE_FACTOR
        vmax = self.slider_max.value() / SCALE_FACTOR

        # Update the labels with the current slider values
        self.min_value_label.setText(f"Current Min Value: {vmin:.4f}")
        self.max_value_label.setText(f"Current Max Value: {vmax:.4f}")

        # Clear the current image
        self.ax.cla()

        # Overlay max gradient zones
        overlay_color = np.where(self.image > self.image.mean(), vmax, vmin)  # 1 for white, 0 for black
        overlay = np.where(self.max_gradient_zones, overlay_color, self.image) # to test

        # Display the image with the new colormap bounds and overlay (replace self.image with overlay)
        self.ax.imshow(self.image, cmap='Greys', vmin=vmin, vmax=vmax)
        self.canvas.draw()


@jit(nopython=True)
def get_tangent_angle(height_diff, distance):
    return height_diff / distance


@jit(nopython=True)
def get_sky_portion_for_direction(image, start_x, start_y, dx, dy, width, height):
    max_tangent = -np.inf
    distance = 0
    x, y = start_x, start_y
    while 0 <= x < width and 0 <= y < height:
        if distance > 0:  # Don't consider the starting pixel
            tangent = get_tangent_angle(image[int(y), int(x)] - image[int(start_y), int(start_x)], distance)
            max_tangent = max(max_tangent, tangent)
        x += dx
        y += dy
        distance += np.sqrt(dx ** 2 + dy ** 2)
    return np.arctan(max_tangent)


@jit(nopython=True)
def compute_sky_visibility(image, num_directions=40):
    height, width = image.shape
    directions = [(np.cos(2 * np.pi * i / num_directions), np.sin(2 * np.pi * i / num_directions)) for i in
                  range(num_directions)]
    sky_visibility = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            total_angle = 0
            for dx, dy in directions:
                total_angle += get_sky_portion_for_direction(image, x, y, dx, dy, width, height)
            sky_visibility[y, x] = total_angle / num_directions
    return sky_visibility


def export_results(visibiliy, dest_path, vmin, vmax, shift_factor, color_factor):
    # Normalize data
    normalized_data = (visibility - vmin) / (vmax - vmin)

    def adjust_color(color, shift_factor, color_factor):
        # Apply the shifting factor
        r, g, b = [x * shift_factor for x in color]

        # Calculate grayscale intensity (average of RGB values)
        intensity = (r + g + b) / 3.0

        # Adjust the coloring factor based on intensity
        adjusted_color_factor = color_factor * (1 - intensity)

        # Apply the adjusted coloring factor to the blue component
        b += adjusted_color_factor * (1 - b)  # Increase blue but ensure it doesn't exceed 1

        return r, g, min(b, 1)  # Ensure blue doesn't exceed 1

    # Original colors
    colors = [(255, 255, 255), (170, 170, 170), (85, 85, 85), (0, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]

    # Adjust colors
    colors_adjusted = [adjust_color(color, shift_factor, color_factor) for color in colors_scaled]

    # Create colormap
    custom_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_adjusted, N=256)

    # Apply a colormap from matplotlib (e.g., 'viridis')
    colored_data = custom_cmap(normalized_data)

    # Convert the RGB data to uint8 [0, 255]
    img = (colored_data[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(dest_path, img_bgr)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Test path
    with rio.open(IMAGE_PATH) as src:
        elevation = src.read(1)
        # Set masked values to np.nan
        elevation[elevation < 0] = np.nan
        height_data = elevation

    visibility = compute_sky_visibility(height_data)

    export_results('dtm.tif', 'out.jpg', -0.1,0.5, 0.8,0.2 )

    app = QApplication(sys.argv)
    viewer = ImageViewer(visibility)
    viewer.show()


    sys.exit(app.exec())
