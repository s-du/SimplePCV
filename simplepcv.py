import matplotlib.pyplot as plt
from numba import jit
import matplotlib.colors as mcol
import cv2
from joblib import Parallel, delayed, cpu_count
import numpy as np


# ----------------PCV-------------------------
def plot_histogram(data, bins=256):
    plt.hist(data.ravel(), bins=bins, color='blue', alpha=0.7)
    plt.title('Histogram of Visibility Values')
    plt.xlabel('Visibility Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


@jit(nopython=True)
def get_tangent_angle(height_diff, distance):
    return height_diff / distance


@jit(nopython=True)
def get_sky_portion_for_direction(image, start_x, start_y, dx, dy, width, height):
    max_tangent = -np.inf
    distance = 0
    x, y = start_x, start_y
    step_size = 3 # or another value that makes sense for your data
    while 0 <= x < width and 0 <= y < height:
        if distance > 0:
            tangent = get_tangent_angle(image[int(y), int(x)] - image[int(start_y), int(start_x)], distance)
            max_tangent = max(max_tangent, tangent)
        x += dx * step_size
        y += dy * step_size
        distance += np.sqrt((dx * step_size)**2 + (dy * step_size)**2)
    return np.arctan(max_tangent)

@jit(nopython=True)
def compute_sky_visibility_for_chunk(image, start_row, end_row, num_directions=40):
    height, width = image.shape
    directions = [(np.cos(2 * np.pi * i / num_directions), np.sin(2 * np.pi * i / num_directions)) for i in range(num_directions)]
    sky_visibility = np.zeros((end_row - start_row, width))
    for y in range(start_row, end_row):
        for x in range(width):
            total_angle = 0
            for dx, dy in directions:
                total_angle += get_sky_portion_for_direction(image, x, y, dx, dy, width, height)
            sky_visibility[y - start_row, x] = total_angle / num_directions
    return sky_visibility


def compute_sky_visibility(image, num_directions=60, n_jobs=-1):
    height, width = image.shape

    # If n_jobs is set to -1, use all available cores
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Ensure n_jobs does not exceed image height
    n_jobs = min(n_jobs, height)

    chunk_size = height // n_jobs
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_sky_visibility_for_chunk)(image, i, min(i + chunk_size, height), num_directions) for i in
        range(0, height, chunk_size))

    return np.vstack(results)


def compute_optimal_vmin_vmax(visibility, bins=256, threshold_low=0.05, threshold_high=0.001):
    hist, bin_edges = np.histogram(visibility, bins=bins)

    # Normalize histogram
    hist = hist / hist.sum()

    # Find the main peak: bin with the maximum count
    main_peak_index = np.argmax(hist)
    print(main_peak_index)
    print(bin_edges[main_peak_index])

    # If the main peak's value is below the threshold, find the next peak
    while bin_edges[main_peak_index] < -1:
        hist[main_peak_index] = 0  # Set the current peak's value to 0
        main_peak_index = np.argmax(hist)  # Find the next peak

    # Starting from the main peak, find vmin by moving left until the count drops below the threshold
    for i in range(main_peak_index, 0, -1):
        if hist[i] < threshold_low:
            vmin = bin_edges[i]
            break
    else:
        vmin = bin_edges[0]

    # Starting from the main peak, find vmax by moving right until the count drops below the threshold
    for i in range(main_peak_index, len(hist) - 1):
        if hist[i] < threshold_high:
            vmax = bin_edges[i + 1]
            break
    else:
        vmax = bin_edges[-1]

    return vmin, vmax
def export_results(visibility, vmin, vmax, color_choice, color_factor, standardize=False, optimize_vrange=True):
    def adjust_color(color, color_factor, h_color):
        r, g, b = color

        # Calculate grayscale intensity (average of RGB values)
        intensity = (r + g + b) / 3.0


        # Adjust the coloring factor based on intensity
        adjusted_color_factor = color_factor * (1 - intensity)

        # Apply the adjusted coloring factor to the blue component
        if h_color == 0:
            r += adjusted_color_factor * (1 - r)  # Increase blue but ensure it doesn't exceed 1
            return min(r, 1), g, b  # Ensure doesn't exceed 1
        elif h_color == 1:
            g += adjusted_color_factor * (1 - g)  # Increase blue but ensure it doesn't exceed 1
            return r, min(g,1), b  # Ensure doesn't exceed 1
        elif h_color == 2:
            b += adjusted_color_factor * (1 - b)  # Increase blue but ensure it doesn't exceed 1
            return r, g, min(b,1)  # Ensure doesn't exceed 1


    if standardize:
        vmin = np.percentile(visibility, 1)  # 1st percentile
        vmax = np.percentile(visibility, 99)
    if optimize_vrange:
        vmin, vmax = compute_optimal_vmin_vmax(visibility)

    # Normalize data
    normalized_data = (visibility - vmin) / (vmax - vmin)

    grayscale_img = (normalized_data * 255).astype(np.uint8)

    # Apply histogram equalization
    #equalized_img = cv2.equalizeHist(grayscale_img)

    # Convert back to float range [0, 1] for coloring
    #equalized_data = equalized_img.astype(np.float32) / 255.0


    # Original colors
    colors = [(255, 255, 255),(50, 50, 50)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]

    # Adjust colors
    colors_adjusted = [adjust_color(color, color_factor, color_choice) for color in colors_scaled]

    # Create colormap
    custom_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_adjusted, N=256)

    # Apply a colormap from matplotlib (e.g., 'viridis')
    colored_data = custom_cmap(normalized_data)

    # Convert the RGB data to uint8 [0, 255]
    img = (colored_data[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img_bgr

# ----------------Hillshade-------------------------

# Constants
PI = np.pi


# Step 1: Compute the Illumination Angle
def compute_illumination_angle(altitude):
    # Convert altitude to zenith angle
    zenith_deg = 90.0 - altitude
    # Convert zenith angle to radians
    zenith_rad = zenith_deg * PI / 180.0
    return zenith_rad


# Step 2: Compute the Illumination Direction
def compute_illumination_direction(azimuth):
    # Convert azimuth angle from geographic to mathematical unit
    azimuth_math = 360.0 - azimuth + 90.0
    if azimuth_math >= 360.0:
        azimuth_math = azimuth_math - 360.0
    # Convert the mathematical azimuth angle to radians
    azimuth_rad = azimuth_math * PI / 180.0
    return azimuth_rad


# Step 3: Compute Slope and Aspect for a cell
def compute_slope_aspect(cell_values, cellsize, z_factor):
    a, b, c, d, e, f, g, h, i = cell_values

    # Compute rate of change in x direction
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize)
    # Compute rate of change in y direction
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize)

    # Compute slope in radians
    slope_rad = np.arctan(z_factor * np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    # Determine aspect in radians
    if dz_dx != 0:
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        if aspect_rad < 0:
            aspect_rad = 2 * PI + aspect_rad
    else:
        if dz_dy > 0:
            aspect_rad = PI / 2
        elif dz_dy < 0:
            aspect_rad = 2 * PI - PI / 2
        else:
            aspect_rad = 0  # Set aspect to zero when terrain is flat

    return slope_rad, aspect_rad


# Step 4: Compute Hillshade for a cell
def compute_hillshade(zenith_rad, azimuth_rad, slope_rad, aspect_rad):
    hillshade = 255.0 * ((np.cos(zenith_rad) * np.cos(slope_rad)) +
                         (np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)))
    return max(hillshade, 0)  # If hillshade value is less than 0, return 0


def compute_hillshade_for_grid(elevation_grid, cellsize=1, z_factor=1, altitude=45, azimuth=315):
    # Get the dimensions of the elevation grid
    rows, cols = elevation_grid.shape

    # Initialize the hillshade matrix with zeros
    hillshade_matrix = np.zeros_like(elevation_grid)

    # Compute zenith and azimuth in radians
    zenith_rad = compute_illumination_angle(altitude)
    azimuth_rad = compute_illumination_direction(azimuth)

    # Helper function to get 3x3 window around a cell
    def get_window(r, c):
        # Handle boundary cases by mirroring
        r_start = max(r - 1, 0)
        r_end = min(r + 2, rows)
        c_start = max(c - 1, 0)
        c_end = min(c + 2, cols)

        return elevation_grid[r_start:r_end, c_start:c_end].flatten()

    # Iterate over each cell in the elevation grid
    for r in range(rows):
        for c in range(cols):
            # Get the 3x3 window around the current cell
            cell_values = get_window(r, c)

            # If we don't have a full 3x3 window (i.e., for boundary cells), continue to next cell
            if len(cell_values) != 9:
                continue

            # Compute slope and aspect for the cell
            slope_rad, aspect_rad = compute_slope_aspect(cell_values, cellsize, z_factor)

            # Compute the hillshade value for the cell
            hillshade_value = compute_hillshade(zenith_rad, azimuth_rad, slope_rad, aspect_rad)

            # Store the hillshade value in the output matrix
            hillshade_matrix[r, c] = hillshade_value

    return hillshade_matrix

# Test the individual functions
altitude = 45  # Default altitude
azimuth = 315  # Default azimuth

zenith_rad = compute_illumination_angle(altitude)
azimuth_rad = compute_illumination_direction(azimuth)