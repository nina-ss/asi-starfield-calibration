import numpy as np
from typing import Tuple
import cv2


def get_star_centroids(binary_img, min_area=3, max_area=200):
    """
    Extract star centroids from a preprocessed binary image.

    Args:
        binary_img (np.ndarray): Binary (0/255) NumPy image.
        min_area (int): Minimum blob area to keep.
        max_area (int): Maximum blob area to keep.

    Returns:
        np.ndarray: Array of centroids (x, y) for each detected star.
    """
    # Ensure binary (0/1)
    binary = (binary_img > 0).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Filter out small/large blobs
    filtered_centroids = [
        centroids[i] for i in range(1, num_labels)
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]

    centroids_array = np.array(filtered_centroids)

    return centroids_array


def stars_to_polar_binary(elev,
                          azim,
                          img_size=4000,
                          elev_range=(0.0, 90.0),      # degrees
                          azim_range=(0.0, 360.0),     # degrees
                          point_size=1,                # side-length of the brush (odd ≥1)
                          rotation_deg=0):
    """
    Convert elevation / azimuth data to a binary polar image **and**
    return the pixel-center coordinates together with the (unique) angles
    that produced them.

    Parameters
    ----------
    elev, azim : array-like
        1-D arrays of equal length.  They may be in degrees or radians;
        the routine auto-detects the unit.
    img_size : int, default 4000
        Width and height of the square output image (pixels).
    elev_range, azim_range : tuple(float, float), default (0,90) & (0,360)
        Minimum / maximum elevation and azimuth that map to the polar plot.
    point_size : int, default 1
        Side length (pixels) of the square “brush” used to draw a star.
        Must be an odd positive integer; "1" draws a single pixel.
    rotation_deg : float, default 0
        Angular offset applied **after** the azimuth is normalised.
        Positive values rotate the whole pattern counter-clockwise
        (i.e. azimuth=0° moves toward the left).  
        Use "rotation_deg = -90" to make azimuth 0 point straight up.

    Returns
    -------
    img : ndarray, shape (img_size, img_size), dtype=np.uint8
        Binary image: 0 = background, 255 = star pixel (or brush block).
    stars_xy_el_az : ndarray, shape (M, 4), dtype=int/float
        Each row contains "[x_center, y_center, elevation, azimuth]" for one
        **unique** pixel centre that was actually plotted.
        "x_center" and "y_center" are integer pixel indices (0-based, with
        (0,0) at the top-left).  Elevation and azimuth are returned in the
        same units as the input (degrees after the internal conversion).
        If several catalogue entries map to the same pixel centre only the
        first one is kept (duplicates are discarded).
    """

    # --------------------------------------------------------------
    # 1. Normalise to degrees (avoid rad/deg mix-up)
    # --------------------------------------------------------------
    elev = np.asarray(elev, dtype=np.float64)
    azim = np.asarray(azim, dtype=np.float64)

    # Heuristic: any value > 2π → treat the whole array as degrees.
    if not (np.any(elev > 2*np.pi) or np.any(azim > 2*np.pi)):
        elev = np.degrees(elev)
        azim = np.degrees(azim)

    # --------------------------------------------------------------
    # 2. Clip to the user-defined field of view
    # --------------------------------------------------------------
    e_min, e_max = elev_range
    a_min, a_max = azim_range

    mask = (elev >= e_min) & (elev <= e_max) & \
           (azim >= a_min) & (azim <= a_max)

    elev = elev[mask]
    azim = azim[mask]

    # --------------------------------------------------------------
    # 3. Normalise azimuth, apply optional rotation
    # --------------------------------------------------------------
    az_norm = (azim - a_min) / (a_max - a_min)          # 0 … 1
    el_norm = (elev - e_min) / (e_max - e_min)          # 0 … 1

    # rotation (degrees → radians) **after** normalisation
    theta = az_norm * 2.0 * np.pi + np.radians(rotation_deg)

    # --------------------------------------------------------------
    # 4. Radius (outer edge = horizon, centre = max elevation)
    # --------------------------------------------------------------
    r_max = img_size / 2.0
    r = (1.0 - el_norm) * r_max                         # highest elevation → outer rim

    # --------------------------------------------------------------
    # 5. Cartesian pixel coordinates of the *centre* of each star
    # --------------------------------------------------------------
    cx = cy = img_size / 2.0
    x_f = cx + r * np.cos(theta)
    y_f = cy - r * np.sin(theta)                        # y grows downwards

    x_center = np.floor(x_f).astype(np.int32)
    y_center = np.floor(y_f).astype(np.int32)

    np.clip(x_center, 0, img_size - 1, out=x_center)
    np.clip(y_center, 0, img_size - 1, out=y_center)

    # --------------------------------------------------------------
    # 6. Remove duplicate pixel centres (keep the first occurrence)
    # --------------------------------------------------------------
    stacked = np.stack((x_center, y_center), axis=1)    # shape (N,2)
    # `return_index` gives the index of the first occurrence of each unique row
    uniq_rows, uniq_idx = np.unique(stacked, axis=0, return_index=True)

    # Unique centre coordinates
    x_unique = uniq_rows[:, 0]
    y_unique = uniq_rows[:, 1]

    # Corresponding elevation / azimuth (taken from the first entry that maps
    # to the pixel).  These are still in degrees because we converted earlier.
    elev_unique = elev[uniq_idx]
    azim_unique = azim[uniq_idx]

    # --------------------------------------------------------------
    # 7. Rasterise – single pixel or square brush
    # --------------------------------------------------------------
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Enforce odd brush size for symmetry
    if point_size < 1:
        raise ValueError("point_size must be >= 1")
    if point_size % 2 == 0:
        point_size += 1

    half = point_size // 2
    offs = np.arange(-half, half + 1, dtype=np.int32)           # (point_size,)

    dy, dx = np.meshgrid(offs, offs, indexing='ij')
    dy = dy.ravel()
    dx = dx.ravel()

    # Broadcast offsets onto *all* unique centres (more memory-efficient than using
    # the full original list, because duplicates have already been removed)
    y_grid = (y_unique[:, None] + dy[None, :])
    x_grid = (x_unique[:, None] + dx[None, :])

    np.clip(y_grid, 0, img_size - 1, out=y_grid)
    np.clip(x_grid, 0, img_size - 1, out=x_grid)

    flat_idx = y_grid.ravel() * img_size + x_grid.ravel()
    img.flat[flat_idx] = 255

    # --------------------------------------------------------------
    # 8. Pack the coordinate/angle table for the caller
    # --------------------------------------------------------------
    stars_xy_el_az = np.column_stack((
        x_unique.astype(np.int32),
        y_unique.astype(np.int32),
        elev_unique,      # degrees
        azim_unique       # degrees
    ))

    return img, stars_xy_el_az


def load_data_image(img_path, rotation, center, threshold=50):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (25, 25), 0)

    stars_enhanced = cv2.subtract(gray, blurred)

    _, thresh = cv2.threshold(stars_enhanced, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    rotated = rotate_image(cleaned, rotation, center)

    #include centroid detection

    return rotated


def rotate_image(image, angle, center):
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def redraw_with_user_xy(
    star_xy_elaz: np.ndarray,           # shape (N,4) → [x, y, elevation, azimuth]
    new_xy:      np.ndarray,           # shape (N,2) → [x_new, y_new] (int-like)
    img_size:    int = 4000,
    point_size:  int = 1               # side length of the brush (odd ≥1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a new binary image whose *on* pixels follow the user-provided
    coordinates while preserving the original elevation and azimuth values.

    Parameters
    ----------
    star_xy_elaz : ndarray, shape (N,4)
        Columns must be "[x_old, y_old, elevation, azimuth]".  The
        elevation/azimuth columns are kept unchanged.
    new_xy : ndarray, shape (N,2)
        Desired pixel centre for each star, "[x_new, y_new]".  Values may be
        outside the image; they will be clipped to "[0, img_size-1]".
    img_size : int, default 4000
        Width and height of the square output image.
    point_size : int, default 1
        Size of the square brush used to render a star (must be odd; an even
        value is automatically increased by one).

    Returns
    -------
    img : ndarray, shape (img_size, img_size), dtype=np.uint8
        Binary mask (0=background, 255=star/brush pixel).
    star_xy_elaz_modified : ndarray, shape (M,4)
        "[x_new, y_new, elevation, azimuth]" for each *unique* centre that was
        actually rasterised (duplicates are discarded).

    Raises
    ------
    ValueError
        If "star_xy_elaz" and "new_xy" do not contain the same number of rows.
    """
    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------
    if star_xy_elaz.shape[0] != new_xy.shape[0]:
        raise ValueError(
            f"Length mismatch: star_xy_elaz has {star_xy_elaz.shape[0]} rows, "
            f"new_xy has {new_xy.shape[0]} rows."
        )
    if star_xy_elaz.shape[1] != 4:
        raise ValueError("star_xy_elaz must have 4 columns: x, y, elevation, azimuth.")
    if new_xy.shape[1] != 2:
        raise ValueError("new_xy must have exactly 2 columns: x_new, y_new.")

    # --------------------------------------------------------------
    # Prepare coordinates – clip to the image canvas
    # --------------------------------------------------------------
    x_new = np.asarray(new_xy[:, 0], dtype=np.int32)
    y_new = np.asarray(new_xy[:, 1], dtype=np.int32)

    np.clip(x_new, 0, img_size - 1, out=x_new)
    np.clip(y_new, 0, img_size - 1, out=y_new)

    # --------------------------------------------------------------
    # Remove duplicate pixel centres (keep the first occurrence)
    # --------------------------------------------------------------
    centres = np.stack((x_new, y_new), axis=1)           # shape (N,2)
    uniq_centre, uniq_idx = np.unique(centres, axis=0, return_index=True)

    x_unique = uniq_centre[:, 0]
    y_unique = uniq_centre[:, 1]

    # Preserve the corresponding elevation/azimuth values
    elev_unique = star_xy_elaz[uniq_idx, 2]
    azim_unique = star_xy_elaz[uniq_idx, 3]

    # --------------------------------------------------------------
    # Brush handling
    # --------------------------------------------------------------
    if point_size < 1:
        raise ValueError("point_size must be >= 1")
    if point_size % 2 == 0:        # enforce odd size for symmetry
        point_size += 1

    half = point_size // 2
    offs = np.arange(-half, half + 1, dtype=np.int32)   # (point_size,)

    dy, dx = np.meshgrid(offs, offs, indexing='ij')
    dy = dy.ravel()
    dx = dx.ravel()

    # Broadcast offsets onto every unique centre
    y_grid = (y_unique[:, None] + dy[None, :])
    x_grid = (x_unique[:, None] + dx[None, :])

    np.clip(y_grid, 0, img_size - 1, out=y_grid)
    np.clip(x_grid, 0, img_size - 1, out=x_grid)

    # --------------------------------------------------------------
    # Rasterise
    # --------------------------------------------------------------
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    flat_idx = y_grid.ravel() * img_size + x_grid.ravel()
    img.flat[flat_idx] = 255

    # --------------------------------------------------------------
    # Assemble the modified star table
    # --------------------------------------------------------------
    star_xy_elaz_modified = np.column_stack((
        x_unique.astype(np.int32),
        y_unique.astype(np.int32),
        elev_unique,
        azim_unique
    ))

    return img, star_xy_elaz_modified
