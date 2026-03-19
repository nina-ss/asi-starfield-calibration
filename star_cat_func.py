"""Starfield calibration functions."""
import numpy as np
import cv2
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import pandas as pd


def load_data_image(img_path,
                    ori_bright_thresh=50,
                    min_area=3,
                    max_area=200,
                    point_size=5):
    """Process original image."""
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (25, 25), 0)

    stars_enhanced = cv2.subtract(gray, blurred)

    _, thresh = cv2.threshold(stars_enhanced,
                              ori_bright_thresh,
                              255,
                              cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    ori_xy = get_star_centroids(cleaned,
                                min_area=min_area,
                                max_area=max_area)

    ori_xy_p = np.floor(ori_xy).astype(int)

    ori_img_binary = create_img_from_xy(x=ori_xy_p[:, 0],
                                        y=ori_xy_p[:, 1],
                                        img_size=img.shape[0],
                                        point_size=point_size)

    return cleaned, ori_xy, ori_img_binary


def get_star_centroids(binary_img,
                       min_area=3,
                       max_area=200):
    """Get xy coordinates of star centroids."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

    filtered_centroids = [
        centroids[i] for i in range(1, n)
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]

    centroids_array = np.array(filtered_centroids)

    return centroids_array


def simulate_starfield(el,
                       az,
                       ori_img_size,
                       ori_img_center,
                       ori_rotation,
                       t):
    """Compute xy coordinates of simulated stars."""
    S = ori_img_size
    xc, yc = ori_img_center
    theta = ori_rotation

    r = S / 2 * ((1 - t) * (90 - el) / 90 + t * np.cos(np.deg2rad(el)))
    x = xc + r * np.cos(np.deg2rad(90 + az + theta))
    y = yc - r * np.sin(np.deg2rad(90 + az + theta))

    x_pixel = np.floor(x)
    y_pixel = np.floor(y)

    filter_index = np.where((x_pixel < S) & (y_pixel < S))
    x_pixel = x_pixel[filter_index]
    y_pixel = y_pixel[filter_index]
    el = el[filter_index]
    az = az[filter_index]

    sim_img_xy_el_az = np.array((x_pixel, y_pixel, el, az))

    return sim_img_xy_el_az


def create_img_from_xy(x,
                       y,
                       img_size,
                       point_size=5):
    """Create an image from xy coordinates of stars."""
    img = np.zeros((img_size, img_size))

    half = point_size // 2
    offs = np.arange(-half, half + 1)

    dy, dx = np.meshgrid(offs, offs, indexing='ij')
    dy = dy.ravel()
    dx = dx.ravel()

    y_grid = (y[:, None] + dy[None, :])
    x_grid = (x[:, None] + dx[None, :])

    flat_idx = y_grid.ravel() * img_size + x_grid.ravel()
    img.flat[flat_idx] = 255

    return img


def rotate_image(image,
                 angle,
                 center):
    """Rotate an image."""
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)

    return result


def load_star_el_az(star_cat_path,
                    Vmag_threshold,
                    observation_time,
                    camera_location):
    """Get el/az coordinates of theoretically visible stars."""
    stars_df = pd.read_csv(star_cat_path, header=0)
    bright_stars_df = stars_df[stars_df["Vmag"] < Vmag_threshold].copy()

    time = Time(observation_time, scale='utc')
    user_loc = EarthLocation(lat=camera_location[0] * u.deg,
                             lon=camera_location[1] * u.deg,
                             height=camera_location[2] * u.m)

    az, el = np.zeros((2, len(bright_stars_df['Name'])))

    coords = SkyCoord(ra=bright_stars_df['Ra'].values * u.deg,
                      dec=bright_stars_df['Dec'].values * u.deg,
                      frame='icrs')

    altaz_frame = AltAz(obstime=time,
                        location=user_loc)

    altaz = coords.transform_to(altaz_frame)

    az = altaz.az.deg
    el = altaz.alt.deg

    filter_index = np.where(el >= 0)
    el = el[filter_index]
    az = az[filter_index]

    return el, az
