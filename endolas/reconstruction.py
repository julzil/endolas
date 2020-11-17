import numpy as np
import json


def debug_trace():
  '''Set a tracepoint in the Python debugger that works with Qt'''
  from PyQt5.QtCore import pyqtRemoveInputHook

  from pdb import set_trace
  pyqtRemoveInputHook()
  set_trace()


def get_laser_grid(alpha, n_rows, n_cols, xy_offset):
    """ Retrieves the xy-coordinates on the projection plane from the laser system.

    :param float alpha: The divergence angle between every ray
    :param int n_rows: Total amount of rows
    :param int n_cols: Total amount of columns
    :param ndarray xy_offset: Offset in x- and y-direction of each ray individually
    :return: The homogeneous coordinates from points on the projection plan
    :rtype: ndarray
    """
    laser_grid = np.zeros((3, n_rows * n_cols))

    middle_index = xy_offset.size // 2
    end_index = xy_offset.size
    x_offset = xy_offset[0:middle_index]
    y_offset = xy_offset[middle_index:end_index]

    grid_index = 0
    for row_index in range(n_rows):
        for col_index in range(n_cols):
            laser_grid[0][grid_index] = np.tan((col_index - ((n_cols - 1) / 2)) * alpha[0]) + x_offset[grid_index][0]
            laser_grid[1][grid_index] = np.tan((row_index - ((n_rows - 1) / 2)) * alpha[0]) + y_offset[grid_index][0]
            laser_grid[2][grid_index] = 1.0
            grid_index += 1

    return laser_grid


def extract_laser_grid(laser_grid, n_rows, n_cols, n_rows_present, n_cols_present,
                       row_offset=0, col_offset=0):
    """ From the laser grid extract only the desired rows and columns to reconstruct.

    :param ndarray laser_grid: The
    :param int n_rows: Total number of rows
    :param int n_cols: Total number of columns
    :param int n_rows_present: Actual number of rows
    :param int n_cols_present: Actual number of columns
    :param int row_offset: The offset in columns from left
    :param int col_offset: The offset in rows from top
    :return: The laser grid with only desired points
    :rtype: ndarray
    """
    laser_grid = laser_grid.reshape(3, n_rows, n_cols)
    laser_grid = laser_grid[:, row_offset:(row_offset+n_rows_present), col_offset:(col_offset+n_cols_present)]
    laser_grid = laser_grid.reshape(3, n_rows_present * n_cols_present)

    return laser_grid


def eldot(a, b):
    """ Apply a inner product along the 0th axis.

    :param ndarray a: First operand of the inner product.
    :param ndarray b: Second operand of the inner product.
    :return: An array with the inner products of each entry in axis 1
    :rtype: ndarray
    """
    return np.sum(a * b, axis=0)


def reconstruct(laser_points, camera_points, R, t):
    """ Carry out the solution of the system of equations. This method was transfered from [#boug]_

    .. [#boug] `Bouguet <http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html>`_ Camera Calibration Toolbox for Matlab

    :param ndarray laser_points: Laser points on their projection plane in homogeneous coordinates
    :param camera_points: Undistorted camera points on their projection plane in homogeneous coordinates
    :param ndarray R: The rotation matrix
    :param ndarray t: The translation vector
    :return: Reconstructed xyz-coordinates in the camera coordinate system.
    :rtype: ndarray
    """
    alpha = np.dot(-R, laser_points)
    t = np.repeat(t, alpha.shape[1], axis=1)
    numerator = eldot(alpha, alpha) * eldot(camera_points, t) - eldot(alpha, camera_points) * eldot(alpha, t)
    denominator = eldot(alpha, alpha) * eldot(camera_points, camera_points) - eldot(alpha, camera_points) ** 2

    denominator = np.where(denominator == 0.0, np.nan, denominator)

    z_camera = numerator / denominator
    camera_points = z_camera * camera_points

    return camera_points


def reconstruct_film(calibration, roi_stack, start_row=0, end_row=18, start_col=0, end_col=18, grid_width=18, grid_height=18):
    """ Reconstruct the points.

    This function is close to the implementation in MATLAB to ensure similar results and to allow the reader to
    comprehend both implementations.

    :param list calibration: The calibration as loaded with scipy.io.loadmat from the MATLAB file
    :param list roi_stack: A list with the single points
    :param int start_row: From which row the user wants to include points into the reconstruction.
    :param int end_row: Up to which row the user wants to include points into the reconstruction.
    :param int start_col: From which column the user wants to include points into the reconstruction.
    :param int end_col: Up to which column the user wants to include points into the reconstruction.
    :param grid_width: The total width of the laser grid
    :param grid_height: The total height of the laser grid
    :return: The data is a list containing frames that contain the xyz-coordinates of the reconstructed points.
    :rtype: list
    """
    # preprocess the data
    points = preprocess_data(roi_stack, start_row, end_row, start_col, end_col, grid_width, grid_height)

    # extract parameters from calibration
    camera_calibration = dict()
    camera_calibration['fc'] = calibration['cam'][0][0][0]
    camera_calibration['cc'] = calibration['cam'][0][0][1]
    camera_calibration['alpha_c'] = calibration['cam'][0][0][2]
    camera_calibration['kc'] = calibration['cam'][0][0][3]
    camera_calibration['A'] = calibration['cam'][0][0][4]

    laser_calibration = dict()
    laser_calibration['r'] = calibration['lsr'][0][0][0]
    laser_calibration['t'] = calibration['lsr'][0][0][1]
    laser_calibration['alpha'] = calibration['lsr'][0][0][2]
    laser_calibration['err'] = calibration['lsr'][0][0][3]
    laser_calibration['R'] = calibration['lsr'][0][0][4]
    laser_calibration['Lambda'] = calibration['lsr'][0][0][5]
    laser_calibration['lsrArrayDims'] = calibration['lsr'][0][0][6]

    # prepare the data structure
    n_rows_present = points.shape[0]
    n_cols_present = points.shape[1]
    frames = points.shape[3]

    # get the laser grid
    laser_grid = get_laser_grid(laser_calibration['alpha'], grid_height, grid_width, laser_calibration['Lambda'])

    # select only present rows and cols
    laser_grid = extract_laser_grid(laser_grid, grid_height, grid_width, n_rows_present, n_cols_present,
                                    row_offset=start_row, col_offset=start_col)

    # retrieve relevant parameters
    A = camera_calibration['A']
    R = laser_calibration['R']
    t = laser_calibration['t']
    kc = camera_calibration['kc']

    data = np.zeros((frames, n_rows_present * n_cols_present, 3))
    # loop over frames
    for frame_index in range(frames):
        # get coordinates in detector coordinates
        U = points[:, :, 0, frame_index]
        V = points[:, :, 1, frame_index]

        # row-major representation from linearized representation
        u = U.reshape(n_rows_present * n_cols_present)
        v = V.reshape(n_rows_present * n_cols_present)

        # assign values to homogeneous coordinates, w=0 for not existing values
        u_vec = np.zeros((3, n_rows_present * n_cols_present))
        for grid_index in range(n_rows_present * n_cols_present):
            u_vec[0, grid_index] = u[grid_index] if not np.isnan(u[grid_index]) else 0.0
            u_vec[1, grid_index] = v[grid_index] if not np.isnan(u[grid_index]) else 0.0
            u_vec[2, grid_index] = 1.0 if not np.isnan(u[grid_index]) else 0.0

        # solve system of equations to find projected coordinates
        x_vec = np.linalg.lstsq(A, u_vec)[0]

        # undistort
        x_vec_undistorted = get_x_vec_undistorted(x_vec, kc)

        # triangulate
        data_3d = reconstruct(laser_grid, x_vec_undistorted, R, t)
        data_3d = np.flip(data_3d.reshape(3, 18, 18), 1).reshape(3, 18 * 18)
        #debug_trace()
        data[frame_index, :, :] = data_3d.transpose()

    return data


def get_x_vec_undistorted(x_vec, kc):
    """ Undistort the points on the projection plane.

    :param ndarray x_vec: The xy-coordinates on the projection plane as homogeneous coordinates with the last entry
                          equals to 1.
    :param ndarray kc: The radial distortion parameters
    :return: The undistorted xy-coordinates
    :rtype: ndarray
    """
    # undistortion
    r2 = x_vec[0, :] ** 2 + x_vec[1, :] ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    rd = 1 + kc[0][0] * r2 + kc[1][0] * r4 + kc[4][0] * r6
    x_vec_undistorted = x_vec[0:2, :] * rd

    # refinement
    a1 = 2 * x_vec[0, :] * x_vec[1, :]
    a2 = r2 + 2 * x_vec[0, :] ** 2
    a3 = r2 + 2 * x_vec[1, :] ** 2

    delta_x_1 = kc[2] * a1 + kc[3] * a2
    delta_x_2 = kc[2] * a3 + kc[3] * a1
    delta_x = np.array([delta_x_1, delta_x_2])

    x_vec_undistorted = x_vec_undistorted + delta_x
    x_vec_undistorted = np.append(x_vec_undistorted, np.array([x_vec[2, :]]), 0)

    return x_vec_undistorted


def preprocess_data(roi_stack, start_row, end_row, start_col, end_col, grid_width, grid_height):
    """ Transfer the keypoints into a data structure that is a multi-dimensional array with the shape
        (grid_height, grid_width, 2, frames) as it is also used in the MATLAB implementation of the reconstruction.

        The third axis first contains x-coordinates and the y_coordinates. It is possible that the grid_height and
        grid_width is actually smaller than the one of the laser as not all rows or columns need to be reconstructed.
        The parameters are used to extract only a subset of points.

        .. note: Indexing the grid along its height width starts with 0 at the left top corner and expands to the
                 right along width and to the bottom along the height. Indexing along the height is therefore inversed
                 to the ordering commonly used in the GUI.

    :param int start_row: From which row the user wants to include points into the reconstruction.
    :param list roi_stack: A list with the single points
    :param int end_row: Up to which row the user wants to include points into the reconstruction.
    :param int start_col: From which column the user wants to include points into the reconstruction.
    :param int end_col: Up to which column the user wants to include points into the reconstruction.
    :param grid_width: The total width of the laser grid
    :param grid_height: The total height of the laser grid
    :return: The data structure containing all points to get reconstructed.
    :rtype: ndarray
    """
    #with open('/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/roi_2_mat/KK17_gap0_Cam_16904_Cine2_0_19.rois') as json_file:
    #    data = json.load(json_file)

    np_data = np.zeros((grid_height, grid_width, 2, len(roi_stack)))

    for frame_index, frame in enumerate(roi_stack):
        for roi_index, roi in enumerate(frame):
            i_w = roi_index % grid_width
            i_h = roi_index // grid_height

            np_data[i_h][i_w][0][frame_index] = roi.x()
            np_data[i_h][i_w][1][frame_index] = roi.y()

    np_data = np.flip(np_data, 0)
    np_data = np_data[start_row:(end_row+1), start_col:(end_col+1), :, :]
    np_data = np.where(np_data == -1, np.nan, np_data)

    return np_data