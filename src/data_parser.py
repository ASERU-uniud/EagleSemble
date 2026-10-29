import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from numpy.linalg import norm



def extract_features(road):
    road = np.array(road)
    diff = road[1:] - road[:-1] # also: np.diff(road, axis=0)
    seg_lengths = norm(diff, axis=1)

    angle_diff = [0] + np.arccos(  (np.vecdot(diff[1:], diff[:-1])   /  ( seg_lengths[:-1]*seg_lengths[1:]) )  ).tolist()

    return list(zip(seg_lengths.tolist(), angle_diff))

def add_padding(road, padding_length):
    return road + [road[-1]]*padding_length

def add_straight_road(road, padding_length):
    last_point = np.array(road[-1])
    second_last = np.array(road[-2])
    straight_vector = last_point-second_last
    for j in range(padding_length):
        road.append((last_point+(j+1)*straight_vector).tolist())
    return road

def set_arrays_size(roads, length=197, extender=add_straight_road):
    for road in roads:
        if len(road) >= length:
            yield road[:length]
        else:
            yield extender(road, length-len(road))

def adjust_array_size(road, length=197, extender=add_straight_road):
    l = len(road)
    if l == length:
        return road
    elif l > length:
        return road[:length]
    else:
        return extender(road, length - len(road))

#---------------------------------------------------------------------------------------------------

#TODO you know it.

def adjust_road_array(road_array, target_size=197):
    if road_array.shape[0] == target_size:
        return road_array
    else:
        current_indices = np.linspace(0, road_array.shape[0] - 1, road_array.shape[0])
        target_indices = np.linspace(0, road_array.shape[0] - 1, target_size)

        interpolator_x = interp1d(current_indices, road_array[:, 0], kind='cubic')
        interpolator_y = interp1d(current_indices, road_array[:, 1], kind='cubic')

        interpolated_x = interpolator_x(target_indices)
        interpolated_y = interpolator_y(target_indices)

        return np.column_stack((interpolated_x, interpolated_y))

# TODO make trajectory array start always from the position relative to reference line ????
def adjust_traj_array(traj_array, target_size=300):
    if traj_array.shape[0] == target_size:
        return traj_array

    elif traj_array.shape[0] > target_size:
        indices = np.linspace(0, traj_array.shape[0] - 1, target_size, dtype=int)
        return traj_array[indices]

    else:
        return np.append(traj_array, [traj_array[-1]]*(target_size-traj_array.shape[0]), axis=0)

def calculate_features(points):
    dx = points[1:, 0] - points[:-1, 0]
    dy = points[1:, 1] - points[:-1, 1]

    raw_angles = np.arctan2(dy, dx)

    segment_angles = np.zeros_like(raw_angles)
    segment_angles[1:] = np.diff(raw_angles)

    _differences = points[1:] - points[:-1]
    segment_lengths = norm(_differences, axis=1)

    return dx, dy, segment_lengths, segment_angles

def extract_road_features(road_points):
    dx, dy, segment_lengths, segment_angles = calculate_features(road_points)
    curvature = segment_angles / segment_lengths
    return np.column_stack((segment_lengths, segment_angles, curvature))

def extract_traj_features(traj_points, road_points):
    traj_points = np.insert(traj_points, 0, road_points[0], axis=0)
    dx, dy, segment_lengths, segment_angles = calculate_features(traj_points)
    return np.column_stack((segment_lengths, segment_angles))


def frenet_frame(traj_points, road_points):
    distance_matrix = cdist(traj_points, road_points, 'euclidean')

    cumulative_dist = np.zeros_like(road_points[:,0])
    cumulative_dist[1:] = np.cumsum(norm(np.diff(road_points, axis=0), axis=1))

    d_vector = np.min(distance_matrix, axis=1)
    s_vector = cumulative_dist[np.argmin(distance_matrix, axis=1)]

    s_vector[1:] = s_vector[1:] - s_vector[:-1]
    d_vector[1:] = d_vector[1:] - d_vector[:-1]
    # dx, dy, segment_lengths, segment_angles = calculate_features(np.insert(traj_points, 0, road_points[0], axis=0))

    return np.column_stack((s_vector, d_vector, ))

def get_orthogonals_v(road_points):
    orthogonals = np.zeros_like(road_points)
    
    # approximate tangent with secant
    orthogonals[0] = road_points[1] - road_points[0]
    orthogonals[-1] = road_points[-1] - road_points[-2]
    orthogonals[1:-1] = road_points[2:] - road_points[:-2]
    
    # rotate -90 degrees
    orthogonals[:, 0], orthogonals[:, 1] = orthogonals[:, 1], -orthogonals[:, 0]

    # normalize
    orthogonals /= norm(orthogonals, axis=1).reshape(-1, 1)

    return orthogonals

def frenet_to_traj(frenet_points, road_points):
    cumulative_length = np.cumsum(norm(road_points[1:]-road_points[:-1], axis=1))
    orthogonals = get_orthogonals_v(road_points)
    j = 0
    traj_points = np.zeros_like(frenet_points)
    for i in range(frenet_points.shape[0]):
        while j < cumulative_length.shape[0] and frenet_points[i, 0] >= cumulative_length[j]:
            j += 1
        traj_points[i] = road_points[j] + (orthogonals[j]*frenet_points[i, 1])
    return traj_points


def geom_features_to_coords(couples):
    segment_lengths, angle_displ = couples[:, 0], couples[:, 1]
    angles = np.cumsum(angle_displ)
    diff_points = np.column_stack((np.cos(angles), np.sin(angles))) * segment_lengths.reshape(-1, 1)
    points = np.cumsum(diff_points, axis=0)
    return points