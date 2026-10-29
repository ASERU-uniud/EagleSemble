import matplotlib.pyplot as plt
from math import sqrt

#TODO Use Numpy
#TODO better approximation of tangents (try scipy)

def sum_vectors(v1, v2):
    (x1, y1), (x2, y2) = v1, v2
    return x1 + x2, y1 + y2

def get_orthogonal(vector):
    x, y = vector
    return y, -x

def normalize(vector, factor=1):
    x, y = vector
    norm = sqrt(x * x + y * y)
    return factor * x / norm, factor * y / norm


def get_tangents(line):
    previous_point = next(line)
    current_point = next(line)
    yield (current_point[0] - previous_point[0]), (current_point[1] - previous_point[1])
    for next_point in line:
        yield (next_point[0] - previous_point[0]), (next_point[1] - previous_point[1])
        previous_point = current_point
        current_point = next_point
    yield (current_point[0] - previous_point[0]), (current_point[1] - previous_point[1])


def get_boundary_lines(line, distances):
    line = list(line)
    orthogonals = [get_orthogonal(t) for t in get_tangents(iter(line))]
    for distance in distances:
        yield [sum_vectors(p, normalize(b, distance)) for p, b in zip(line, orthogonals)]

def plot_trajectory(trajectory, traj_color='#FFFF00'):
    trajectory_x, trajectory_y = list(zip(*trajectory))
    plt.plot(trajectory_x, trajectory_y, traj_color)


def plot_road(road, lane_width=3.0, line_width=0.2, road_color='#000000',
                    line_color='#FFFFFF', ):
    distances = [lane_width + line_width * 2,
                 lane_width + line_width / 2,
                 lane_width - line_width / 2,
                 line_width / 2,
                 -line_width / 2,
                 -lane_width + line_width / 2,
                 -lane_width - line_width / 2,
                 -lane_width - line_width * 2]

    (road_left,
     line_left_l,
     line_left_r,
     line_central_l,
     line_central_r,
     line_right_l,
     line_right_r,
     road_right) = list(get_boundary_lines(road, distances))

    road_x, road_y = list(zip(*(road_left+road_right[::-1])))
    line_left_x, line_left_y = list(zip(*(line_left_l+line_left_r[::-1])))
    line_central_x, line_central_y = list(zip(*(line_central_l+line_central_r[::-1])))
    line_right_x, line_right_y = list(zip(*(line_right_l+line_right_r[::-1])))

    plt.fill(road_x, road_y, road_color)
    plt.fill(line_left_x, line_left_y, line_color)
    plt.fill(line_central_x, line_central_y, line_color)
    plt.fill(line_right_x, line_right_y, line_color)


def plot_simulation(road, trajectory, outcome, lane_width=3.0, line_width=0.2, road_color='#000000',
                    line_color='#FFFFFF', ):
    clear()
    plot_road(road, lane_width, line_width, road_color, line_color)
    traj_color = '#FF0000' if outcome == 0 else '#00FF00'
    plot_trajectory(trajectory, traj_color)

def show(block=None):
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('equal')
    plt.show(block=block)

def clear():
    plt.clf()