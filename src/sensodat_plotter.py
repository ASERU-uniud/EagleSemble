import matplotlib.pyplot as plt
import shapely

def plot_trajectory(trajectory, traj_color='#FFFF00', plot_car=False):
    trajectory_x, trajectory_y = list(zip(*trajectory))
    plt.plot(trajectory_x, trajectory_y, traj_color)

    if plot_car:
        xn, yn = trajectory_x[-1], trajectory_y[-1]
        car = shapely.box(xn-1, yn-1, xn+1, yn+1)
        plt.plot(*car.exterior.xy, traj_color)

#TODO re-design for performance DONE???
def offset(line: shapely.LineString | list[tuple[float, ...]],
           distance:float=0, width:float=0, chunk_size=30):

    coords = list(line.coords) if isinstance(line, shapely.LineString) else line

    if width > 0:
        r = offset(coords, distance=distance + width / 2, width=0, chunk_size=chunk_size)
        l = offset(coords, distance=distance - width / 2, width=0, chunk_size=chunk_size)
        coords = r + l[::-1] + r[0:1]
        if isinstance(line, shapely.LineString):
            coords = shapely.Polygon(coords)
        return coords

    if distance != 0:
        chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
        offsets = [shapely.LineString(coordinates=chunk).offset_curve(distance=distance) for chunk in chunks]
        coords = sum([list(o.coords) for o in offsets], [])

    if isinstance(line, shapely.LineString):
        coords = shapely.LineString(coords)

    return coords


def plot_road(road, lane_width=5.0, line_width=0.2, road_color='#000000',
                    line_color='#FFFFFF', ):
    reference_line = shapely.LineString(coordinates=road)
    right_line = offset(reference_line, distance=lane_width, width=line_width)
    left_line = offset(reference_line, distance=-lane_width, width=line_width)
    center_line = offset(reference_line, width=line_width)
    road_polygon = offset(reference_line, width=2*lane_width+4*line_width)

    plt.fill(*road_polygon.exterior.xy, road_color)
    plt.fill(*left_line.exterior.xy, line_color)
    plt.fill(*center_line.exterior.xy, line_color)
    plt.fill(*right_line.exterior.xy, line_color)


def plot_simulation(road, trajectory, outcome, lane_width=5.0, line_width=0.2, road_color='#000000',
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