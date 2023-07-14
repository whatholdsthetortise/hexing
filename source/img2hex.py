import math
import numpy as np
from source.HexCovKernel import HexConvKernel


class Point2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2d(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point2d(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Point2d(self.x / scalar, self.y / scalar)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def floor(self):
        return Point2d(math.floor(self.x), math.floor(self.y))


class HexImage:
    ###############################################################################################
    ################################ Coordinate system Explaination ###############################
    ###############################################################################################
    #  *     *     *     *     *     *     *     *     *     *
    #     o     o     o     o     o     o     o     o     o
    #  *     *     *     *     *     *     *     *     *     *
    #     o     o     o     o     o     o     o     o     o
    #  *     *     *     *     *     *     *     *     *     *
    #     o     o     o     o     o     o     o     o     o
    #  *     *     *     *     *     *     *     *     *     *
    #     o     o     o     o     o     o     o     o     o
    #  *     *     *     *     *     *     *     *     *     *
    #     o     o     o     o     o     o     o     o     o
    #  *     *     *     *     *     *     *     *     *     *
    #     o     o     o     o     o     o     o     o     o
    #  *     *     *     *     *     *     *     *     *     *
    # The stars represent the coordinates associated with the outer grid
    # The o's represent the coordinates associated with the inner grid

    def __init__(self, rect_image = None):
        if rect_image is None:
            self.rect_shape = None
            self.outer_grid = None
            self.inner_grid = None
            self.length_scalar = 1.3
            self.min_value = 0.0
            self.max_value = 1.0
        else:
            self.rect_shape = rect_image.shape
            self.outer_grid = None
            self.inner_grid = None
            self.length_scalar = 1.3
            self.min_value = 0.0
            self.max_value = 1.0
            self.convert_to_hex(rect_image)

    def convert_to_hex(self, image):
        # Calculate grid size
        x_width = math.floor(self.rect_shape[0] / self.length_scalar)
        x_step = self.rect_shape[0] / (x_width - 1)
        y_width = math.floor(self.rect_shape[1] / (self.length_scalar * math.sqrt(3)))
        y_step = self.rect_shape[1] / (y_width - 1)
        inner_offset = Point2d(x_step / 2, y_step / 2)

        # Initialize grid
        self.outer_grid = np.zeros((x_width, y_width))
        self.inner_grid = np.zeros((x_width - 1, y_width - 1))

        for x in range(x_width - 2):
            for y in range(y_width - 2):
                # Calculate lower left corner of rectangular grid
                point = Point2d(x * x_step, y * y_step) + inner_offset
                lower_left_corner = point.floor()

                lever = (point.x - lower_left_corner.x)
                v0 = float(image[lower_left_corner.x, lower_left_corner.y]) + lever * (
                        float(image[lower_left_corner.x + 1, lower_left_corner.y]) - float(image[
                    lower_left_corner.x, lower_left_corner.y]))
                v1 = float(image[lower_left_corner.x, lower_left_corner.y + 1]) + lever * (
                        float(image[lower_left_corner.x + 1, lower_left_corner.y + 1]) - float(image[
                    lower_left_corner.x, lower_left_corner.y + 1]))
                lever = (point.y - lower_left_corner.y)

                self.inner_grid[x, y] = v0 + (v1 - v0) * lever

        for x in range(1, x_width - 1):
            for y in range(1, y_width - 1):
                # Calculate lower left corner of rectangular grid
                point = Point2d(x * x_step, y * y_step)
                lower_left_corner = point.floor()

                lever = (point.x - lower_left_corner.x)
                v0 = float(image[lower_left_corner.x, lower_left_corner.y]) + lever * (float(image[lower_left_corner.x + 1, lower_left_corner.y]) - float(image[lower_left_corner.x, lower_left_corner.y]))
                v1 = float(image[lower_left_corner.x, lower_left_corner.y + 1]) + lever * (float(image[lower_left_corner.x + 1, lower_left_corner.y + 1]) - float(image[lower_left_corner.x, lower_left_corner.y + 1]))
                lever = (point.y - lower_left_corner.y)

                self.outer_grid[x, y] = v0 + (v1 - v0) * lever

        # set the corners of the hex outer grid
        self.outer_grid[0, 0] = float(image[0, 0])
        self.outer_grid[-1, -1] = float(image[-1, -1])
        self.outer_grid[0, -1] = float(image[0, -1])
        self.outer_grid[-1, 0] = float(image[-1, 0])

        # fill in between the corners on the edges
        for x in range(1, x_width - 1):
            point = x * x_step
            llx = math.floor(point)
            lever = (point - llx)
            self.outer_grid[x, 0] = float(image[llx, 0]) + lever * (float(image[llx + 1, 0]) - float(image[llx, 0]))
            self.outer_grid[x, -1] = float(image[llx, -1]) + lever * (float(image[llx + 1, -1]) - float(image[llx, -1]))

        for y in range(1, y_width - 1):
            point = y * y_step
            lly = math.floor(point)
            lever = (point - lly)
            self.outer_grid[0, y] = float(image[0, lly]) + lever * (float(image[0, lly + 1]) - float(image[0, lly]))
            self.outer_grid[-1, y] = float(image[-1, lly]) + lever * (float(image[-1, lly + 1]) - float(image[-1, lly]))
        self.min_value = min(np.min(self.outer_grid), np.min(self.inner_grid))
        self.max_value = max(np.max(self.outer_grid), np.max(self.inner_grid))
        self.normalize(self.min_value, self.max_value)

    def min(self):
        return self.min_value

    def max(self):
        return self.max_value

    def normalize(self, new_min, new_max):
        self.outer_grid = (self.outer_grid - self.min()) / (self.max() - self.min()) * (new_max - new_min) + new_min
        self.inner_grid = (self.inner_grid - self.min()) / (self.max() - self.min()) * (new_max - new_min) + new_min

    def convolve(self, kernel):
        new_outer_grid = np.zeros(self.outer_grid.shape)
        new_inner_grid = np.zeros(self.inner_grid.shape)

        # Convolve the outer grid with the kernel
        for i in range(self.outer_grid.shape[0]):
            for j in range(self.outer_grid.shape[1]):

                # handle outer grid
                valid_points = 0
                for k, rel_move in enumerate(relative_moves_outer_grid):
                    if rel_move[0] == 0:
                        if (0 <= (i + rel_move[1]) < self.outer_grid.shape[0]) and (0 <= (j + rel_move[2]) < self.outer_grid.shape[1]):
                            # we have a valid point
                            new_outer_grid[i, j] += self.outer_grid[i + rel_move[1], j + rel_move[2]] * kernel.kernel[k]
                            valid_points += 1
                    else:
                        if (0 <= (i + rel_move[1]) < self.inner_grid.shape[0]) and (0 <= (j + rel_move[2]) < self.inner_grid.shape[1]):
                            # we have a valid point
                            new_outer_grid[i, j] += self.inner_grid[i + rel_move[1], j + rel_move[2]] * kernel.kernel[k]
                            valid_points += 1
                if valid_points > 0:
                    new_outer_grid[i, j] /= valid_points
                # handle inner grid
                if (0 <= i < self.inner_grid.shape[0]) and (0 <= j < self.inner_grid.shape[1]):
                    valid_points = 0
                    for k, rel_move in enumerate(relative_moves_inner_grid):
                        if rel_move[0] == 0:
                            if (0 <= (i + rel_move[1]) < self.outer_grid.shape[0]) and (
                                    0 <= (j + rel_move[2]) < self.outer_grid.shape[1]):
                                # we have a valid point
                                new_inner_grid[i, j] += self.outer_grid[i + rel_move[1], j + rel_move[2]] * kernel.kernel[k]
                                valid_points += 1
                        else:
                            if (0 <= (i + rel_move[1]) < self.inner_grid.shape[0]) and (
                                    0 <= (j + rel_move[2]) < self.inner_grid.shape[1]):
                                # we have a valid point
                                new_inner_grid[i, j] += self.inner_grid[i + rel_move[1], j + rel_move[2]] * kernel.kernel[k]
                                valid_points += 1
                    if valid_points > 0:
                        new_inner_grid[i, j] /= valid_points

        convolved = HexImage()
        convolved.outer_grid = new_outer_grid
        convolved.inner_grid = new_inner_grid
        convolved.length_scalar = self.length_scalar
        convolved.min_value = self.min_value
        convolved.max_value = self.max_value
        convolved.rect_shape = self.rect_shape
        return convolved

    def convert_to_rectangular_image(self):
        rect_image = np.zeros(self.rect_shape)

        for i in range(self.rect_shape[0]):
            for j in range(self.rect_shape[1]):
                self.rect_image[i, j] = self.outer_grid[i, j]


relative_moves_outer_grid = [[0, -1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, -1, 0], [1, 0, -1], [1, -1, -1]]
relative_moves_inner_grid = [[1, -1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]]