import numpy as np
import cv2
import matplotlib.pyplot as plt
from source import img2hex
from source.HexCovKernel import HexConvKernel


def main():
    image = cv2.imread('images/Lenna.png')
    # image = cv2.imread('images/dots.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()
    hex = img2hex.HexImage(gray)
    kernel = HexConvKernel()

    # Make it a horizontal line filter
    kernel.kernel = np.array([0, 0, 0, -1, -1, 1, 1])
    x_edge = hex.convolve(kernel)

    # Make it a + 60 degree line filter
    kernel.kernel = np.array([-1, 0, 1, 0, -1, 1, 0])
    p60 = hex.convolve(kernel)

    # Make it a - 60 degree line filter
    kernel.kernel = np.array([-1, 0, 1, 1, 0, 0, -1])
    n60 = hex.convolve(kernel)

    plt.imshow(np.abs(x_edge.inner_grid) + np.abs(p60.inner_grid) + np.abs(n60.inner_grid))
    plt.show()


if __name__ == '__main__':
    main()
