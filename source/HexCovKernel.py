import numpy as np


class HexConvKernel:
    def __init__(self):
        # ''' Currently only radius 1 is supported '''
        #     __
        #  __/  \__
        # /  \__/  \
        # \__/  \__/
        # /  \__/  \
        # \__/  \__/
        #    \__/      7 hexagons
        # '''
        self.kernel = np.random.random((7,))
