import pathlib
from typing import NamedTuple

import numpy as np
from skimage.io import imread, imsave


class CommonOptions(NamedTuple):
    input_image: pathlib.Path
    output_image: pathlib.Path

    def load_image(self):
        return imread(self.input_image)

    def save_image(self, img: np.ndarray):
        args = {}
        if self.output_image.suffix != '.png':
            args['quality'] = 100
        imsave(self.output_image, img)
