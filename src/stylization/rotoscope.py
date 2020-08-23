from typing import NamedTuple

import click
import numpy as np
from ils_smoothing import ILSSmoothingFilter

from ._effect import Effect
from ._types import CommonOptions

class RotoscopeEffect(Effect):
    '''Create a rotoscoping effect similar to "A Scanner Darkly".'''
    class Config(NamedTuple):
        '''Configuration options for the rotoscope effect.'''
        smoothing: float = 5
        edge_preservation: float = 1
        iterations: int = 8

    def __init__(self, config: Config = Config()):
        '''Initialize a new rotoscope effect object.

        See the :class:`RotoscopeEffect.Config` structure for the various
        options.

        Parameters
        ----------
        config : Config, optional
            runtime configuration
        '''
        self._ils_filter = ILSSmoothingFilter(config.smoothing,
                                              config.edge_preservation,
                                              config.iterations)

    def apply(self, img: np.ndarray) -> np.ndarray:
        '''Apply the rotoscope effect filter to an image.

        Parameters
        ----------
        img : np.ndarray
            input RGB image

        Returns
        -------
        np.ndarray
            processed image
        '''
        return img

    def _validate(self, img: np.ndarray):
        '''Determine if the image can be processed.'''
        if img.dtype != np.uint8 and img.dtype != np.float:
            raise ValueError('Image must be either 8-bit or floating point.')

        if img.ndim != 3 and img.shape[2] != 3:
            raise ValueError('Image must be 3-channel RGB.')


@click.command('rotoscope')
@click.pass_obj
def command(options: CommonOptions):
    '''Render an image with a rotoscope-like style.'''
    img = options.load_image()
    effect = RotoscopeEffect()
    if not effect.can_process(img):
        raise click.ClickException('Cannot processing image.')
    options.save_image(effect.apply(img))
