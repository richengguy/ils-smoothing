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
        from skimage.io import imsave
        smoothed = np.zeros_like(img)
        edge_response = np.zeros(img.shape)

        # 1. Run the ILS filter to get a smoothed colour, while also tapping the
        #    edge responses from within the filter.
        for i in range(img.ndim):
            smoothed[:, :, i] = self._ils_filter.apply(img[:, :, i])

            state: ILSSmoothingFilter.FilterStages
            if (state := self._ils_filter.filter_stages) is None:  # type: ignore
                raise RuntimeError('Could not get internal filter state.')

            edge_response[:, :, i] = state.edge_filter()

        imsave('smoothed.png', smoothed)

        # 2. Take the average response across the colour channels.
        edge_response = np.mean(edge_response, axis=2)
        delta = edge_response.max() - edge_response.min()
        imsave('edge-response.png', edge_response)

        # 3. Map into onto a known range, e.g. [-1, 1], and then apply a
        #    non-linear function to create a basic edge map.
        scaled = edge_response/delta
        mask = np.abs(scaled) < 0.01
        # mask = np.logical_not(mask)
        imsave('scaled.png', scaled)
        imsave('mask.png', np.logical_not(mask))

        overlay = smoothed * mask[:, :, np.newaxis]
        imsave('overlay.png', overlay)

        return overlay

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
