from typing import NamedTuple

import click
import numpy as np
from skimage.util import img_as_float, img_as_ubyte
from ils_smoothing import ILSSmoothingFilter

from ._types import CommonOptions, Effect
from .effects import LineCleanup, SoftThreshold


class RotoscopeEffect(Effect):
    '''Create a rotoscoping effect similar to "A Scanner Darkly".'''
    class Config(NamedTuple):
        '''Configuration options for the rotoscope effect.'''
        quantization: int = 0  #: colour quantization amount; set to '0' to disable
        line_cleanup: bool = False  #: enable the line cleanup algorithm
        debug_output: bool = False  #: enable debugging output

    def __init__(self, ils_filter: ILSSmoothingFilter,
                 edge_threshold: Effect,
                 config: Config = Config()):
        '''Initialize a new rotoscope effect object.

        See the :class:`RotoscopeEffect.Config` structure for the various
        options.

        Parameters
        ----------
        ils_filter : ILSSmoothingFilter
            the ILS filter performing the edge-aware smoothing
        edge_threshold : Effect
            effect used to manipulate edge response before drawing edge lines
            on the image
        config : Config, optional
            runtime configuration
        '''
        self._ils_filter = ils_filter
        self._edge_threshold = edge_threshold
        self._config = config

    def _implementation(self, img: np.ndarray) -> np.ndarray:
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

            edge_response[:, :, i] = state.edge_response()

        if self._config.debug_output:
            imsave('smoothed.png', img_as_ubyte(smoothed))

        # The remaining processing is easier to work with if in float.
        if img.dtype == np.uint8:
            smoothed = img_as_float(smoothed)

        # 2. Quantize the colours to distinct levels.
        if self._config.quantization > 0:
            quantized = np.round(self._config.quantization*smoothed) / self._config.quantization
        else:
            quantized = smoothed.copy()

        if self._config.debug_output:
            imsave('quantized.png', quantized)

        # 3. Take the average response across the colour channels.
        edge_response = np.mean(edge_response, axis=2)
        edge_response = edge_response / edge_response.max()

        if self._config.debug_output:
            imsave('edge-response.png', edge_response)

        # 4. Use a non-linear function to boost strong edges and suppress weak
        #    ones.
        scaled = self._edge_threshold.apply(edge_response)
        scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())

        if self._config.line_cleanup:
            scaled = LineCleanup().apply(scaled)

        if self._config.debug_output:
            imsave('scaled-edges.png', scaled)

        # 5. Apply the edge mask onto the quantized image.
        mask = 1 - scaled
        out = quantized * mask[:, :, np.newaxis]

        if self._config.debug_output:
            imsave('mask.png', mask)

        if img.dtype == np.uint8:
            out = img_as_ubyte(out)

        return out

    def _validate(self, img: np.ndarray):
        '''Determine if the image can be processed.'''
        if img.dtype != np.uint8 and img.dtype != np.float:
            raise ValueError('Image must be either 8-bit or floating point.')

        if img.ndim != 3 and img.shape[2] != 3:
            raise ValueError('Image must be 3-channel RGB.')


@click.command('rotoscope')
@click.option('-s', '--smoothing', default=0.4,
              help='The amount of smoothing to apply.')
@click.option('-c', '--colours', default=0,
              help='Apply colour quantization, giving the effect of a reduced '
                   'colour pallet.')
@click.option('-t', '--edge-threshold', default=0.1,
              help='Control when edges will appear in the output.')
@click.option('-e', '--edge-scaling', default=10,
              help='Control how gradually edges appear.')
@click.pass_obj
def command(options: CommonOptions, smoothing: float, colours: int,
            edge_threshold: float, edge_scaling: float):
    '''Render an image with a rotoscope-like style.'''
    img = options.load_image()

    ils = ILSSmoothingFilter(5, smoothing, 8)
    edge_filter = SoftThreshold(edge_threshold, edge_scaling)
    config = RotoscopeEffect.Config(quantization=colours)
    effect = RotoscopeEffect(ils, edge_filter, config)

    if not effect.can_process(img):
        raise click.ClickException('Cannot processing image.')
    options.save_image(effect.apply(img))
