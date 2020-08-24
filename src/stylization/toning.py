from typing import NamedTuple

import click
import numpy as np
from skimage import filters
from skimage.color import rgb2gray
from skimage.util import img_as_float

from ._types import CommonOptions, Effect
from .effects import SimpleThreshold, DetailedToning, LineCleanup


class ToningEffect(Effect):
    '''Render an image as a set of discrete grey-levels.

    The effect uses the eXtended Difference of Gaussians (xDoG) algorithm, with
    some morphilogical edge filtering to avoid aliasing.
    '''
    class Config(NamedTuple):
        strength: float = 10
        blur: float = 0.2
        k: float = 1.6

    def __init__(self, style: Effect, config: Config = Config()):
        '''Initialize the tone-effect filter.

        Parameters
        ----------
        style: Effect
            the toning style, implemented via an effect filter
        config: Config
            runtime configuration
        '''
        self._config = config
        self._style = style

    def _validate(self, img: np.ndarray):
        if img.dtype != np.uint8 and img.dtype != np.float:
            raise ValueError('Only 8-bit and floating point images are supported.')

    def _implementation(self, img: np.ndarray) -> np.ndarray:
        img = img_as_float(rgb2gray(img))

        # 1. Calculate the xDoG response.
        G1 = filters.gaussian(img, self._config.blur)
        G2 = filters.gaussian(img, self._config.k * self._config.blur)
        DoG = G1 - G2
        xDoG = G1 + self._config.strength*(DoG)

        # 2. Apply the toning.
        toned = self._style.apply(xDoG)

        return toned


@click.command('comicbook')
@click.option('-s', '--style', type=click.Choice(['two-tone', 'detailed']),
              default='two-tone', help='Toning style.')
@click.option('-b', '--blur', default=1.0,
              help='Size of the blurring filter, in pixels.')
@click.option('-p', '--strength', default=10,
              help='Strength of the edge filter.')
@click.option('-t', '--threshold', default=0.5,
              help='Toning threshold (on [0, 1]).')
@click.option('-d', '--detail', default=10,
              help='Detail amount, used with the \'detailed\' tone style.')
@click.pass_obj
def command(options: CommonOptions, style: str, blur: float, strength: float,
            threshold: float, detail: float):
    '''Render an using a two-tone, almost comicbook-like effect.'''
    img = options.load_image()

    styling: Effect
    if style == 'two-tone':
        styling = SimpleThreshold(threshold)
    elif style == 'detailed':
        styling = DetailedToning(threshold, detail)

    config = ToningEffect.Config(strength=strength, blur=blur)
    effect = ToningEffect(styling, config)
    if not effect.can_process(img):
        raise click.ClickException('Cannot processing image.')

    out = effect.apply(img)

    if style == 'two-tone':
        out = LineCleanup().apply(out)

    options.save_image(out)
