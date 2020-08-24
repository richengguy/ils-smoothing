import numpy as np
from scipy.ndimage import convolve

from ._types import Effect


class SimpleThreshold(Effect):
    '''A basic thresholding operator.

    This simply sets the output value to '1' if the intensity :math:`I` is
    greater than some threshold and '0' otherwise.
    '''
    def __init__(self, level: float):
        '''Initialize the toning operator.

        Parameters
        ----------
        level : float
            threshold level; must be between 0 and 1
        '''
        self.level = level  #: threshold level

    def _implementation(self, img: np.ndarray) -> np.ndarray:
        out = np.zeros_like(img)
        out[img > self.level] = 1
        return out

    def _validate(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError('Image must be greyscale.')

        if img.dtype != np.float:
            raise ValueError('Image must be floating point.')


class SoftThreshold(Effect):
    '''Apply a soft threshold to an image.

    It applies

    .. math::

        T(I; \\epsilon, \\phi) = \\frac{1}{2}\\left\\(1 + \\tanh(2\\pi x) \\right\\),

    where

    .. math::

        x = \\phi(x - \\epsilon).
    '''
    def __init__(self, level: float, scale: float):
        '''Initialize the threshold effect.

        Parameters
        ----------
        level : float
            threshold where edges are boosted/suppressed
        scale : float
            Adjust how sharp the thresold is.
        '''
        self._level = level
        self._scale = scale

    def _validate(self, img: np.ndarray):
        if img.dtype != np.float:
            raise ValueError('Image must be floating-point.')

    def _implementation(self, img: np.ndarray) -> np.ndarray:
        x = self._scale*(img - self._level)
        return 0.5*(np.tanh(2*np.pi*x) + 1)


class DetailedToning(Effect):
    '''A thresholded toning style that preserves some image detail.

    This implements

    .. math::

        T(I; \\epsilon, \\phi) = \\begin{cases}
                1 & I > \\epsilon \\
                1 + \\tanh\\left\\{ \\phi (I - \\epsilon) \\right\\} & I \\le \\epsilon
               \\end{case},

    which will impose a hard threshold if the image intensity :math:`I` is
    greater than :math:`\\epsilon` and a soft threshold if :math:`I` is less
    than that value.  This assumes that :math:`I \\in [0, 1]`.
    '''
    def __init__(self, level: float, scaling: float):
        '''Initialize the toning operator.

        Parameters
        ----------
        level : float
            threshold level
        scaling : float
            detail scaling
        '''
        self.level = level  #: threshold level :math:`\\epsilon`
        self.scaling = scaling  #: detail scaling amount :math:`\\phi`

    def _implementation(self, img: np.ndarray) -> np.ndarray:
        out = np.zeros_like(img)

        mask = out > self.level
        out[mask] = 1
        out[~mask] = 0.5*(1 + np.tanh(self.scaling*(img[~mask] - self.level)))

        return out

    def _validate(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError('Image must be greyscale.')

        if img.dtype != np.float:
            raise ValueError('Image must be floating point.')


class LineCleanup(Effect):
    '''Performs some simple processing to clean up any thresholded image lines.

    This uses a straightforward corner detection approach to find corners and
    then compute the average value at that location.
    '''
    def _validate(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError('Image must be greyscale.')

    def _implementation(self, img: np.ndarray) -> np.ndarray:
        h = np.array([[0, 1, -1]])

        horz_edges = convolve(img, h, mode='nearest')
        vert_edges = convolve(img, h.T, mode='nearest')

        corners = np.logical_and(np.abs(horz_edges) > 0, np.abs(vert_edges) > 0)
        blur = convolve(img, np.ones((3, 3)), mode='nearest') / 9

        out = img.copy()
        out[corners] = blur[corners]
        return out
