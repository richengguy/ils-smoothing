from typing import Tuple

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import convolve
from skimage.util import img_as_float32


def _charbonnier_derivative(x: np.ndarray, p: float, e: float) -> np.ndarray:
    '''Computes the derivate of the Charbonnier penalty function.

    The Charbonnier penalty is defined as

    ..math::

        \\phi(x) = \\left(x^2 + \\epsilon)^\\frac{p}{2}.

    Therefore, its derivative, with respect to :math:`x`, is given by

    ..math::

        \\frac{d \\phi(x)}{dx} = p(x^2 + \\epsilon)^\\(\\frac{p}{2} - 1\\).

    This is used in the filter's optimization loop.

    Parameters
    ----------
    x : np.ndarray
        value going into the penalty function
    p : float
        the "strength" of penalizer
    e : float
        value of :math:`\\epsilon`

    Returns
    -------
    np.ndarray
        the value of the derivative
    '''
    return p * (x**2 + e) ** (p/2 - 1)


def _frequency_response(h: np.ndarray, sz: Tuple[int, int]) -> np.ndarray:
    '''Generate the frequency response of a given filter kernel.

    The frequency response, :math:`H`, of the kernel, :math:`h`, is a
    zero-padded 2D FFT.

    Parameters
    ----------
    h : numpy.ndarray
        a :math:`N \\times M` filter kernel
    sz : rows, columns
        a 2-element tuple containing the output size

    Returns
    -------
    numpy.ndarray
        the zero-padded 2D FFT of the filter kernel, i.e.
        :math:`\\mathcal{F}\\{h\\}`

    Raises
    ------
    ValueError
        if the output size is smaller than the kernel
    '''
    rows, cols = sz
    if h.ndim != 2:
        raise ValueError('Filter must be 2D.')

    if h.shape[0] > rows or h.shape[1] > cols:
        raise ValueError('Output shape must be larger that the filtering kernel.')

    h_padded = np.zeros(sz)
    h_padded[:h.shape[0], :h.shape[1]] = h
    return fft2(h_padded)


class ILSSmoothingFilter:
    '''Implementation of the Iterative Least Squares smoothing filter.

    This is an implementation of the "Real-time Smoothing via Iterative Least
    Squares" smoothing algorithm by Liu et al.

    Attributes
    ----------
    smoothing : float
        controls the amount that the filter smooths by
    edge_preservation : float
        controls how well the filter preserves edges
    iterations : int
        the number of iterations that that are performed within the filter
    '''
    def __init__(self, smoothing: float, edge_preservation: float, iterations: int = 4):
        '''Initialize a new ILS filter.

        Parameters
        ----------
        smoothing : float
            filter smoothing parameter
        edge_preservation : float
            edge preservation amount, must be between 0 and 1
        iterations : int, optional
            number of filter iterations, defaults to '4'
        '''
        if edge_preservation < 0 or edge_preservation > 1:
            raise ValueError('Edge preservation value must be between 0 and 1.')

        self.smoothing = smoothing
        self.edge_preservation = edge_preservation
        self.iterations = iterations

    def apply(self, img: np.ndarray) -> np.ndarray:
        '''Apply the filter onto some image.

        Parameters
        ----------
        img : numpy.ndarray
            input image of any size

        Returns
        -------
        numpy.ndarray
            filtered output image
        '''
