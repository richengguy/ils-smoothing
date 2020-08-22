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

        \\frac{d \\phi(x)}{dx} = px(x^2 + \\epsilon)^\\(\\frac{p}{2} - 1\\).

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
    return p * x * (x**2 + e) ** (0.5 * p - 1)


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
        controls the amount that the filter smooths by; this is :math:`\\lambda`
        in the paper
    edge_preservation : float
        controls how well the filter preserves edges; this is :math:`p` in the
        paper
    iterations : int
        the number of iterations that that are performed within the filter
    epsilon : float
        small, non-zero constant, i.e. :math:`\\epsilon`
    '''
    def __init__(self, smoothing: float, edge_preservation: float,
                 iterations: int = 4, epsilon: float = 1e-4):
        '''Initialize a new ILS filter.

        Parameters
        ----------
        smoothing : float
            filter smoothing parameter
        edge_preservation : float
            edge preservation amount, must be between 0 and 1
        iterations : int, optional
            number of filter iterations, defaults to '4'
        epsilon : float, optional
            small, non-zero value to ensure the Charbonnier penalty is non-zero
            at zero; optional and defaults to '0.0001'
        '''
        if edge_preservation < 0 or edge_preservation > 1:
            raise ValueError('Edge preservation value must be between 0 and 1.')

        self.smoothing = smoothing
        self.edge_preservation = edge_preservation
        self.iterations = iterations
        self.epsilon = epsilon
        self._c = self.smoothing * self.epsilon ** (self.smoothing/2 - 1)

    def apply(self, img: np.ndarray) -> np.ndarray:
        '''Apply the filter onto some image.

        Parameters
        ----------
        img : numpy.ndarray
            input image of any size and type; must be greyscale

        Returns
        -------
        numpy.ndarray
            filtered output image

        Raises
        ------
        ValueError
            if the input image isn't greyscale
        '''
        if img.ndim != 2:
            raise ValueError('Input must be a monochrome image.')
        if img.ndim == 3 and img.shape[0] != 1:
            raise ValueError('Input must be a monochrome image.')

        img = img_as_float32(img)
        grad = np.array([[1, -1]])

        # Pre-compute all of the static Fourier transforms.
        fourier_delta_x = _frequency_response(grad, img.shape)
        fourier_delta_y = _frequency_response(grad.T, img.shape)
        fourier_img = fft2(img)

        # Compute the denominator of equation (9).  This is done in two steps
        # for clarity.
        denominator = (np.conj(fourier_delta_x)*fourier_delta_x +
                       np.conj(fourier_delta_y)*fourier_delta_y)
        denominator = 1 + (self._c / 2) * self.smoothing * denominator

        # Prepare for the iterative part.
        output = img

        # Run the loops.
        for i in range(self.iterations):
            # 1. Compute the gradients of the smoothed image.
            doutput_x = convolve(output, grad)
            doutput_y = convolve(output, grad.T)

            # 2. Compute the "update" images; this is equation (7) in the paper.
            u_x = self._c * doutput_x - _charbonnier_derivative(doutput_x, self.edge_preservation, self.epsilon)  # noqa: E501
            u_y = self._c * doutput_y - _charbonnier_derivative(doutput_y, self.edge_preservation, self.epsilon)  # noqa: E501

            # 3. Compute the "update" derivatives.
            # The sign is flipped so that it exploits the symmetry in a
            # real-valued FFT, namely x(-t) = -X*(w).  This allows a major
            # simplification in the update equations.
            du_x = convolve(u_x, -grad)
            du_y = convolve(u_y, -grad.T)

            # 4. Compute the numerator of equation (9).
            numerator = fourier_img + 0.5 * self.smoothing * fft2(du_x + du_y)

            # 5. Compute the inverse FFT and take the real value.
            output = ifft2(numerator / denominator).real

        return output
