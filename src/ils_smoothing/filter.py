import enum
from typing import Optional, Tuple

import numpy as np
from scipy.fft import fft2, ifft2
from skimage.util import img_as_float, img_as_ubyte


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


class Direction(enum.IntEnum):
    '''Gradient filtering direction.'''
    VERTICAL = 0
    HORIZONTAL = 1


def gradient_frequency(direction: Direction, outsz: Tuple[int, int]) -> np.ndarray:
    '''Compute the frequency response of the backwards difference kernel.

    A backwards difference, i.e. :math:`\\Delta x[n] = x[n] - x[n - 1]`, is
    equivalent to the filtering kernel
    :math:`h[n] = \\begin{bmatrix} 1 & -1 \\end{bmatrix}`.  When doing any sort
    of frequency-domain filtering we often need to zero-pad so that the filter's
    frequency response image is the same szie the image being filtered.  Because
    the kernel is so small, we can directly calculate the DFT.

    Given a sequence :math:`h[n]` of length :math:`N`, where :math:`h[0] = 1`
    and :math:`h[1] = -1`, the DFT :math:`\\mathcal{F}\\{h[n]\\} = H[k]` is

    .. math::

        H[k] &= \\sum_{n=0}^{N-1} h[n]e^{-j 2\\pi n \\frac{k}{N}} \\\\
             &= (1)e^{-j 2\\pi \\frac{k}{N}(0)} + (-1)e^{-j 2\\pi \\frac{k}{N}(1)} \\\\
             &= 1 - e^{-j 2\\pi \\frac{k}{N}}.

    Extending this into 2D is trivial. We can just replicate the 1D solution for
    each row because all of the values "below" :math:`h[n]` are zero from the
    zero-padding,  This is just an outcome of the DFT being separable (each
    dimension is its own 1D DFT, so a 2D DFT is two 1D DFTs, a 3D DFT is three
    1D DFTs and so on).

    Parameters
    ----------
    direction : Direction
        the kernel's direction
    outsz : Tuple[int, int]
        output size for the frequency domain image

    Returns
    -------
    ndarray
        a numpy array with the frequency-domain response for the gradient filter
    '''  # noqa: E501
    rows, cols = outsz
    if direction == Direction.HORIZONTAL:
        N = cols
    elif direction == Direction.VERTICAL:
        N = rows

    k = np.arange(N)
    w = 2 * np.pi * k / N
    H = 1 - np.exp(-1j * w)

    if direction == Direction.HORIZONTAL:
        return np.tile(H, (rows, 1))
    elif direction == Direction.VERTICAL:
        return np.tile(H[np.newaxis].T, (1, cols))


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
    use_padding : bool
        if enabled, then the filter will apply padding to account for the fact
        that frequency-domain filtering can introduce border artifacts
    '''
    class FilterStages:
        '''Data type that stores the filter's intermediate states.'''
        def __init__(self, padding: int, sz: Tuple[int, int],
                     hpf: np.ndarray, lpf: np.ndarray):
            self._padding = padding
            self._sz = sz
            self._edge_filter = hpf.copy()
            self._low_pass_filter = lpf.copy()

        def edge_filter(self, *, is_frequency: bool = False) -> np.ndarray:
            '''The edge filtering component of the ILS filter.

            Parameters
            ----------
            is_frequency : bool, optional
                determine the returned value is in the frequency or spatial
                domain; default is ``False``

            Returns
            -------
            np.ndarray
                filter response
            '''
            return self._return_filter(self._edge_filter, is_frequency)

        def low_pass_filter(self, *, is_frequency: bool = False) -> np.ndarray:
            '''The low pass filtering component in the ILS filter.

            Parameters
            ----------
            is_frequency : bool, optional
                determine the returned value is in the frequency or spatial
                domain; default is ``False``

            Returns
            -------
            np.ndarray
                filter response
            '''
            return self._return_filter(self._low_pass_filter, is_frequency)

        def _return_filter(self, response: np.ndarray, is_frequency: bool) -> np.ndarray:
            if is_frequency:
                return response
            out = ifft2(response).real
            return out[self._padding:(self._sz[0] + self._padding),
                       self._padding:(self._sz[1] + self._padding)]

    def __init__(self, smoothing: float, edge_preservation: float,
                 iterations: int = 4, epsilon: float = 1e-4,
                 use_padding: bool = True):
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
        use_padding : bool, optional
            enable/disable border padding; optional and default is ``True``

        Raises
        ------
        ValueError
            if any of the properties are invalid, e.g. zero or negative
            iterations, negative smoothing, etc
        '''
        if iterations < 1:
            raise ValueError('Number of iterations must be larger than 1.')

        if smoothing < 0:
            raise ValueError('Smoothing value cannot be negative.')

        if edge_preservation < 0 or edge_preservation > 1:
            raise ValueError('Edge preservation value must be between 0 and 1.')

        self.smoothing = smoothing
        self.edge_preservation = edge_preservation
        self.iterations = iterations
        self.epsilon = epsilon
        self.use_padding = use_padding
        self._c = self.edge_preservation * self.epsilon ** (self.edge_preservation/2 - 1)
        self._internal_state: Optional[ILSSmoothingFilter.FilterStages] = None

    @property
    def filter_stages(self) -> Optional[FilterStages]:
        '''FilterStages: The internal state of the ILS filter after execution.

        This will be ``None`` if the filter hasn't been used.  Call this after
        using :meth:`apply` to get the filter's internal state.
        '''
        return self._internal_state

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
        is_ubyte = img.dtype == np.uint8
        if img.ndim != 2:
            raise ValueError('Input must be a monochrome image.')
        if img.ndim == 3 and img.shape[0] != 1:
            raise ValueError('Input must be a monochrome image.')

        # Setup the image (includes padding).
        if self.use_padding:
            pad_width = max(int(img.shape[0]/8), int(img.shape[1]/8))
            padded = np.pad(img_as_float(img), pad_width, mode='edge')
        else:
            pad_width = 0
            padded = img_as_float(img)

        # Pre-compute all of the static Fourier transforms.
        fourier_delta_x = gradient_frequency(Direction.HORIZONTAL, padded.shape)
        fourier_delta_y = gradient_frequency(Direction.VERTICAL, padded.shape)

        # Compute the denominator of equation (9).  This is done in two steps
        # for clarity.  Note that this is the exact inverse of a Laplacian
        # sharpening filter.
        laplacian = np.abs(fourier_delta_x)**2 + np.abs(fourier_delta_y)**2
        low_pass_filter = 1 / (1 + 0.5 * self._c * self.smoothing * laplacian)

        # Perform the iterative filtering.
        output = padded
        fourier_output = fft2(output)

        for i in range(self.iterations):
            # 1. Compute the gradients of the smoothed image.
            doutput_x = ifft2(fourier_delta_x * fourier_output).real
            doutput_y = ifft2(fourier_delta_y * fourier_output).real

            # 2. Compute the "edge penalty" images; this is equation (7) in the
            #    paper.
            u_x = self._c * doutput_x - _charbonnier_derivative(doutput_x, self.edge_preservation, self.epsilon)  # noqa: E501
            u_y = self._c * doutput_y - _charbonnier_derivative(doutput_y, self.edge_preservation, self.epsilon)  # noqa: E501

            # 3. Compute the edge filter by essentially computing the gradients
            #    of the penalty images.
            U_x = np.conj(fourier_delta_x) * fft2(u_x)
            U_y = np.conj(fourier_delta_y) * fft2(u_y)
            edge_penalty = U_x + U_y

            # 4. Compute the numerator of equation (9).  This is
            numerator = fourier_output + 0.5 * self.smoothing * edge_penalty

            # 5. Compute the inverse FFT and take the real value.
            fourier_output = numerator * low_pass_filter

        # Store the internal state.
        self._internal_state = ILSSmoothingFilter.FilterStages(pad_width,
                                                               img.shape,
                                                               edge_penalty,
                                                               low_pass_filter)

        # Transform back into spatial domain and remove the excess padding.
        output = np.abs(ifft2(fourier_output))
        output = output[pad_width:(pad_width+img.shape[0]), pad_width:(pad_width+img.shape[1])]
        if is_ubyte:
            output[output < 0] = 0
            output[output > 1] = 1
            output = img_as_ubyte(output)
        return output
