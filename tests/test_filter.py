import numpy as np
from numpy.testing import assert_allclose

import scipy.fft

import pytest

from ils_smoothing.filter import _charbonnier_derivative, _frequency_response


class TestFrequencyResponse:
    def test_basic(self):
        x = np.array([[1, -1]])
        sz = (10, 10)

        H = _frequency_response(x, sz)
        h = scipy.fft.ifft2(H)

        expected = np.zeros(sz)
        expected[:x.shape[0], :x.shape[1]] = x

        assert H.shape == sz
        assert_allclose(h.real, expected, atol=1e-9)

    def test_invalid_filter_size_raises_exception(self):
        with pytest.raises(ValueError):
            _frequency_response(np.zeros((10,)), (10, 10))

    def test_invalid_output_height_raises_exception(self):
        with pytest.raises(ValueError):
            _frequency_response(np.zeros((3, 3)), (1, 10))

    def test_invalid_output_width_raises_exception(self):
        with pytest.raises(ValueError):
            _frequency_response(np.zeros((3, 3)), (10, 1))


class TestCharbonnierDerivative:
    def test_basic(self):
        assert _charbonnier_derivative(1, 1, 1e-4) == 0.9999500037496876
        assert _charbonnier_derivative(1, 2, 1e-4) == 2.0
        assert _charbonnier_derivative(2, 1, 1e-4) == 0.9999875002343701172943091202333
        assert _charbonnier_derivative(-1, 1, 1e-4) == -_charbonnier_derivative(1, 1, 1e-4)
