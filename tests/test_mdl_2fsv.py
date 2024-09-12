import math
import numpy as np
import pytest

from ajdmom.mdl_2fsv.euler import r2FSV
from ajdmom.mdl_2fsv.mom import m
from ajdmom.mdl_2fsv.cmom import cm
from ajdmom.mdl_2fsv.cov import cov


@pytest.fixture
def settings():
    # h should be in par, since it is needed for m(), cm(), cov()
    par = {'mu': 0.125, 'k1': 0.1, 'theta1': 0.25, 'sigma_v1': 0.1,
           'k2': -math.log(math.exp(-0.1) / 3), 'theta2': 0.25 * 3, 'sigma_v2': 0.1 * 2,
           'h': 1}
    v0 = (par['theta1'], par['theta2'])
    y_series = r2FSV(v0, par, N=16000 * 1000, n_segment=10, h=1)
    # threshold for reporting discrepancy
    rel_err = 0.150  # 15%
    abs_err = 0.001
    return par, y_series, rel_err, abs_err


def test_m(settings):
    """Test moment computing"""
    par, y_series, rel_err, abs_err = settings
    #
    ns = [1, 2, 3, 4, 5]
    print("")
    for n in ns:
        expected = np.mean(y_series ** n)
        actual = m(n, par)
        expected = pytest.approx(expected, rel=rel_err, abs=abs_err)
        msg = "Diff in the {}th moment is > {:} or {:.0%} "
        msg += "between theory and sample."
        devi = max(expected.abs, expected.rel * abs(expected.expected))
        temp = "{:d}-th moment: actual({: f}) V.S. expected({: f} ± {:f})"
        print(temp.format(n, actual, expected.expected, devi))
        assert actual == expected, msg.format(n, abs_err, rel_err)


def test_cm(settings):
    """Test central moment computing"""
    par, y_series, rel_err, abs_err = settings
    #
    ns = [2, 3, 4, 5]
    diff = y_series - np.mean(y_series)
    del y_series
    print("")
    for n in ns:
        expected = np.mean(diff ** n)
        actual = cm(n, par)
        expected = pytest.approx(expected, rel=rel_err, abs=abs_err)
        msg = "Diff in the {}th central moment is > {:} or {:.0%} "
        msg += "between theory and sample."
        devi = max(expected.abs, expected.rel * abs(expected.expected))
        temp = "{:d}-th central moment: actual({: f}) V.S. expected({: f} ± {:f})"
        print(temp.format(n, actual, expected.expected, devi))
        assert actual == expected, msg.format(n, abs_err, rel_err)


def test_cov(settings):
    """Test covariance computing"""
    par, y_series, rel_err, abs_err = settings
    #
    N = len(y_series)
    pre = y_series[0:(N - 1)]
    nxt = y_series[1:N]
    #
    nms = [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3), (4, 1), (3, 2), (2, 3), (1, 4)]
    print("")
    for nm in nms:
        n, m = nm
        expected = np.cov(pre ** n, nxt ** m)[0][1]
        actual = cov(n, m, par)
        expected = pytest.approx(expected, rel=rel_err, abs=abs_err)
        msg = "Diff in the lag-1 autocov({},{}) is > {:} or {:.0%} "
        msg += "between theory and sample."
        devi = max(expected.abs, expected.rel * abs(expected.expected))
        temp = "cov({:d},{:d}): actual({: f}) V.S. expected({: f} ± {:f})"
        print(temp.format(n, m, actual, expected.expected, devi))
        assert actual == expected, msg.format(n, m, abs_err, rel_err)
