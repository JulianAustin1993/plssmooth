import basis
import numpy as np
import pytest

import plssmooth


def test_plss():
    domain = (-1, 1)
    C = np.random.normal(0, 5, (8, 10))
    bs = basis.Bspline(domain, 8, 3)
    t = np.linspace(*domain, 128)
    B = bs(t)
    P = bs.penalty(2)
    Y_true = np.matmul(B, C).T
    C_hat, log_lambda = plssmooth.plss(Y_true, B, P, -8.0)
    assert np.allclose(C_hat, C, atol=1e-4)


def test_plss_rgcv():
    domain = (-1, 1)
    C = np.random.normal(0, 5, (8, 10))
    bs = basis.Bspline(domain, 8, 3)
    t = np.linspace(*domain, 128)
    B = bs(t)
    P = bs.penalty(2)
    Y_true = np.matmul(B, C).T
    C_hat, log_lambda = plssmooth.plss_rgcv(Y_true, B, P, gamma=0.4)
    assert np.allclose(C_hat, C, atol=1e-4)


class TestPLSS:
    def test_method(self):
        model = plssmooth.PLSS()
        assert model.method == "fixed"
        with pytest.raises(NotImplementedError):
            model.method = 'random'
        model.method = 'rgcv'
        assert model.method == 'rgcv'

    def test_options(self):
        model = plssmooth.PLSS(options={})
        assert model.options == {"bounds": [-8, 8], "N": 20}
        with pytest.raises(ValueError):
            model.options = 'error'
        model.options = {'bounds': [-6, 6], 'N': 50}
        assert model.options == {'bounds': [-6, 6], 'N': 50}

    def test_fit(self):
        domain = (-1, 1)
        C = np.random.normal(0, 5, (8, 10))
        bs = basis.Bspline(domain, 8, 3)
        t = np.linspace(*domain, 128)
        B = bs(t)
        P = bs.penalty(2)
        Y_true = np.matmul(B, C).T
        model = plssmooth.PLSS(method='rgcv', options={'bounds': [-8, 8], 'N': 50})
        C_hat, log_lambda = model.fit(Y_true, B, P)
        assert log_lambda != 0
        assert np.allclose(np.matmul(B, C_hat).T, Y_true, atol=1e-3)
