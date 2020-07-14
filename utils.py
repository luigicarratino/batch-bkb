# MIT License
#
# Copyright (c) 2020 Laboratory for Computational and Statistical Learning
#
# authors: Daniele Calandriello, Luigi Carratino
# email:   daniele.calandriello@iit.it
# Website: http://lcsl.mit.edu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sklearn.gaussian_process.kernels import Kernel


class FastMixinKernel(Kernel):
    def __init__(self, gp_kernel, pairwise_kernel):
        self.gp_kernel = gp_kernel
        self.pairwise_kernel = pairwise_kernel

    def __call__(self, X, Y=None, **kwargs):
        if Y is None:
            job_size = X.shape[0]**2
        else:
            job_size = X.shape[0] * Y.shape[0]

        if job_size < 1000**2:
            return self.gp_kernel(X, Y)
        else:
            return self.pairwise_kernel(X, Y, **kwargs)

    def diag(self, X):
        return self.gp_kernel.diag(X)

    def is_stationary(self):
        return self.gp_kernel.is_stationary()

def diagonal_dot(X, dot):
    """Given a similarity function dot, and data X, calculate diagonal of the similarity dot on the data X
    """
    n = X.shape[0]
    if isinstance(dot, FastMixinKernel):
        res = dot.diag(X)
    else:
        res = np.zeros(n)
        for i in range(n):
            res[i] = dot(X[i, :])

    return res

def stable_invert_root(U, S):
    """Given an eigendecomposition with eigenvectors U and eigenvalues S compute a pseudoinverse
    by removing numerically unstable eigenvalues

    Return a thin decomposition (i.e. without unstable eigenvectors/eigenvalues)
    """

    n = U.shape[0]

    assert U.shape == (n, n)
    assert S.shape == (n,)

    thresh = S.max() * max(S.shape) * np.finfo(S.dtype).eps
    stable_eig = np.logical_not(np.isclose(S, 0., atol=thresh))
    m = sum(stable_eig)

    U_thin = U[:, stable_eig]
    S_thin = S[stable_eig]

    assert U_thin.shape == (n, m)
    assert S_thin.shape == (m,)

    S_thin_inv_root = (1 / np.sqrt(S_thin)).reshape(-1, 1)

    assert np.all(np.isfinite(S_thin_inv_root))

    return U_thin, S_thin_inv_root