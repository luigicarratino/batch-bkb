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

"""Implementation of the BKB [1] and Batch-BKB (BBKB) [2] algorithms.

Both BKB and BBKB are designed to maximize an unknown function f over
a set of arms (candidates) with size A.
All algorithms (we use here BKB as an example) are designed to be used in
the following online protocol:

    opt_alg = BKB(**init_args)
    y_init = f(arms[init_idx_set, :])
    opt_alg.initialize(arms, init_idx_set, y_init)
    for t in range(T):
        idx_chosen_arms, _ = opt_alg.predict()
        y_t = f(arms[idx_chosen_arms, :])
        opt_alg.update(idx_chosen_arms, y_t)

The protocol is based on an three methods: initialize(...), predict(...),
and update(...), following this specification.

opt_alg.initialize(arms, init_idx_set, y_init) initializes the object for the
optimization loop using a set of arms, and a short list of s evaluations of f:
- arms: a set of A candidates represented as d-dimensional vectors, collected in a (A x d) numpy.ndarray
        a copy of arms is stored in the opt_alg object to be used with the predict() and update() methods.
- init_idx_set: a list containing the indices of the arms pre-evaluated in order to initialize the optimization loop
- y_init: an (s,) numpy.ndarray containing the evaluations of f on the initialization arms

opt_alg.predict() returns a list containing the indices of the most promising
arms to be evaluated (according to the model contained in opt_alg).

opt_alg.update(idx_chosen_arms, y_t) updates the model contained in opt_alg
using the feedback y_t evaluated on the arms indexed by idx_chosen_arms.
- idx_chosen_arms: is a list containing scalar indices that index into the arms matrix.
  That is, if idx_chosen_arms contains the index 10, we need to evaluate the 10th
  arm (10th row of the arms matrix), and provide this evaluation to update(...).
- y_t: (b,) numpy.ndarray containing the evaluations of f on the b idx_chosen_arms arms.

[1] Gaussian Process Optimization with Adaptive Sketching: Scalable and No Regret
    COLT 2019 by D. Calandriello, L. Carratino, A. Lazaric, M. Valko, L. Rosasco.
[2] Near-linear time Gaussian process optimization with adaptive batching and resparsification
    ICML 2020 by D. Calandriello*, L. Carratino*, A. Lazaric, M. Valko, L. Rosasco.
"""

import numpy as np
from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError

from utils import diagonal_dot, stable_invert_root


class BKB(object):
    """
    """
    def __init__(self, lam=1., dot=None, fnorm=1., noise_variance=1., delta=.5, qbar=1, verbose=0):
        """
        Args:
            lam: float, regularization parameter lambda used in the Gaussian Process.
            Larger regularization reduces uncertainty and makes approximation easier
            and BKB's iteration computationally faster.
            However large regularization increases model bias and can lead to under-exploration.
            A common heuristic is to set lam to be equal to the prior on the
            noise_variance, or to start with a small lam and increase it
            if the model over-explores or becomes too hard to approximate
            (i.e. BKB runs too slowly).

            dot: prior GP covariance (a.k.a. the RKHS's dot or inner product).
            It must satisfy the interface defined by scikit-learn's
            implementation of kernel functions in Gaussian Processes.
            Any of the kernels provided by sklearn (e.g. sklearn.gaussian_process.kernels.RBF
            or sklearn.gaussian_process.kernels.PairwiseKernel) should work out of the box.
            For a more sophisticated example, see the provided utils.FastMixinKernel

            fnorm: float, upper bound on the norm of the reward function.
            Larger fnorm requires (and lead to) more exploration and slower convergence.
            A common heuristic is to try a small fnorm value at first, and
            increase it if the algorithm over-explores.

            noise_variance: float, upper bound on the variance of the noise of the reward function.
            High variance requires (and leads to) more exploration and slower
            convergence.
            A common heuristic is to try a small noise_variance value at first, and
            increase it if the algorithm over-explores.

            delta: float, parameter controlling the probability of maintaining
            valid confidence intervals at all steps. Smaller delta
            induces larger confidence intervals and slower convergence, but
            reduces the probability of selecting a sub-optimal arm asymptotically.
            A common heuristic is to try a constant delta value (e.g. 1/2) at first
            and decrease it if the algorithm over-explores, or to set delta = 1/T.

            qbar: float, oversampling parameter used to construct BKB's internal
            sparse GP approximation. The qbar parameter is used to increase the
            rank of the approximation by a qbar factor. This makes each
            prediction and update slower and more memory intensive, but
            reduces variance over-estimation and increases model accuracy,
            so the actual optimization runtime might increase or decrease.
            Empirically, a small factor qbar in [2,10] seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate.

            verbose: bool, controls verbosity of debug output, including progress bars
        """
        self.lam = lam
        self.dot = dot
        self.fnorm = fnorm
        self.noise_variance = noise_variance
        self.delta = delta
        self.qbar = qbar

        self.t = 1
        self.beta = 1
        self.logdet = 1
        self.k = 1
        self.m = 1
        self.d = 1

        self.arm_set = np.zeros((self.k, self.d))
        self.arm_set_norms = np.zeros(self.k)
        self.arm_set_embedded = np.zeros((self.k, self.m))
        self.arm_set_embedded_norms = np.zeros((self.k, 1))
        self.y = np.zeros(self.k)

        self.pulled_arms_count = np.zeros(self.k)
        self.pulled_arms_y = np.zeros(self.k)

        self.dict_arms_count = np.zeros(self.k)
        self.dict_arms_matrix = np.zeros((self.k, self.d))

        self.A = np.zeros((self.m, self.m))
        self.w = np.zeros(self.m)
        self.Q = np.zeros((self.m, self.m))
        self.R = np.zeros((self.m, self.m))

        self.means = np.zeros(self.k)
        self.variances = np.zeros(self.k)
        self.conf_intervals = np.zeros(self.k)

        self.verbose = verbose

    @property
    def pulled_arms_matrix(self):
        return self.arm_set_embedded[self.pulled_arms_count != 0, :]

    @property
    def unique_arms_pulled(self):
        return np.count_nonzero(self.pulled_arms_count)

    @property
    def dict_size(self):
        return self.dict_arms_count.sum()

    @property
    def dict_size_unique(self):
        return np.count_nonzero(self.dict_arms_count)

    @property
    def embedding_size(self):
        return self.m

    # update functions
    def _update_embedding(self):

        K_mm = self.dot(self.dict_arms_matrix)
        K_km = self.dot(self.arm_set, self.dict_arms_matrix)

        try:
            U, S, _ = svd(K_mm)
        except LinAlgError:
            U, S, _ = svd(K_mm, lapack_driver='gesvd')
            print("numerical problem, had to use other")

        U_thin, S_thin_inv_sqrt = stable_invert_root(U, S)
        self.arm_set_embedded = K_km.dot(U_thin * S_thin_inv_sqrt.T)
        self.arm_set_embedded_norms = np.linalg.norm(self.arm_set_embedded, axis=1)
        np.square(self.arm_set_embedded_norms, out=self.arm_set_embedded_norms)

        self.m = len(S_thin_inv_sqrt)
        assert(self.arm_set_embedded.shape == (self.k, self.m))
        assert(np.all(self.arm_set_embedded_norms
                      <= self.arm_set_norms + 100 * self.m**2 * np.finfo(self.arm_set_embedded_norms.dtype).eps))
        assert(np.all(np.isfinite(self.arm_set_embedded_norms)))

    def _reset_posterior(self, idx_to_update=None):
        #initialize A, P, and w
        pulled_arms_matrix = self.pulled_arms_matrix
        reweight_counts_vec = np.sqrt(self.pulled_arms_count[self.pulled_arms_count != 0].reshape(-1, 1))

        self.A = ((pulled_arms_matrix * reweight_counts_vec).T.dot(pulled_arms_matrix * reweight_counts_vec)
                  + self.lam * np.eye(self.m))

        self.Q, self.R = qr(self.A)

        self.w = solve_triangular(self.R, self.Q.T.dot(pulled_arms_matrix.T.dot(self.y[self.pulled_arms_count != 0])))

        self.means = self.arm_set_embedded.dot(self.w)

        assert np.all(np.isfinite(self.means))

        self._update_variances(idx_to_update)
        self.conf_intervals = self.beta * np.sqrt(self.variances)

    def _update_variances(self, idx_to_update=None):
        if idx_to_update is None:
            temp = solve_triangular(self.R,
                                    (self.arm_set_embedded.dot(self.Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
            temp *= self.arm_set_embedded
            self.variances = (self.arm_set_norms - self.arm_set_embedded_norms) / self.lam + np.sum(temp, axis=1)
        else:
            temp = solve_triangular(self.R,
                                    (self.arm_set_embedded[idx_to_update, :].dot(self.Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
            temp *= self.arm_set_embedded[idx_to_update, :]
            self.variances[idx_to_update] = (
                    (self.arm_set_norms[idx_to_update] - self.arm_set_embedded_norms[idx_to_update]) / self.lam
                    + np.sum(temp, axis=1)
            )
        assert np.all(self.variances >= 0.)
        assert np.all(np.isfinite(self.variances))

    def _update_beta(self):
        self.logdet = (self.variances * self.pulled_arms_count).sum() * np.log(self.pulled_arms_count.sum())
        self.beta = np.sqrt(self.lam) * self.fnorm + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))
        assert np.isfinite(self.beta)

    def initialize(self, arm_set, index_init, y_init):
        """Initialize the BKB algorithm

        Args:
            arm_set:  set of the d-dimensional A arms, is a (A x d) numpy.ndarray
            index_init: set of the indexes of the arms used to initialize BKB, is a list of s scalars
            y_init: feedback of the arms used for initialization, is a (s,) numpy.ndarray
        """
        self.k = arm_set.shape[0]
        self.t = len(index_init)
        self.m = len(np.unique(index_init))

        self.arm_set = arm_set
        self.arm_set_norms = diagonal_dot(self.arm_set, self.dot)

        #initialize X (represented using arm pull count) and y
        self.pulled_arms_count = np.zeros(self.k)
        self.y = np.zeros(self.k)
        for i, idx in enumerate(index_init):
            self.pulled_arms_count[idx] = self.pulled_arms_count[idx] + 1
            self.y[idx] = self.y[idx] + y_init[i]

        #initialize dict
        self.dict_arms_count = np.zeros(self.k)

        for idx in index_init:
            self.dict_arms_count[idx] = self.dict_arms_count[idx] + 1

        self.dict_arms_matrix = self.arm_set[self.dict_arms_count != 0, :]
        self._update_embedding()
        self._reset_posterior()
        self._update_beta()

    def predict(self):
        """Choose the candidate arm at time t

        Returns:
            chosen_arm_idx_l:  index of the arm chosen by BKB at time t, is a list containing an int
            ucbs: upper confidence bounds at time t, is a (A,) numpy.ndarray
        """
        ucbs = self.means + self.beta * np.sqrt(self.variances)
        assert np.all(np.isfinite(ucbs))
        chosen_arm_idx = np.argmax(ucbs)

        if self.verbose > 1:
            print(f'chosen {chosen_arm_idx} {ucbs[chosen_arm_idx]} {self.means[chosen_arm_idx]} {self.variances[chosen_arm_idx]}'
                  f'beta {self.beta} ucbs {self.means.max()} {self.means.min()} {self.variances.max()} {self.variances.min()}')
        chosen_arm_idx_l = [chosen_arm_idx]
        return chosen_arm_idx_l, ucbs

    def update(self, chosen_arm_idx_list, loss_list, random_state):
        """Properly update BKB given the new feedback

        Args:
            chosen_arm_idx_list: list containing the indexes of the arms chosen by BKB at time t
            loss_list: numpy.array containing the feedback corresponding to the chosen arms at time t
            random_state: np.random.RandomState used for reproducibility
        """
        self._update_pulled_arm_list_and_feedback(chosen_arm_idx_list, loss_list)

        self._resample_dict(random_state)
        self._update_embedding()
        self._reset_posterior()
        self._update_beta()

        return self

    # miscellaneous utils
    def _update_pulled_arm_list_and_feedback(self, chosen_arm_idx_list, loss_list):
        assert len(chosen_arm_idx_list) == len(loss_list)
        assert np.all(np.isfinite(loss_list))

        for i, chosen_arm_idx in enumerate(chosen_arm_idx_list):
            self.pulled_arms_count[chosen_arm_idx] = self.pulled_arms_count[chosen_arm_idx] + 1
            self.y[chosen_arm_idx] = self.y[chosen_arm_idx] + loss_list[i]
        self.t += len(chosen_arm_idx_list)

        assert self.t == self.pulled_arms_count.sum()

    def _resample_dict(self, random_state):
        resample_dict = random_state.rand(self.k) < (self.variances * self.pulled_arms_count * self.qbar)
        assert resample_dict.sum() > 0
        self.dict_arms_count = np.zeros(self.k)
        self.dict_arms_count[resample_dict] = 1
        self.dict_arms_matrix = self.arm_set[self.dict_arms_count != 0, :]

class Batch_BKB(BKB):
    """
    This version of the algorithm used the adaptive batching rule of LEMMA 1 in [2]

    [2] Near-linear time Gaussian process optimization with adaptive batching and resparsification
    ICML 2020 by D. Calandriello*, L. Carratino*, A. Lazaric, M. Valko, L. Rosasco.
    """
    def __init__(self, *args, ratio_threshold=2, **kwargs):
        """
        Args:
            ratio_threshold: positive float controlling the dimension of the batches (in the paper \tilda C in Algorithm 1).
            Larger values lead to larger batch-sizes, but can increase the overall regret of the algorithm.
        """
        super().__init__(*args, **kwargs)
        self.global_ratio_bound = 1.
        self.ratio_threshold = ratio_threshold


    def predict(self):
        """Choose the candidate arms in the current bach-size

        Returns:
            chosen_arm_idx_arr: batches of the b indexes of arms chosen by Batch BKB at time t, is a (b,) numpy.array of int
            ucbs: upper confidence bounds at time t, is a (A,) numpy.ndarray
        """
        tmp_variances = self.variances.copy()
        tmp_Q = self.Q.copy()
        tmp_R = self.R.copy()
        tmp_ucbs = self.means + self.beta * np.sqrt(tmp_variances)


        self.global_ratio_bound = 1.

        chosen_arm_idx_list = []

        while True:
            assert np.all(np.isfinite(tmp_ucbs))
            chosen_arm_idx = np.argmax(tmp_ucbs)
            chosen_arm_idx_list.append(chosen_arm_idx)
            chosen_arm = self.arm_set_embedded[chosen_arm_idx, :].reshape(1, -1)

            if self.verbose > 1:
                print(f'chosen {chosen_arm_idx} {tmp_ucbs[chosen_arm_idx]} {self.means[chosen_arm_idx]} {tmp_variances[chosen_arm_idx]}'
                      f'beta {self.beta} tmp_ucbs {self.means.max()} {self.means.min()} {tmp_variances.max()} {tmp_variances.min()}')

            tmp_Q, tmp_R = qr_update(tmp_Q, tmp_R, chosen_arm.squeeze(), chosen_arm.squeeze())

            variance_chosen_arm = np.dot(chosen_arm,
                                         solve_triangular(tmp_R, tmp_Q.T.dot(chosen_arm.T))).item()
            assert np.all(variance_chosen_arm >= 0.)
            assert np.all(np.isfinite(variance_chosen_arm))

            tmp_ucbs[chosen_arm_idx] = self.means[chosen_arm_idx] + self.beta * np.sqrt(variance_chosen_arm)
            candidates_argmax = (tmp_ucbs >= tmp_ucbs[chosen_arm_idx]).nonzero()[0]

            temp = solve_triangular(tmp_R,
                                    (self.arm_set_embedded[candidates_argmax, :].dot(tmp_Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
            temp *= self.arm_set_embedded[candidates_argmax, :]

            tmp_variances = ((self.arm_set_norms[candidates_argmax]
                              - self.arm_set_embedded_norms[candidates_argmax]) / self.lam
                             + np.sum(temp, axis=1))
            assert np.all(tmp_variances >= 0.)
            assert np.all(np.isfinite(tmp_variances))

            tmp_ucbs[candidates_argmax] = self.means[candidates_argmax] + self.beta * np.sqrt(tmp_variances)

            self.global_ratio_bound += self.variances[chosen_arm_idx].item()

            if self.global_ratio_bound > self.ratio_threshold:
                break

        chosen_arm_idx_arr = np.array(chosen_arm_idx_list)
        return chosen_arm_idx_arr, tmp_ucbs

    def update(self, chosen_arm_idx_list, loss_list, random_state):
        """Properly update Batch BKB given the new feedback

        Args:
            chosen_arm_idx_list: list containing the indexes of the arms chosen by Batch BKB in the current batch
            loss_list: numpy.array containing the feedback corresponding to the chosen arms at time t
            random_state: np.random.RandomState used for reproducibility
        """
        self._update_pulled_arm_list_and_feedback(chosen_arm_idx_list, loss_list)

        # we need to use the old variance estimate as overestimate when updating beta
        self.logdet = self.logdet + np.log(1 + self.variances[chosen_arm_idx_list]).sum()
        self.beta = (
                np.sqrt(self.lam) * self.fnorm
                + np.sqrt(self.noise_variance * (self.ratio_threshold * self.logdet + np.log(1 / self.delta)))
        )
        assert(np.isfinite(self.beta))

        self._resample_dict(random_state)
        self._update_embedding()
        self._reset_posterior()
        # no updating beta as we already did it

        return self

class Local_Batch_BKB(Batch_BKB):
    """
    This version of the  algorithm used the adaptive batching rule of LEMMA 3 in [2]

    [2] Near-linear time Gaussian process optimization with adaptive batching and resparsification
    ICML 2020 by D. Calandriello, L. Carratino, A. Lazaric, M. Valko, L. Rosasco.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_ratios_bound = np.ones(self.k)

    def predict(self):
        """Choose the candidate arms in the current bach-size

        Returns:
            chosen_arm_idx_arr:  batches of the b indexes of arms chosen by Batch BKB at time t, is a (b,) numpy.array of int
            ucbs: upper confidence bounds at time t, is a (A,) numpy.ndarray
        """
        tmp_variances = self.variances.copy()
        tmp_Q = self.Q.copy()
        tmp_R = self.R.copy()

        ucbs = self.means + self.beta * np.sqrt(tmp_variances)
        self.local_ratios_bound = np.ones(self.k)
        self.global_ratio_bound = 1.

        chosen_arm_idx_list = []

        switch_global_to_local = False
        while True:
            assert np.all(np.isfinite(ucbs))
            chosen_arm_idx = np.argmax(ucbs)
            chosen_arm_idx_list.append(chosen_arm_idx)
            chosen_arm = self.arm_set_embedded[chosen_arm_idx, :].reshape(1, -1)

            if self.verbose > 1:
                print(
                    f'chosen {chosen_arm_idx} {ucbs[chosen_arm_idx]} {self.means[chosen_arm_idx]} {tmp_variances[chosen_arm_idx]}'
                    f'beta {self.beta} ucbs {self.means.max()} {self.means.min()} {tmp_variances.max()} {tmp_variances.min()}')

            tmp_Q, tmp_R = qr_update(tmp_Q, tmp_R, chosen_arm.squeeze(), chosen_arm.squeeze())

            variance_chosen_arm = np.dot(chosen_arm,
                                         solve_triangular(tmp_R, tmp_Q.T.dot(chosen_arm.T))).item()
            assert np.all(variance_chosen_arm >= 0.)
            assert np.all(np.isfinite(variance_chosen_arm))

            ucbs[chosen_arm_idx] = self.means[chosen_arm_idx] + self.beta * np.sqrt(variance_chosen_arm)
            candidates_argmax = (ucbs >= ucbs[chosen_arm_idx]).nonzero()[0]

            temp = solve_triangular(tmp_R,
                                    (self.arm_set_embedded[candidates_argmax, :].dot(tmp_Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T

            temp *= self.arm_set_embedded[candidates_argmax, :]

            tmp_variances = ((self.arm_set_norms[candidates_argmax]
                              - self.arm_set_embedded_norms[candidates_argmax]) / self.lam
                             + np.sum(temp, axis=1))
            assert np.all(tmp_variances >= 0.)
            assert np.all(np.isfinite(tmp_variances))

            ucbs[candidates_argmax] = self.means[candidates_argmax] + self.beta * np.sqrt(tmp_variances)

            if not switch_global_to_local:
                self.global_ratio_bound += + self.variances[chosen_arm_idx].item()

                if self.global_ratio_bound > self.ratio_threshold:
                    switch_global_to_local = True

                    tmp_accumulator = np.zeros((self.k, len(chosen_arm_idx_list)))
                    # covars_until_switch
                    tmp_accumulator += np.dot(self.arm_set_embedded,
                                                     solve_triangular(self.R,
                                                          self.Q.T.dot(self.arm_set_embedded[chosen_arm_idx_list, :].T))
                                                    )

                    # residuals_until_switch
                    tmp_accumulator += (self.dot(self.arm_set, self.arm_set[chosen_arm_idx_list, :])
                                        - self.arm_set_embedded.dot(self.arm_set_embedded[chosen_arm_idx_list, :].T)
                                       ) / self.lam

                    np.square(tmp_accumulator, out=tmp_accumulator)

                    self.local_ratios_bound += tmp_accumulator.sum(axis=1) / self.variances
                    assert(np.all(np.isfinite(self.local_ratios_bound)))
            else:
                covars_chosen_arm = np.dot(self.arm_set_embedded,
                                           solve_triangular(self.R,
                                                            self.Q.T.dot(chosen_arm.T))
                                           )

                residual_chosen_arm = (self.dot(self.arm_set, self.arm_set[[chosen_arm_idx], :])
                                       - self.arm_set_embedded.dot(self.arm_set_embedded[[chosen_arm_idx], :].T)) / self.lam

                self.local_ratios_bound += np.square(covars_chosen_arm + residual_chosen_arm).reshape(-1)/self.variances
                assert(np.all(np.isfinite(self.local_ratios_bound)))

                if self.local_ratios_bound.max() > self.ratio_threshold:
                    break

        chosen_arm_idx_arr = np.array(chosen_arm_idx_list)
        return chosen_arm_idx_arr, ucbs


