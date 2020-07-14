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
from bkb_lib import BKB
from sklearn.gaussian_process.kernels import PairwiseKernel
import matplotlib.pyplot as plt

r_state = np.random.RandomState(seed=42)

d = 3
k = 1000
T = 2000
dot_func = PairwiseKernel(metric='linear')
w_star = r_state.randn(d).reshape(1,-1)
arms = r_state.randn(k, d)
arms_score = dot_func(arms, w_star.reshape(1,-1))
best_arm = np.argmax(arms_score)

noise_ratio = 0.01
noise_std = np.sqrt((arms_score.max() - arms_score.min()) * noise_ratio)
f = lambda x: dot_func(x, w_star) + r_state.randn() * noise_std

bkb_alg = BKB(
    lam=noise_std ** 2.,
    dot=dot_func,
    noise_variance=noise_std ** 2.,
    fnorm=1.0,
    delta=0.5,
    qbar=1,
    verbose=0
)

instant_regret = np.zeros(T)
cum_regret = 0


bkb_alg.initialize(arms, [0,1], np.array([f(arms[0, :]), f(arms[1, :])]).reshape(-1))

#run it
cumulative_regret_over_time = np.zeros((T,))
t = 0
while t < T:
    chosen_arms_idx, ucbs = bkb_alg.predict()
    if t + len(chosen_arms_idx) > T:
        chosen_arms_idx = chosen_arms_idx[:T - t]

    #batch
    feedback_list = []
    for i in range(len(chosen_arms_idx)):
        arm_idx = chosen_arms_idx[i]
        feedback_list.append(f(arms[arm_idx,:]))
        instant_regret[t] = arms_score[best_arm] - arms_score[arm_idx]
        cum_regret = cum_regret + instant_regret[t]
        cumulative_regret_over_time[t] = cum_regret
        t = t + 1

    bkb_alg.update(chosen_arms_idx, np.array(feedback_list), r_state)

plt.plot(np.arange(T), cumulative_regret_over_time)
plt.xlabel('t')
plt.ylabel('cumulative regret')
plt.show()
