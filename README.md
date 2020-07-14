# Batch-BKB: no-regret scalable GP optimization
`batch-bkb` is the first Bayesian optimization (a.k.a. Gaussian process or bandit optimization) algorithm that is both provably no-regret and guaranteed to run in near-linear time time.

This repository contains an implementation of the algorithm as described in the ICML 2020 paper
["Near-linear time Gaussian process optimization with adaptive batching and resparsification"](https://arxiv.org/abs/2002.09954)
by [Calandriello Daniele](https://scholar.google.com/citations?user=R7c1UMMAAAAJ),
[Luigi Carratino](https://luigicarratino.com/), [Alessandro Lazaric](https://scholar.google.com/citations?user=6JZ3R6wAAAAJ&hl=en), [Michal Valko](http://researchers.lille.inria.fr/~valko/hp/) and [Lorenzo Rosasco](https://rubrica.unige.it/personale/UkNHXVxs),

This repository also contains an implementation of `bkb` as described in the COLT 2019 paper
["Gaussian Process Optimization with Adaptive Sketching: Scalable and No Regret"](https://arxiv.org/abs/1903.05594)
by [Calandriello Daniele](https://scholar.google.com/citations?user=R7c1UMMAAAAJ),
[Luigi Carratino](https://luigicarratino.com/), [Alessandro Lazaric](https://scholar.google.com/citations?user=6JZ3R6wAAAAJ&hl=en), [Michal Valko](http://researchers.lille.inria.fr/~valko/hp/) and [Lorenzo Rosasco](https://rubrica.unige.it/personale/UkNHXVxs)

## Resources

|Link | Resource|
|---|---|
| [ArXiv](https://arxiv.org/abs/2002.09954) | Paper
| [Poster](https://github.com/luigicarratino/batch-bkb/blob/master/poster.pdf) | Poster

## Dependencies
Our code requires access to a scientific python stack, including `numpy`, `scipy`, and `sklearn`

## Algorithm usage
An example usage can be find in the file `example_batch_bkb.py` and `example_bkb.py`


## Citation
If you use `batch-bkb` or the related experiments code please cite:
```
@incollection{icml2020bbkb,
    title = {Near-linear time Gaussian process optimization with adaptive batching and resparsification},
    author = {Daniele Calandriello and Luigi Carratino and Alessandro Lazaric and Michal Valko and Lorenzo Rosasco},
    booktitle = {Proceedings of the 37th International Conference on Machine Learning},
    year = {2020},
}

@InProceedings{pmlr-v99-calandriello19a,
  title = 	 {Gaussian Process Optimization with Adaptive Sketching: Scalable and No Regret},
  author = 	 {Calandriello, Daniele and Carratino, Luigi and Lazaric, Alessandro and Valko, Michal and Rosasco, Lorenzo},
  booktitle = 	 {Proceedings of the Thirty-Second Conference on Learning Theory},
  pages = 	 {533--557},
  year = 	 {2019},
  editor = 	 {Beygelzimer, Alina and Hsu, Daniel},
  volume = 	 {99},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--28 Jun},
  publisher = 	 {PMLR},
}
```

## Contact
For any question, you can contact daniele.calandriello@iit.it or luigi.carratino@dibris.unige.it
