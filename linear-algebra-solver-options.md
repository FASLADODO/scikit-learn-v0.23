# Scikit-Learn solver options

- **'auto'** - chosen by algorithm based on datatype.
- **'svd'** - singular value decomposition.
- **'cholesky'** - [scipy.linalg.solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html) to find a closed-form answer.
- **'sparse_cg'** - conjugate gradient solver [(scipy.sparse.linalg.cg)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)
- **'lsqr'** - regularized least squares [(scipy.sparse.linalg.lsqr)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html). fastest & iterative.
- **'sag'** - stochastic average gradient descent. Iterative & often faster when #samples, #features are large.
- **'saga'** - unbiased version of sag.

