# README.md
- Working notes on Scikit-Learn v0.23
- Implemented in Jupyter Lab notebooks
- Initial preliminary release 2020/06/11
- In progress.
- Brian Piercy, @brianpiercy, bjpcjp at gmail dot com

### **Getting Started**
- [Getting Started](getting-started.ipynb)

### **Clustering**
- [Affinity Propagation](clustering-affinity-propagation.ipynb)
    + About, Example
- [BiClustering](clustering-biclustering.ipynb)
    + About, Spectral CoClusters, 20newsgroups, Spectral BiClusters, Metrics
- [Birch](clustering-birch.ipynb)
    + About, vs Kmeans minibatch
- [DBSCAN](clustering-dbscan.ipynb)
    + [API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN), Examples
- [Hierarchical (Agglomerative)](clustering-hierarchical.ipynb)
    + [API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)
- [K-means & Voronoi Diagrams](clustering-kmeans.ipynb)
    + [API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans), Background, Voronoi Diagrams, Minibatch
- [Mean Shift](clustering-mean-shift.ipynb)
    + [API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift)
- [Optics](clustering-optics.ipynb)
    + [API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS)
- [Spectral](clustering-spectral.ipynb)
    + [API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering), Image Segmentation, Coins

### **Covariance (CV)**
- [Empirical aka MLE, Shrunk](covariance.ipynb)
    + Empirical
    + Shrunk
    + Optimal Shrinkages (Ledoit-Wolf, OAS)
    + Comparison
    + Sparse Inverse CV (= "precision matrix")
    + Robust CV Estimation - FastMCD
    + Mahalanobis Distance

### **Decision Trees**
- [Classification, Regression, Multiple Outputs](decision-trees.ipynb)

### **Decomposition**
- [Partial Least Squares, Canonical Correlation Analysis](cross-decomposition.ipynb)
- [Dictionary Learning](decomposition-dictionary-learning.ipynb)
    + Sparse Dicts
    + Generic, Images, MiniBatch
- [Factor Analysis](decomposition-FA.ipynb)
    + Model Selection
- [Independent Component Analysis (ICA)](decomposition-ICA.ipynb)
    + Noise source separation
- [Latent Dirichlet Allocation (LDA)](decomposition-LDA.ipynb)
    + About
    + Topic extraction
- [Non-Negative Matrix Factorization (NNMF)](decomposition-NNMF.ipynb)
    + About
    + Topic extraction
    + Beta-convergence loss functions (sqd. Frobenius, Kullback-Leibler, Itakura-Saito)
- [Principal Component Analysis (PCA)](decomposition-PCA.ipynb)
    + LDA vs PCA
    + Incremental
    + w/ randomized SVD
    + Kernels (extend PCA to non-linears)
    + Sparse

### **Density Estimation (DE)**
- [Histograms, Kernels, Kernel DE KDE/digits](density-estimation.ipynb)

### **Disciminant Analysis**
- [Linear (LDA), Quadratic (QDA)](discriminant-analysis-LDA-QDA.ipynb)

### **Ensembles**
- [AdaBoost Classifier/Regressor](ensembles-adaboost.ipynb)
    + Example, DT w/ Adaboost + Multiclass Outs
- [Bagging Classifier/Regressor](ensembles-bagging.ipynb)
    + Bagged vs single Estimator
- [Gradient Boosting (Stochastic-, Histogram-)](ensembles-gradient-boosting.ipynb)
    + Example
    + Histogram-based (Classifiers, Regressors)
- [Gradient Boosting](ensembles-gradient-tree-boosting.ipynb)
    + Example 
    + Histogram-based
- [Random Forest Classifiers/Regressors](ensembles-random-forests.ipynb)
    + Example
    + Extremely Randomized Trees
    + Decision Surfaces
    + Pixel Importances
    + Random Tree Embedding
    + Hashing Feature Transforms -- Totally Random Trees
- [Stacking Classifiers/Regressors](ensembles-stacking.ipynb)
    + Example
- [Voting Classifiers/Regressors](ensembles-voting.ipynb)
    + About
    + Example Soft
    + Regressor

### **Feature Engineering**
- [Selection](feature-selection.ipynb)
    + Low-Variance Removal
    + Select: K best, Percentile, FalsePR, FalseDR, Family-wise, Configurable
    + Recursive
    + Select From Model
    + L1-Based
    + Tree-Based

- [Transformers](feature-transformer.ipynb)
    + Column-Wise (Feature-wise)
    + HTML Viz / Jupyter notebooks

### **Gaussian Models**
- [Gaussian Models](gaussian-models.ipynb)
    + Regression (GPR)
    + GPR vs KRR, Moana Loa CO2
    + Classification (GPC)
    + GPC vs Dot-Product Kernels
    + [Kernels](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Kernel.html#sklearn.gaussian_process.kernels.Kernel) (aka CV functions)
- [Gaussian Mixture Models (GMMs)](gaussian-mixture-models.ipynb)
    + About
    + Example
    + Example: Model Selection
    + Expectation Maximization (EM)
    + Variational Bayes GMMs

- [Extraction - Dicts](transforms-feature-extraction-dicts.ipynb)
- [Extraction - Images](transforms-feature-extraction-images.ipynb)
- [Extraction- Text](transforms-feature-extraction-text.ipynb)
- [Hashing](transforms-feature-hashing.ipynb)
- [Pipelines](transforms-feature-pipelines.ipynb)
- [Unions](transforms-feature-unions.ipynb)
- [Imputation](transforms-imputation.ipynb)
- [Kernel Approximation](transforms-kernel-approximations.ipynb)
- [Pairwise Utilities](transforms-pairwise-utilities.ipynb)
- [Preprocessing](transforms-preprocessing.ipynb)
- [Random Projections](transforms-random-projection.ipynb)

### **Gradient Descent**
- [Stochastic GD](gradient-descent-SGD.ipynb)

### **Kernels**
- [Kernels (Wikipedia)](https://en.wikipedia.org/wiki/Kernel_method)

### **Label Spreading**
- [About](label-spreading.ipynb)

### **Logistic Regression**
- [Logistic Regression](logistic-regression.ipynb)

### **Metrics**
- [Classification](metrics-classification.ipynb)
- [Clustering](metrics-clustering.ipynb)
- [Cross Validation](metrics-cross-validation.ipynb)
- [Dummy Estimators](metrics-dummy-estimatores.ipynb)
- [Multilabel Rankers](metrics-multilabel-ranking.ipynb)
- [Regression](metrics-regression.ipynb)

### **Regression**
- [Bayes](regression-Bayes.ipynb)
- [Elastic Net](regression-elastic-net.ipynb)
- [Isotonic](regression-isotonic.ipynb)
- [Kernel Ridge](regression-kernel-ridge.ipynb)
- [LARS](regression-LARS.ipynb)
- [Lasso](regression-Lasso.ipynb)
- [OLS](regression-OLS.ipynb)
- [OMP](regression-OMP.ipynb)
- [Polynomial](regression-polynomial.ipynb)
- [RANSAC, Theil-Sen, Huber](regression-RANSAC-Theil-Sen-Huber.ipynb)
- [Ridge](regression-ridge.ipynb)
- [Tweedie](regression-tweedie.ipynb)

### **Transforms**
