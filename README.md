# README.md
- Working notes on Scikit-Learn v0.23
- Implemented in Jupyter Lab notebooks
- Initial release: 2020/06/11
- This snapshot: 2020/07/28
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
- [About](covariance.ipynb)
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
- [Cross Decomposition](cross-decomposition.ipynb)
    + Partial Least Squares (PLS)
    + Canonical Correlation Analysis (CCA)
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

* [Extraction - Dicts](transforms-feature-extraction-dicts.ipynb)
* [Extraction - Images](transforms-feature-extraction-images.ipynb)
* [Extraction- Text](transforms-feature-extraction-text.ipynb)
* [Hashing](transforms-feature-hashing.ipynb)
* [Pipelines](transforms-feature-pipelines.ipynb)
* [Unions](transforms-feature-unions.ipynb)
* [Imputation](transforms-imputation.ipynb)
* [Kernel Approximation](transforms-kernel-approximations.ipynb)
* [Pairwise Utilities](transforms-pairwise-utilities.ipynb)
* [Preprocessing](transforms-preprocessing.ipynb)
* [Random Projections](transforms-random-projection.ipynb)

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

### **Gradient Descent**
- [Stochastic GD](gradient-descent-SGD.ipynb)

### **Kernels**
- [Kernels (Wikipedia)](https://en.wikipedia.org/wiki/Kernel_method)

### **Semi-Supervised Learning**
- [Label Propagation](label-spreading.ipynb)

### **Logistic Regression**
- [Logistic Regression](logistic-regression.ipynb)
    + About; solvers (liblinear, newton-cg, lbfgs, sag, saga)
    + L1 Penalty vs Sparsity
    + Multinomial & OvR LR

### **Metrics**
- [Classification](metrics-classification.ipynb)
    + Accuracy, Balanced Accuracy, Cohen's, Confusion Matrix, Classification Report, Hamming Loss, Precision, Recall, f-measure, Jaccard Similarity, Hinge Loss, Log Loss, Matthews Coeff, Multilabel Confusion Matrix, Receiver Operating Characteristic (ROC), ROC-AUC, Zero-One, Brier
- [Clustering](metrics-clustering.ipynb)
    + Adjusted Rand Index (ARI), Mutual Info Scores, Homogeneity, Completeness, V-Measure, Silhouette Coefficient, Calinski-Harabasz, Davies-Bouldin, Contingency Matrix
- [Cross Validation](metrics-cross-validation.ipynb)
    + Scoring
    + IID Data Splits
        * K-fold/Repeated/Stratified/Grouped
        * Leave (One/P/One Group/P Groups) Out
        * Shuffle Split/Stratified/Grouped
        * Predefined
        * Time Series
        * Notes / Best practices
- [Display Objects](metrics-display-objects.ipynb)
- [Dummy Estimators](metrics-dummy-estimators.ipynb)
    + Regressors & Classifiers
- [Multilabel Rankers](metrics-multilabel-ranking.ipynb)
    + Metrics
    + Coverage Error, Average Precision, Ranking Loss, NDCG
- [Regression](metrics-regression.ipynb)
    + Metrics
    + Explained Variance, Max Error, MAE, MSE, MSLE, MedAE, R^2, Tweedie Deviance

### **Multiclass & Multilabel problems**
- [for personal experiments](multiclass-multilabel.ipynb) (SL natively supports multiclass ops)

### **Linear Problems**
- [Bayes](regression-Bayes.ipynb)
    + About, Example, Example (Synth Dataset), Ex: Sinusoid Curve Fitting, Auto Relevance Determination (ARD)
- [Elastic Net (EN)](regression-elastic-net.ipynb)
    + About, Example, Multi-Task EN
- [Isotonic](regression-isotonic.ipynb)
    + About, Example
- [Kernel Ridge](regression-kernel-ridge.ipynb)
    + About, KRR vs SVR
- [Least-Angle Regression (LARS)](regression-LARS.ipynb)
    + About, Lasso LARS
- [Lasso](regression-Lasso.ipynb)
    + About, ex: Tomography Reconstruction, Multi-Regression
- [Ordinary Least Squares (OLS)](regression-OLS.ipynb)
    + About, Example
- [Orthogonal Matching Pursuit (OMP)](regression-OMP.ipynb)
    + About, Example
- [Polynomial Regression / Basis Functions](regression-polynomial.ipynb)
    + About, Example, Example: Function Approximation
- [Regression Robustness (outliers)](regression-RANSAC-Theil-Sen-Huber.ipynb)
    + About, RANSAC, Theil-Sen, Huber
    + Example: Sinusoid Curve Fitting
- [Ridge (Regression, Classification)](regression-Ridge.ipynb)
    + About, Example, Built-in CV
- [Tweedie](regression-tweedie.ipynb)
    + About
    + Example (French auto liability)
