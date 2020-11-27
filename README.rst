Penalised Least Square Smooth
=============================

.. _fda: https://cran.r-project.org/web/packages/fda/

`plssmooth` python package implements **P** enalised **L**\ east **S**\ quare **SMOOTH**\ing methodology. Such methodology
is common in functional data analysis and as such this package is heavily dependent on monographs such as, [1].
Similarly it is heavily influenced by the fda_ package in the R language.

What does plssmooth do?
-----------------------

`plssmooth` implements penalised spline smoothing for observed data to estimate the set of coefficients in a basis
expansion of the data in a known basis system. In fact the penalised least square estimator for the coefficients
:math:`\hat{\mathbf{c}}` is chosen by minimising:

.. math::
    \sum_{i=1}^n (y_i - B^\top(t) \mathbf{c})^2 + \lambda \mathbf{c}^\top P \mathbf{c}

where :math:`P` is a penalty matrix of the basis system, :math:`B^\top(t)`.


.. [1] J. O. Ramsay and B. W. Silverman, Functional data analysis. New York (N.Y.): Springer Science+Business Media, 2010.
