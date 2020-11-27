import numpy as np


def pls_smooth(Y, B, P, omega=0.0, method=None):
    """Calculate the coefficients from a penalised least squares smooth of the data with basis matrix B with penalty P.

    Args:
        Y (np.ndarray): Two dimensional observation array with subjects in first dimension and observations per subject
        in second dimension.

        B (np.ndarray): Basis matrix evaluated at observation points. First dimension corresponds to observation points.
        Second dimension is the number of basis functions in the expansion.

        P (np.ndarray): Penalty matrix to use for regularisation term.

        omega (float, Optional): Regularisation parameter to use. Will be overriden if `method` parameter is not `fixed`
        . Defualts to 0.

        method ({'fixed', 'gcv', 'rgcv'}, Optional): Method to choose omega in the calculation of the smooth:

            * 'fixed' will keep `omega` fixed as given. Default option.
            * 'gcv' will use generalised cross validation methodology to choose `omega`.
            * `rgcv` will use robust generalised cross validation to choose `omega`.


    Returns:
        C (np.ndarray): Matrix of estimated coefficients based on the penalised spline smooth fit.
        omega (np.ndarray): The omega value used in the smoothing methodology.

    """
    pass
