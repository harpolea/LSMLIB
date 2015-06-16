import numpy as np

__docformat__ = 'restructuredtext'

def locateLS1d(phi, dx=1.):
    r"""
    Locates level set in 1d

    :Arguments:
        - `\phi`:            :math:`\phi` - 1d level set function
        - 'dx':              grid spacing

    :Returns:
        - `zeros`:           1d boolean array locating level set
        - `alpha`:           1d float array containing burnt fraction
    """

    zeros = np.zeros_like(phi, dtype=bool)
    alpha = np.zeros_like(phi)

    # if modulus of signed distance function is less than 0.5*dx,
    # cell contains zero level set
    zeros[:] = (np.abs(phi[:]) < 0.5 * dx)

    # burnt cells
    alpha[phi[:] > 0.] = 1.

    # correct cells cut by zero level set
    alpha[zeros] = 0.5 + phi[zeros]/dx

    return zeros, alpha


def locateLS2d(phi, dx=1., dy=1.):
    r"""
    Locates level set in 2d

    :Arguments:
        - `\phi`:            :math:`\phi` - 2d level set function
        - 'dx, dy':              grid spacing

    :Returns:
        - `zeros`:           2d boolean array locating level set
        - `alpha`:           2d float array containing burnt fraction
    """
    zeros = np.zeros_like(phi, dtype=bool)
    alpha = np.zeros_like(phi)

    # if modulus of signed distance function is less than
    # 0.5 * sqrt(dx^2 + dy^2), cell contains zero level set
    zeros[:] = (np.abs(phi[:]) < 0.5 * np.sqrt(dx**2 + dy**2))

    # burnt cells
    alpha[phi[:] > 0.] = 1.

    # correct cells cut by zero level set
    alpha[zeros] = 0.5 + phi[zeros]/np.sqrt(dx**2 + dy**2)

    return zeros, alpha
