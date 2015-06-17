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


def locateLS2d(phi, norm_x, norm_y, dx=1., dy=1.):
    r"""
    Locates level set in 2d

    :Arguments:
        - `\phi`:            :math:`\phi` - 2d level set function
        - `norm_*`:          components of unit normal vector
        - 'dx, dy':          grid spacing

    :Returns:
        - `zeros`:           2d boolean array locating level set
        - `alpha`:           2d float array containing burnt fraction
    """
    zeros = np.zeros_like(phi, dtype=bool)
    alpha = np.zeros_like(phi)

    # normalise norms some more
    normMag = np.sqrt(norm_x[:]**2 + norm_y[:]**2)
    norm_x[:] /=normMag[:]
    norm_y[:] /=normMag[:]

    # if modulus of signed distance function is less than
    # 0.5 * sqrt([n_x*dx]^2 + [n_y*dy]^2), cell contains zero level set
    zeros[:] = (np.abs(phi[:]) < 0.5*(np.abs(norm_x[:])*dx + np.abs(norm_y[:])*dy) / \
                    np.sqrt(norm_x[:]**2 + norm_y[:]**2))

    #print(0.5*(np.abs(norm_x[:])*dx + np.abs(norm_y[:])*dy) / \
    #                np.sqrt(norm_x[:]**2 + norm_y[:]**2))

    # burnt cells
    alpha[phi[:] > 0.] = 1.

    # correct cells cut by zero level set
    alpha[zeros] = 0.5 + phi[zeros] / \
            ((np.abs(norm_x[zeros])*dx + np.abs(norm_y[zeros])*dy) / \
                            np.sqrt(norm_x[zeros]**2 + norm_y[zeros]**2))

    return zeros, alpha
