import numpy as np

__docformat__ = 'restructuredtext'

def locateLS1d(phi, dx=1.):
    r"""
    Locates level set in 1d. A cell contains the level set if the signed distance function satisfies

    .. math::

        |\phi| < \frac{\Delta x}{2}.

    :Arguments:
        - `\phi`:            1d level set function. MUST be a signed distance function.
        - `dx`:              grid spacing

    :Returns:
        - `zeros`:           1d boolean array locating level set
        - `alpha`:           1d float array containing burnt fraction
    """

    # check input is 1d
    if len(phi.shape) != 1:
        raise ValueError, "phi must be 1d"

    #initialise
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
    Locates level set in 2d. Cell contains zero level set if the signed distance function satisfies

    .. math::

        |\phi| < \frac{|n_x|\Delta x + |n_y|\Delta y}{2\sqrt{n_x^2 + n_y^2}},

    where :math:`n_x`, :math:`n_y` are components of the unit normal to the level set.

    :Arguments:
        - `\phi`:            2d level set function. MUST be a signed distance function.
        - `norm_*`:          components of unit normal vector
        - `dx`, `dy`:          grid spacing

    :Returns:
        - `zeros`:           2d boolean array locating level set
        - `alpha`:           2d float array containing burnt fraction
    """
    # check input is 2d
    #if len(phi.shape) != 2:
    #    raise ValueError, "phi must be 2d"

    #initialise
    zeros = np.zeros_like(phi, dtype=bool)
    alpha = np.zeros_like(phi)

    # test to see if cells contain zero level set
    zeros[:] = (np.abs(phi[:]) < 0.5*(np.abs(norm_x[:])*dx + np.abs(norm_y[:])*dy) / \
                    np.sqrt(norm_x[:]**2 + norm_y[:]**2))

    # burnt cells
    alpha[phi[:] > 0.] = 1.

    # correct cells cut by zero level set
    alpha[zeros] = 0.5 + phi[zeros] / \
            ((np.abs(norm_x[zeros])*dx + np.abs(norm_y[zeros])*dy) / \
                            np.sqrt(norm_x[zeros]**2 + norm_y[zeros]**2))

    return zeros, alpha


def locateLS3d(phi, norm_x, norm_y, norm_z, dx=1., dy=1., dz=1.):
    r"""
    Locates level set in 3d. Cell contains zero level set if the signed distance function satisfies

    .. math::

        |\phi| < \frac{|n_x|\Delta x + |n_y|\Delta y + |n_z|\Delta z}{2\sqrt{n_x^2 + n_y^2 + n_z^2}},

    where :math:`n_x`, :math:`n_y`, :math:`n_z` are components of the unit normal to the level set. Have assumed that the 2d method extends simply to the 3d but **yet to check the maths**.

    :Arguments:
        - `\phi`:            3d level set function. MUST be a signed distance function.
        - `norm_*`:          components of unit normal vector
        - `dx`, `dy`, `dz`:          grid spacing

    :Returns:
        - `zeros`:           3d boolean array locating level set
        - `alpha`:           3d float array containing burnt fraction
    """
    # check input is 3d
    #if len(phi.shape) != 3:
        #raise ValueError, "phi must be 3d"

    #initialise
    zeros = np.zeros_like(phi, dtype=bool)
    alpha = np.zeros_like(phi)

    # test to see if cells contain zero level set
    zeros[:] = (np.abs(phi[:]) < 0.5*(np.abs(norm_x[:])*dx + np.abs(norm_y[:])*dy + np.abs(norm_z[:])*dz) / \
                    np.sqrt(norm_x[:]**2 + norm_y[:]**2 + norm_z[:]**2))

    # burnt cells
    alpha[phi[:] > 0.] = 1.

    # correct cells cut by zero level set
    alpha[zeros] = 0.5 + phi[zeros] / \
            ((np.abs(norm_x[zeros])*dx + np.abs(norm_y[zeros])*dy + \
             np.abs(norm_z[zeros])*dz) / np.sqrt(norm_x[zeros]**2 + \
             norm_y[zeros]**2 + norm_z[zeros]**2))

    return zeros, alpha
