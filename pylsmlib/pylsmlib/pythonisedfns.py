import numpy as np

__docformat__ = 'restructuredtext'

def gradPhi(phi0, dx=1., dy=1., dz=1., order=2):
    r"""
    Computes the fourth-order, central,
    finite difference approximation to the gradient of :math:`\phi`

    :Arguments:
        - `\phi`:         :math:`\phi`
        - `dx`, `dy`, `dz`:  grid cell size

    :Returns:
        - `phi_*`:      components of  :math:`\nabla \phi`
    """

    phi_z, phi_y, phi_x = np.gradient(phi, dx, dy, dx)

    return phi_x, phi_y, phi_z


def surfaceAreaLevelSet(phi0, phi_x, phi_y, phi_z, ibLims,
                            dx=1., dy=1., dz=1.,
                            epsilon=1., order=2):

    r"""
    Computes the surface area of the surface defined by the zero level set.

     :Parameters:

      - `phi0`:              level set function
      - `phi_*` :            components of :math:`\nabla \phi`
      - `dx`:       grid spacing
      - `epsilon`:          width of numerical smoothing to use for Heaviside function
      - `*_ib`:             index range for interior box

     :Returns:

        - `area`:            area of the surface defined by the zero level                               set

    """

    area = 0.

    dV = dx * dy * dz

    for k in range(ibLims[4], ibLims[5]):
        for j in range(ibLims[2], ibLims[3]):
            for i in range(ibLims[0], ibLims[1]):

                if np.abs(phi[i,j,k]) < epsilon:
                    delta = 0.5 * (1. + np.cos(np.pi * phi[i,j,k] /epsilon))/epsilon

                    area+= delta * dV * np.sqrt(phi_x[i,j,k]**2 + phi_y[i,j,k]**2 + phi_z[i,j,k]**2)

    return area


def meanCurvature(phi, phi_x, phi_y, phi_z, fbLims,
                            dx=1., dy=1., dz=1., zero_tol=1.e-11, order=2):
    r"""

      Computes mean curvature

      .. math::

        \kappa = ( \phi_{xx}\phi_y^2 + \phi_{yy}\phi_x^2 - 2\phi_{xy}\phi_x\phi_y +
                    \phi_{xx}\phi_z^2 + \phi_{zz}\phi_x^2 - 2\phi_{xz}\phi_x\phi_z +\\
                    \phi_{yy}\phi_z^2 + \phi_{zz}\phi_y^2 - 2\phi_{yz}\phi_y\phi_z )
                  ( | \nabla \phi | ^ 3 )

      Standard centered 27 point stencil, second order differencing used.
      First order derivatives assumed precomputed.

      :Parameters:

        - `phi`:  level set function
        - `phi_*`:  first order derivatives of :math:`\phi`
        - `*_fb`:   index range for fillbox
        - `dx`:   grid spacing

    :Returns:

        - `kappa`: curvature data array

    """

    #initialise
    kappa = np.zeros_like(phi)

    phi_xx = np.zeros_like(phi)
    phi_yy= np.zeros_like(phi)
    phi_zz = np.zeros_like(phi)
    phi_xy = np.zeros_like(phi)
    phi_yz = np.zeros_like(phi)
    phi_zx = np.zeros_like(phi)

    ilo, ihi, jlo, jhi, klo, khi = fbLims
    ihi+=1
    jhi+=1
    khi+=1

    phi_xx[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi[ilo+2:ihi+2,jlo:jhi,klo:khi] + \
        16.*phi[ilo+1:ihi+1,jlo:jhi,klo:khi] -\
        30.*phi[ilo:ihi,jlo:jhi,klo:khi] -\
        phi[ilo-2:ihi-2,jlo:jhi,klo:khi] +\
        16.*phi[ilo-1:ihi-1,jlo:jhi,klo:khi]) / \
        (12 * dx**2)

    phi_yy[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi[ilo:ihi,jlo+2:jhi+2,klo:khi] + \
        16.*phi[ilo:ihi,jlo+1:jhi+1,klo:khi] -\
        30.*phi[ilo:ihi,jlo:jhi,klo:khi] -\
        phi[ilo:ihi,jlo-2:jhi-2,klo:khi] +\
        16.*phi[ilo:ihi,jlo-1:jhi-1,klo:khi]) / \
        (12 * dy**2)

    phi_zz[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi[ilo:ihi,jlo:jhi,klo+2:khi+2] + \
        16.*phi[ilo:ihi,jlo:jhi,klo+1:khi+1] -\
        30.*phi[ilo:ihi,jlo:jhi,klo:khi] -\
        phi[ilo:ihi,jlo:jhi,klo-2:khi-2] +\
        16.*phi[ilo:ihi,jlo:jhi,klo-1:khi-1]) / \
        (12 * dz**2)

    phi_xy[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi_x[ilo:ihi,jlo+2:jhi+2,klo:khi] + \
        8.*phi_x[ilo:ihi,jlo+1:jhi+1,klo:khi] +\
        phi_x[ilo:ihi,jlo-2:jhi-2,klo:khi] -\
        8.*phi_x[ilo:ihi,jlo-1:jhi-1,klo:khi]) / \
        (12 * dy)

    phi_yz[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi_y[ilo:ihi,jlo:jhi,klo+2:khi+2] + \
        8.*phi_y[ilo:ihi,jlo:jhi,klo+1:khi+1] +\
        phi_y[ilo:ihi,jlo:jhi,klo-2:khi-2] -\
        8.*phi_y[ilo:ihi,jlo:jhi,klo-1:khi-1]) / \
        (12 * dz)

    phi_zx[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi_z[ilo+2:ihi+2,jlo:jhi,klo:khi] + \
        8.*phi_z[ilo+1:ihi+1,jlo:jhi,klo:khi] +\
        phi_z[ilo-2:ihi-2,jlo:jhi,klo:khi] -\
        8.*phi_z[ilo-1:ihi-1,jlo:jhi,klo:khi]) / \
        (12 * dx)

    denominator = sqrt(phi_x[:]**2 + phi_y[:]**2 + phi_z[:]**2) ** 3

    kappa[ilo:ihi,jlo:jhi,klo:khi] = phi_xx[ilo:ihi,jlo:jhi,klo:khi] * \
        phi_y[ilo:ihi,jlo:jhi,klo:khi]**2 + \
        phi_yy[ilo:ihi,jlo:jhi,klo:khi]*phi_x[ilo:ihi,jlo:jhi,klo:khi]**2 -\
        2.* phi_xy[ilo:ihi,jlo:jhi,klo:khi] * phi_x[ilo:ihi,jlo:jhi,klo:khi]*\
        phi_y[ilo:ihi,jlo:jhi,klo:khi] + \
        phi_xx[ilo:ihi,jlo:jhi,klo:khi]*phi_z[ilo:ihi,jlo:jhi,klo:khi]**2 +\
        phi_zz[ilo:ihi,jlo:jhi,klo:khi]*phi_x[ilo:ihi,jlo:jhi,klo:khi]**2 -\
        2.* phi_zx[ilo:ihi,jlo:jhi,klo:khi] * phi_x[ilo:ihi,jlo:jhi,klo:khi]*\
        phi_z[ilo:ihi,jlo:jhi,klo:khi] +\
        phi_yy[ilo:ihi,jlo:jhi,klo:khi]*phi_z[ilo:ihi,jlo:jhi,klo:khi]**2 +\
        phi_zz[ilo:ihi,jlo:jhi,klo:khi]*phi_y[ilo:ihi,jlo:jhi,klo:khi]**2 -\
        2.* phi_yz[ilo:ihi,jlo:jhi,klo:khi]*phi_y[ilo:ihi,jlo:jhi,klo:khi] *\
        phi_z[ilo:ihi,jlo:jhi,klo:khi]

    kappa[denominator < zero_tol] = 0.
    kappa[denominator >= zero_tol] /= denominator[denominator >= zero_tol]

    return kappa


def signedUnitNormal(phi, phi_x, phi_y, phi_z,
                        dx=1., dy=1., dz=1., zero_tol=1.e-11):
    r"""
      Computes the signed unit normal
      vector (sgn(phi)*normal) to the interface from :math:`\nabla \phi`
      using the following smoothed sgn function

      .. math::

        sgn(\phi) = \phi / \sqrt{ \phi^2 + |\nabla \phi|^2 * dx^2 }


      :Parameters:

       - `phi_*`:         components of :math:`\nabla \phi`
       - `phi`:           level set function
       - `dx`:    grid spacing

      :Returns:

        - `normal_*`:     components of unit normal vector

      :Notes:

        - When :math:`| \nabla \phi |`  is close to zero, the unit normal is arbitrarily set to be :math:`(1.0, 0.0, 0.0)`.
        - Note: this is really inefficiently implemented currently and can deffo be improved.
    """

    normal_x = np.zeros_like(phi)
    normal_y = np.zeros_like(phi)
    normal_z = np.zeros_like(phi)
    phiMask = (np.abs(phi[:]) > zero_tol)
    normal_x[phiMask] = 1.0

    normGradPhiSq = phi_x[:]**2 + phi_y[:]**2 + phi_z[:]**2
    phiMask[:] *= (normGradPhiSq[:] >= zero_tol)

    sgnPhi = phi[:] / np.sqrt(phi[:]**2 + normGradPhiSq[:] * np.max([dx,dy,dz])**2)

    normal_x[phiMask] = sgnPhi[phiMask] * phi_x[phiMask] / \
                        np.sqrt(normGradPhiSq[:])
    normal_y[phiMask] = sgnPhi[phiMask] * phi_y[phiMask] / \
                        np.sqrt(normGradPhiSq[:])
    normal_z[phiMask] = sgnPhi[phiMask] * phi_z[phiMask] / \
                        np.sqrt(normGradPhiSq[:])

    return normal_x[:], normal_y[:], normal_z[:]
