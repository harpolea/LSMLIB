import numpy as np

__docformat__ = 'restructuredtext'

def gradPhi(phi, dx=1., dy=1., dz=1.):
    r"""
    Computes the fourth-order, central,
    finite difference approximation to the gradient of :math:`\phi`

    :Arguments:
        - `\phi`:            :math:`\phi`
        - `dx`, `dy`, `dz`:  grid cell size

    :Returns:
        - `phi_*`:           components of  :math:`\nabla \phi`
    """

    phi_z, phi_y, phi_x = np.gradient(phi, dx, dy, dz)

    return phi_x, phi_y, phi_z


def gradPhi2d(phi, dx=1., dy=1.):
    r"""
    Computes the fourth-order, central,
    finite difference approximation to the gradient of :math:`\phi`

    :Arguments:
        - `\phi`:            :math:`\phi`
        - `dx`, `dy`:  grid cell size

    :Returns:
        - `phi_*`:           components of  :math:`\nabla \phi`
    """

    phi_y, phi_x = np.gradient(phi, dx, dy)

    return phi_x, phi_y


def surfaceAreaLevelSet(phi, phi_x, phi_y, phi_z, ibLims,
                            dx=1., dy=1., dz=1.,
                            epsilon=1.):

    r"""
    Computes the surface area of the surface defined by the zero level set.

     :Parameters:

      - `phi`:              level set function
      - `phi_*` :           components of :math:`\nabla \phi`
      - `dx`, `dy`, `dz`:   grid spacing
      - `epsilon`:          width of numerical smoothing to use for Heaviside function
      - `ibLims`:           index range for interior box (the region which the function will actually look at - it's probably more useful to just have this as an optional mask)

     :Returns:

        - `area`:           area of the surface defined by the zero level                               set

    """

    dV = dx * dy * dz   #volume element
    ilo, ihi, jlo, jhi, klo, khi = ibLims
    ihi+=1
    jhi+=1
    khi+=1

    mask = np.zeros_like(phi, dtype=bool)
    delta = np.zeros_like(phi)
    # define mask so don't need doubly nested for loops
    mask[ilo:ihi, jlo:jhi, klo:khi] = (np.abs(phi[ilo:ihi, jlo:jhi, klo:khi]) < epsilon)
    delta[mask] = 0.5 * (1. + np.cos(np.pi * phi[mask] /epsilon))/epsilon

    return np.sum(delta[mask] * dV * np.sqrt(phi_x[mask]**2 + phi_y[mask]**2 + phi_z[mask]**2))

def surfaceAreaLevelSet2d(phi, phi_x, phi_y, ibLims,
                            dx=1., dy=1.,
                            epsilon=1.):

    r"""
    Computes the surface area of the surface defined by the zero level set.

     :Parameters:

      - `phi`:              level set function
      - `phi_*` :           components of :math:`\nabla \phi`
      - `dx`, `dy`:   grid spacing
      - `epsilon`:          width of numerical smoothing to use for Heaviside function
      - `ibLims`:           index range for interior box (the region which the function will actually look at - it's probably more useful to just have this as an optional mask)

     :Returns:

        - `area`:           area of the surface defined by the zero level                               set

    """

    dV = dx * dy   #volume element
    ilo, ihi, jlo, jhi = ibLims
    ihi+=1
    jhi+=1

    mask = np.zeros_like(phi, dtype=bool)
    delta = np.zeros_like(phi)
    # define mask so don't need doubly nested for loops
    mask[ilo:ihi, jlo:jhi] = (np.abs(phi[ilo:ihi, jlo:jhi]) < epsilon)
    delta[mask] = 0.5 * (1. + np.cos(np.pi * phi[mask] /epsilon))/epsilon

    return np.sum(delta[mask] * dV * np.sqrt(phi_x[mask]**2 + phi_y[mask]**2))


def meanCurvature(phi, phi_x, phi_y, phi_z, fbLims,
                            dx=1., dy=1., dz=1., zero_tol=1.e-11):
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

        - `phi`:            level set function
        - `phi_*`:          first order derivatives of :math:`\phi`
        - `fbLims`:         index range for fillbox (the region which the function will actually look at - it's probably more useful to just have this as an optional mask)
        - `dx`, `dy`, `dz`: grid spacing
        - `zero_tol`:       any curvature less than this will be set to zero

    :Returns:

        - `kappa`:          curvature data array

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
        (12. * dx**2)

    phi_yy[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi[ilo:ihi,jlo+2:jhi+2,klo:khi] + \
        16.*phi[ilo:ihi,jlo+1:jhi+1,klo:khi] -\
        30.*phi[ilo:ihi,jlo:jhi,klo:khi] -\
        phi[ilo:ihi,jlo-2:jhi-2,klo:khi] +\
        16.*phi[ilo:ihi,jlo-1:jhi-1,klo:khi]) / \
        (12. * dy**2)

    phi_zz[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi[ilo:ihi,jlo:jhi,klo+2:khi+2] + \
        16.*phi[ilo:ihi,jlo:jhi,klo+1:khi+1] -\
        30.*phi[ilo:ihi,jlo:jhi,klo:khi] -\
        phi[ilo:ihi,jlo:jhi,klo-2:khi-2] +\
        16.*phi[ilo:ihi,jlo:jhi,klo-1:khi-1]) / \
        (12. * dz**2)

    phi_xy[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi_x[ilo:ihi,jlo+2:jhi+2,klo:khi] + \
        8.*phi_x[ilo:ihi,jlo+1:jhi+1,klo:khi] +\
        phi_x[ilo:ihi,jlo-2:jhi-2,klo:khi] -\
        8.*phi_x[ilo:ihi,jlo-1:jhi-1,klo:khi]) / \
        (12. * dy)

    phi_yz[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi_y[ilo:ihi,jlo:jhi,klo+2:khi+2] + \
        8.*phi_y[ilo:ihi,jlo:jhi,klo+1:khi+1] +\
        phi_y[ilo:ihi,jlo:jhi,klo-2:khi-2] -\
        8.*phi_y[ilo:ihi,jlo:jhi,klo-1:khi-1]) / \
        (12. * dz)

    phi_zx[ilo:ihi,jlo:jhi,klo:khi] = \
        (-phi_z[ilo+2:ihi+2,jlo:jhi,klo:khi] + \
        8.*phi_z[ilo+1:ihi+1,jlo:jhi,klo:khi] +\
        phi_z[ilo-2:ihi-2,jlo:jhi,klo:khi] -\
        8.*phi_z[ilo-1:ihi-1,jlo:jhi,klo:khi]) / \
        (12. * dx)

    denominator = np.sqrt(phi_x[:,:,:]**2 + phi_y[:,:,:]**2 + phi_z[:,:,:]**2) ** 3

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


def meanCurvature2d(phi, phi_x, phi_y, fbLims,
                            dx=1., dy=1., zero_tol=1.e-11):
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

        - `phi`:            level set function
        - `phi_*`:          first order derivatives of :math:`\phi`
        - `fbLims`:         index range for fillbox (the region which the function will actually look at - it's probably more useful to just have this as an optional mask)
        - `dx`, `dy`: grid spacing
        - `zero_tol`:       any curvature less than this will be set to zero

    :Returns:

        - `kappa`:          curvature data array

    """

    #initialise
    kappa = np.zeros_like(phi)

    phi_xx = np.zeros_like(phi)
    phi_yy= np.zeros_like(phi)
    phi_xy = np.zeros_like(phi)

    ilo, ihi, jlo, jhi = fbLims
    ihi+=1
    jhi+=1

    phi_xx[ilo:ihi,jlo:jhi] = \
        (-phi[ilo+2:ihi+2,jlo:jhi] + 16.*phi[ilo+1:ihi+1,jlo:jhi] - \
        30.*phi[ilo:ihi,jlo:jhi] - phi[ilo-2:ihi-2,jlo:jhi] + \
        16.*phi[ilo-1:ihi-1,jlo:jhi]) / (12. * dx**2)

    phi_yy[ilo:ihi,jlo:jhi] = \
        (-phi[ilo:ihi,jlo+2:jhi+2] + 16.*phi[ilo:ihi,jlo+1:jhi+1,] - \
        30.*phi[ilo:ihi,jlo:jhi] - phi[ilo:ihi,jlo-2:jhi-2] + \
        16.*phi[ilo:ihi,jlo-1:jhi-1]) / (12. * dy**2)

    phi_xy[ilo:ihi,jlo:jhi] = \
        (-phi_x[ilo:ihi,jlo+2:jhi+2] + 8.*phi_x[ilo:ihi,jlo+1:jhi+1] + \
        phi_x[ilo:ihi,jlo-2:jhi-2] - 8.*phi_x[ilo:ihi,jlo-1:jhi-1]) / \
        (12. * dy)

    denominator = phi_x[:,:]**2 + phi_y[:,:]**2

    kappa[ilo:ihi,jlo:jhi] = phi_xx[ilo:ihi,jlo:jhi] * \
        phi_y[ilo:ihi,jlo:jhi]**2 + \
        phi_yy[ilo:ihi,jlo:jhi]*phi_x[ilo:ihi,jlo:jhi]**2 - \
        2.* phi_xy[ilo:ihi,jlo:jhi] * phi_x[ilo:ihi,jlo:jhi] * \
        phi_y[ilo:ihi,jlo:jhi]

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

       - `phi`:             level set function
       - `phi_*`:           components of :math:`\nabla \phi`
       - `dx`, `dy`, `dz`:  grid spacing
       - `zero_tol`:        only calculate normal if :math:`| \nabla \phi |` is greater than this

      :Returns:

        - `normal_*`:       components of unit normal vector

      :Notes:

        - When :math:`| \nabla \phi |`  is close to zero, the unit normal is arbitrarily set to be :math:`(1.0, 0.0, 0.0)`
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
                        np.sqrt(normGradPhiSq[phiMask])
    normal_y[phiMask] = sgnPhi[phiMask] * phi_y[phiMask] / \
                        np.sqrt(normGradPhiSq[phiMask])
    normal_z[phiMask] = sgnPhi[phiMask] * phi_z[phiMask] / \
                        np.sqrt(normGradPhiSq[phiMask])

    return normal_x[:], normal_y[:], normal_z[:]


def signedUnitNormal2d(phi, phi_x, phi_y,
                        dx=1., dy=1., zero_tol=1.e-11):
    r"""
      Computes the signed unit normal
      vector (sgn(phi)*normal) to the interface from :math:`\nabla \phi`
      using the following smoothed sgn function

      .. math::

        sgn(\phi) = \phi / \sqrt{ \phi^2 + |\nabla \phi|^2 * dx^2 }


      :Parameters:

       - `phi`:             level set function
       - `phi_*`:           components of :math:`\nabla \phi`
       - `dx`, `dy`:  grid spacing
       - `zero_tol`:        only calculate normal if :math:`| \nabla \phi |` is greater than this

      :Returns:

        - `normal_*`:       components of unit normal vector

      :Notes:

        - When :math:`| \nabla \phi |`  is close to zero, the unit normal is arbitrarily set to be :math:`(1.0, 0.0, 0.0)`
    """

    normal_x = np.zeros_like(phi)
    normal_y = np.zeros_like(phi)
    phiMask = (np.abs(phi[:]) > zero_tol)
    normal_x[phiMask] = 1.0

    normGradPhiSq = phi_x[:]**2 + phi_y[:]**2
    phiMask[:] *= (normGradPhiSq[:] >= zero_tol)

    sgnPhi = phi[:] / np.sqrt(phi[:]**2 + normGradPhiSq[:] * np.max([dx,dy])**2)

    normal_x[phiMask] = sgnPhi[phiMask] * phi_x[phiMask] / \
                        np.sqrt(normGradPhiSq[phiMask])
    normal_y[phiMask] = sgnPhi[phiMask] * phi_y[phiMask] / \
                        np.sqrt(normGradPhiSq[phiMask])

    return normal_x[:], normal_y[:]


def strainRate(phi, phi_x, phi_y, phi_z, u, v, w, dx=1., dy=1., dz=1.):
    r"""
      Computes the strain rate

      .. math::

        S = -\vec{n}\cdot\vec{\nabla}\vec{v}\cdot\vec{n} = -n^i\nabla_iv^jn^k\delta_{jk}

      :Parameters:

       - `phi`:             level set function
       - `phi_*`:           components of :math:`\nabla \phi`
       - `u`, `v`, `w`:     components of velocity
       - `dx`, `dy`, `dz`:  grid spacing

      :Returns:

        - `S`:              strain rate

    """

    norm_x, norm_y, norm_z = signedUnitNormal(phi, phi_x, phi_y, phi_z,
                            dx=dx, dy=dy, dz=dz)
    gradu_x, gradu_y, gradu_z = np.gradient(u, dx, dy, dz)
    gradv_x, gradv_y, gradv_z = np.gradient(v, dx, dy, dz)
    gradw_x, gradw_y, gradw_z = np.gradient(w, dx, dy, dz)

    gradvs_x = np.array([gradu_x, gradv_x, gradw_x])
    gradvs_y = np.array([gradu_y, gradv_y, gradw_y])
    gradvs_z = np.array([gradu_z, gradv_z, gradw_z])

    norms = np.array([norm_x, norm_y, norm_z])

    S = np.zeros_like(phi)
    Stemp = np.zeros_like([phi,phi,phi])

    for i in range(0,3):
        Stemp[0,:,:,:] += gradvs_x[i,:,:,:] * norms[i,:,:,:]
        Stemp[1,:,:,:] += gradvs_y[i,:,:,:] * norms[i,:,:,:]
        Stemp[2,:,:,:] += gradvs_z[i,:,:,:] * norms[i,:,:,:]

    S[:,:,:] = -(norm_x[:,:,:] * Stemp[0,:,:,:] + \
          norm_y[:,:,:] * Stemp[1,:,:,:] + \
          norm_z[:,:,:] * Stemp[2,:,:,:])

    return S[:,:,:]


def strainRate2d(phi, phi_x, phi_y, u, v, dx=1., dy=1.):
    r"""
      Computes the strain rate

      .. math::

        S = -\vec{n}\cdot\vec{\nabla}\vec{v}\cdot\vec{n} = -n^i\nabla_iv^jn^k\delta_{jk}

      :Parameters:

       - `phi`:             level set function
       - `phi_*`:           components of :math:`\nabla \phi`
       - `u`, `v`:     components of velocity
       - `dx`, `dy`:  grid spacing

      :Returns:

        - `S`:              strain rate

    """
    norm_x, norm_y = signedUnitNormal2d(phi, phi_x, phi_y,
                            dx=dx, dy=dy)
    gradu_x, gradu_y = np.gradient(u, dx, dy)
    gradv_x, gradv_y = np.gradient(v, dx, dy)


    gradvs_x = np.array([gradu_x, gradv_x])
    gradvs_y = np.array([gradu_y, gradv_y])

    norms = np.array([norm_x, norm_y])

    S = np.zeros_like(phi)
    Stemp = np.zeros_like([phi,phi])

    for i in range(0,2):
        Stemp[0,:,:] += gradvs_x[i,:,:] * norms[i,:,:]
        Stemp[1,:,:] += gradvs_y[i,:,:] * norms[i,:,:]

    S[:,:] = -(norm_x[:,:] * Stemp[0,:,:] + \
          norm_y[:,:] * Stemp[1,:,:])

    return S[:,:]


def laminarFlameSpeed(phi, sL0, marksteinLength, u, v, w, ibLims, dx=1., dy=1., dz=1.):
    r"""
      Computes the laminar flame speed

      .. math::

        s_L = s_L^0 - s_L^0 \mathcal{L}\kappa - \mathcal{L}S

      :Parameters:

       - `phi`:             level set function
       - `sL0`:             burning velocity of the unstretched laminar flame
       - `marksteinLength`: Markstein length :math:`\mathcal{L}`
       - `u`, `v`, `w`:     components of velocity
       - `ibLims`:          index range for interior box (the region which the function will actually look at - it's probably more useful to just have this as an optional mask)
       - `dx`, `dy`, `dz`:  grid spacing

      :Returns:

        - `sL`:             laminar flame speed

    """

    phi_x, phi_y, phi_z = gradPhi(phi, dx=dx, dy=dy, dz=dz)
    kappa = meanCurvature(phi, phi_x, phi_y, phi_z, ibLims,
                                dx=dx, dy=dy, dz=dz)
    S = strainRate(phi, phi_x, phi_y, phi_z, u, v, w, dx=dx, dy=dy, dz=dz)

    sL = sL0 * (np.ones_like(phi) - marksteinLength * kappa[:,:,:]) - \
            marksteinLength * S[:,:,:]

    return sL[:,:,:]

def laminarFlameSpeed2d(phi, sL0, marksteinLength, u, v, ibLims, dx=1., dy=1., dz=1.):
    r"""
      Computes the laminar flame speed

      .. math::

        s_L = s_L^0 - s_L^0 \mathcal{L}\kappa - \mathcal{L}S

      :Parameters:

       - `phi`:             level set function
       - `sL0`:             burning velocity of the unstretched laminar flame
       - `marksteinLength`: Markstein length :math:`\mathcal{L}`
       - `u`, `v`:     components of velocity
       - `ibLims`:          index range for interior box (the region which the function will actually look at - it's probably more useful to just have this as an optional mask)
       - `dx`, `dy`:  grid spacing

      :Returns:

        - `sL`:             laminar flame speed

    """
    phi_x, phi_y = gradPhi2d(phi, dx=dx, dy=dy)
    kappa = meanCurvature2d(phi, phi_x, phi_y, ibLims,
                                dx=dx, dy=dy)
    S = strainRate2d(phi, phi_x, phi_y, u, v, dx=dx, dy=dy)

    sL = sL0 * (np.ones_like(phi) - marksteinLength * kappa[:,:]) - \
            marksteinLength * S[:,:]

    return sL[:,:]
