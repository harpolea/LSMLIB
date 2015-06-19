from lsmlib import computeDistanceFunction3d_
from lsmlib import computeExtensionFields3d_
from lsmlib import solveEikonalEquation3d_
from lsmlib import lsm3dcomputesignedunitnormal
from lsmlib import lsm3dsurfaceareazerolevelset
from lsmlib import lsm3dcomputemeancurvatureorder2local
from lsmlib import lsm3dcomputemeancurvatureorder2
from lsmlib import lsm3dcomputegaussiancurvatureorder2
from lsmlib import lsm3dcentralgradorder4
import pythonisedfns
import lsmfns
import numpy as np

__docformat__ = 'restructuredtext'

def _getVersion():
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        version = get_distribution(__name__).version
    except DistributionNotFound:
        version = "unknown, try running `python setup.py egg_info`"

    return version

__version__ = _getVersion()

def toFloatArray(arr):
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    return arr.astype(float)

def getShape(phi, dx, order):

    phi = toFloatArray(phi)

    shape = phi.shape

    if type(dx) in (int, float, tuple, list):
        dx = np.array(dx)

    if dx.shape == ():
        dx = np.resize(dx, (len(shape),))

    if len(dx) != len(shape):
        raise ValueError, "dx must be of length len(phi.shape)"

    if order not in (1, 2):
        raise ValueError, 'order must be 1 or 2'

    dx = dx.astype(float)

    if np.prod(dx) == 0:
        raise ValueError, 'dx must be greater than zero'

    if len(shape) == 1:
        nx, ny = shape[0], 1
        dx, dy = dx[0], 1.
    elif len(shape) == 2:
        ny, nx = shape
        dy, dx = dx[0], dx[1]
    elif len(shape) == 3:
        raise Exception, "3D meshes not yet implemented"

    return nx, ny, dx, dy, shape, phi

def get3dShape(phi, dx, order):
    phi = toFloatArray(phi)

    shape = phi.shape

    if len(shape) < 3:
        nx, ny, dx, dy, shape, phi = getShape(phi, dx, order)
        nz = 1
        dz = 0.
    elif len(shape) == 3:
        if type(dx) in (int, float, tuple, list):
            dx = np.array(dx)

        if dx.shape == ():
            dx = np.resize(dx, (len(shape),))

        if len(dx) != len(shape):
            raise ValueError, "dx must be of length len(phi.shape)"

        if order not in (1, 2):
            raise ValueError, 'order must be 1 or 2'

        dx = dx.astype(float)

        if np.prod(dx) == 0:
            raise ValueError, 'dx must be greater than zero'

        nz, ny, nx = shape
        dz, dy, dx = dx[0], dx[1], dx[2]


    return nx, ny, nz, dx, dy, dz, shape, phi


def surfaceAreaZeroLevelSet(phi0, phi_x, phi_y, phi_z,
                            gbGradPhiLims, gbPhiLims, ibLims, dx=1.,
                            epsilon=1., order=2):

    r"""
    Computes the surface area of the surface defined by the zero level set.

     :Parameters:

      - `phi0`:              level set function
      - `phi_*` :            components of :math:`\nabla \phi`
      - `dx`:       grid spacing
      - `epsilon`:          width of numerical smoothing to use for Heaviside function
      - `*_gb`:             index range for ghostbox
      - `*_ib`:             index range for interior box

     :Returns:

        - `area`:            area of the surface defined by the zero level                               set

    """

    nx, ny, nz, dx, dy, dz, _, phi0 = get3dShape(phi0, dx, order)

    #print(nx,ny,nz,dx,dy,dz)

    ilo_grad_phi_gb = gbGradPhiLims[0]
    ihi_grad_phi_gb = gbGradPhiLims[1]
    jlo_grad_phi_gb = gbGradPhiLims[2]
    jhi_grad_phi_gb = gbGradPhiLims[3]
    klo_grad_phi_gb = gbGradPhiLims[4]
    khi_grad_phi_gb = gbGradPhiLims[5]

    ilo_phi_gb = gbPhiLims[0]
    ihi_phi_gb = gbPhiLims[1]
    jlo_phi_gb = gbPhiLims[2]
    jhi_phi_gb = gbPhiLims[3]
    klo_phi_gb = gbPhiLims[4]
    khi_phi_gb = gbPhiLims[5]

    ilo_ib = ibLims[0]
    ihi_ib = ibLims[1]
    jlo_ib = ibLims[2]
    jhi_ib = ibLims[3]
    klo_ib = ibLims[4]
    khi_ib = ibLims[5]

    return lsm3dsurfaceareazerolevelset(phi0.flatten(), phi_x.flatten(),
                                phi_y.flatten(), phi_z.flatten(),
                                ilo_grad_phi_gb, ihi_grad_phi_gb,
                                jlo_grad_phi_gb,
                                jhi_grad_phi_gb, klo_grad_phi_gb,
                                khi_grad_phi_gb,
                                ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                                jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                                ilo_ib, ihi_ib, jlo_ib, jhi_ib, klo_ib, khi_ib,
                                nx=nx, ny=ny, nz=1, dx=dx, dy=dy, dz=1.,
                                epsilon=epsilon)


def computeMeanCurvatureLocal(phi0, phi_x, phi_y, phi_z,
                            kappa, grad_phi_mag, gbKappaLims,
                            index_x, index_y, index_z, narrow_band, mark_fb,
                            gbGradPhiLims, gbPhiLims, nbgbLims, nbLims,
                            dx=1., order=2):
    r"""

      Computes mean curvature

      .. math::

        \kappa = \nabla ( \nabla\phi / |\nabla\phi|)

      .. math::

        \kappa = ( \phi_{xx}\phi_y^2 + \phi_{yy}\phi_x^2 - 2\phi_{xy}\phi_x\phi_y +
                \phi_{xx}\phi_z^2 + \phi_{zz}\phi_x^2 - 2\phi_{xz}\phi_x\phi_z +\\
                \phi_{yy}\phi_z^2 + \phi_{zz}\phi_y^2 - 2\phi_{yz}\phi_y\phi_z )/
              ( | \nabla \phi | ^ 3 )

      Note that this value is technically twice the mean curvature.
      Standard centered 27 point stencil, second order differencing used.
      First order derivatives assumed precomputed.

      :Parameters:

        - `kappa`: curvature data array
        - `phi0`:  level set function
        - `phi_*`:  first order derivatives of :math:`\phi`
        - `grad_phi_mag`:  gradient magnitude of :math:`\phi`
        - `*_gb`:   index range for ghostbox
        - `dx`:   grid spacing
        - `index_[xyz]`:  [xyz] coordinates of local (narrow band) points
        - `n*_index`:  index range of points to loop over in index_*
        - `narrow_band`:   array that marks voxels outside desired fillbox
        - `mark_fb`:      upper limit narrow band value for voxels in fillbox

    :Returns:

        - `kappa`: curvature data array

    """

    nx, ny, nz, dx, dy, dz, shape, phi0 = get3dShape(phi0, dx, order)
    ilo_kappa_gb = gbKappaLims[0]
    ihi_kappa_gb = gbKappaLims[1]
    jlo_kappa_gb = gbKappaLims[2]
    jhi_kappa_gb = gbKappaLims[3]
    klo_kappa_gb = gbKappaLims[4]
    khi_kappa_gb = gbKappaLims[5]

    ilo_grad_phi_gb = gbGradPhiLims[0]
    ihi_grad_phi_gb = gbGradPhiLims[1]
    jlo_grad_phi_gb = gbGradPhiLims[2]
    jhi_grad_phi_gb = gbGradPhiLims[3]
    klo_grad_phi_gb = gbGradPhiLims[4]
    khi_grad_phi_gb = gbGradPhiLims[5]

    ilo_phi_gb = gbPhiLims[0]
    ihi_phi_gb = gbPhiLims[1]
    jlo_phi_gb = gbPhiLims[2]
    jhi_phi_gb = gbPhiLims[3]
    klo_phi_gb = gbPhiLims[4]
    khi_phi_gb = gbPhiLims[5]

    ilo_nb_gb = nbgbLims[0]
    ihi_nb_gb = nbgbLims[1]
    jlo_nb_gb = nbgbLims[2]
    jhi_nb_gb = nbgbLims[3]
    klo_nb_gb = nbgbLims[4]
    khi_nb_gb = nbgbLims[5]

    nlo_index = nbLims[0]
    nhi_index = nbLims[1]

    kappa = lsm3dcomputemeancurvatureorder2local(kappa.flatten(),
                          ilo_kappa_gb, ihi_kappa_gb, jlo_kappa_gb,
                          jhi_kappa_gb, klo_kappa_gb, khi_kappa_gb,
                          phi0.flatten(),
                          ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                          jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                          phi_x.flatten(), phi_y.flatten(), phi_z.flatten(),
                          grad_phi_mag.flatten(),
                          ilo_grad_phi_gb, ihi_grad_phi_gb,
                          jlo_grad_phi_gb, jhi_grad_phi_gb,
                          klo_grad_phi_gb, khi_grad_phi_gb,
                          index_x,
                          index_y,
                          index_z,
                          nlo_index, nhi_index,
                          narrow_band,
                          ilo_nb_gb, ihi_nb_gb, jlo_nb_gb,
                          jhi_nb_gb, klo_nb_gb, khi_nb_gb,
                          mark_fb,
                          nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

    return kappa.reshape(shape)



def computeMeanCurvature(phi0, phi_x, phi_y, phi_z,
                            kappa, grad_phi_mag, gbKappaLims,
                            gbGradPhiLims, gbPhiLims, fbKappaLims,
                            dx=1., order=2):
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

        - `kappa`: curvature data array
        - `phi0`:  level set function
        - `phi_*`:  first order derivatives of :math:`\phi`
        - `grad_phi_mag`:  gradient magnitude of :math:`\phi`
        - `*_gb`:   index range for ghostbox
        - `*_fb`:   index range for fillbox
        - `dx`:   grid spacing

    :Returns:

        - `kappa`: curvature data array

    """

    nx, ny, nz, dx, dy, dz, shape, phi0 = get3dShape(phi0, dx, order)
    ilo_kappa_gb = gbKappaLims[0]
    ihi_kappa_gb = gbKappaLims[1]
    jlo_kappa_gb = gbKappaLims[2]
    jhi_kappa_gb = gbKappaLims[3]
    klo_kappa_gb = gbKappaLims[4]
    khi_kappa_gb = gbKappaLims[5]

    ilo_grad_phi_gb = gbGradPhiLims[0]
    ihi_grad_phi_gb = gbGradPhiLims[1]
    jlo_grad_phi_gb = gbGradPhiLims[2]
    jhi_grad_phi_gb = gbGradPhiLims[3]
    klo_grad_phi_gb = gbGradPhiLims[4]
    khi_grad_phi_gb = gbGradPhiLims[5]

    ilo_phi_gb = gbPhiLims[0]
    ihi_phi_gb = gbPhiLims[1]
    jlo_phi_gb = gbPhiLims[2]
    jhi_phi_gb = gbPhiLims[3]
    klo_phi_gb = gbPhiLims[4]
    khi_phi_gb = gbPhiLims[5]

    ilo_kappa_fb = fbKappaLims[0]
    ihi_kappa_fb = fbKappaLims[1]
    jlo_kappa_fb = fbKappaLims[2]
    jhi_kappa_fb = fbKappaLims[3]
    klo_kappa_fb = fbKappaLims[4]
    khi_kappa_fb = fbKappaLims[5]

    kappa = lsm3dcomputemeancurvatureorder2(kappa.flatten(),
                          ilo_kappa_gb, ihi_kappa_gb, jlo_kappa_gb,
                          jhi_kappa_gb, klo_kappa_gb, khi_kappa_gb,
                          phi0.flatten(),
                          ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                          jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                          phi_x.flatten(), phi_y.flatten(), phi_z.flatten(),
                          grad_phi_mag.flatten(),
                          ilo_grad_phi_gb, ihi_grad_phi_gb,
                          jlo_grad_phi_gb, jhi_grad_phi_gb,
                          klo_grad_phi_gb, khi_grad_phi_gb,
                          ilo_kappa_fb, ihi_kappa_fb, jlo_kappa_fb,
                          jhi_kappa_fb, klo_kappa_fb, khi_kappa_fb,
                          nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

    return kappa.reshape(shape)


def computeGaussianCurvature(phi0, phi_x, phi_y, phi_z,
                            kappa, grad_phi_mag, gbKappaLims,
                            gbGradPhiLims, gbPhiLims, fbKappaLims,
                            dx=1., order=2):
    r"""

      Computes Gaussian curvature

      .. math::

        \kappa = [  \phi_x^2(\phi_{yy}\phi_{zz} - \phi_{yz}^2) +
                     \phi_y^2(\phi_{xx}\phi_{zz} - \phi_{xz}^2) +
                     \phi_z^2(\phi_{xx}\phi_{yy} - \phi_{xy}^2) + \\
                 2( \phi_x\phi_y(\phi_{xz}\phi_{yz} - \phi_{xy}\phi_{zz}) +
                     \phi_y\phi_z(\phi_{xy}\phi_{xz} - \phi_{yz}\phi_{xx}) +
                     \phi_x\phi_z(\phi_{xy}\phi_{yz} - \phi_{xz}\phi_{yy}) ) ] /
                  ( | \nabla \phi | ^ 4 )

      Standard centered 27 point stencil, second order differencing used.
      First order derivatives assumed precomputed.

      :Parameters:

        - `kappa`: curvature data array
        - `phi0`:  level set function
        - `phi_*`:  first order derivatives of :math:`\phi`
        - `grad_phi_mag`:  gradient magnitude of :math:`\phi`
        - `*_gb`:   index range for ghostbox
        - `*_fb`:   index range for fillbox
        - `dx`:   grid spacing

    :Returns:

        - `kappa`: curvature data array

    """

    nx, ny, nz, dx, dy, dz, shape, phi0 = get3dShape(phi0, dx, order)
    ilo_kappa_gb = gbKappaLims[0]
    ihi_kappa_gb = gbKappaLims[1]
    jlo_kappa_gb = gbKappaLims[2]
    jhi_kappa_gb = gbKappaLims[3]
    klo_kappa_gb = gbKappaLims[4]
    khi_kappa_gb = gbKappaLims[5]

    ilo_grad_phi_gb = gbGradPhiLims[0]
    ihi_grad_phi_gb = gbGradPhiLims[1]
    jlo_grad_phi_gb = gbGradPhiLims[2]
    jhi_grad_phi_gb = gbGradPhiLims[3]
    klo_grad_phi_gb = gbGradPhiLims[4]
    khi_grad_phi_gb = gbGradPhiLims[5]

    ilo_phi_gb = gbPhiLims[0]
    ihi_phi_gb = gbPhiLims[1]
    jlo_phi_gb = gbPhiLims[2]
    jhi_phi_gb = gbPhiLims[3]
    klo_phi_gb = gbPhiLims[4]
    khi_phi_gb = gbPhiLims[5]

    ilo_kappa_fb = fbKappaLims[0]
    ihi_kappa_fb = fbKappaLims[1]
    jlo_kappa_fb = fbKappaLims[2]
    jhi_kappa_fb = fbKappaLims[3]
    klo_kappa_fb = fbKappaLims[4]
    khi_kappa_fb = fbKappaLims[5]

    kappa = lsm3dcomputegaussiancurvatureorder2(kappa.flatten(),
                          ilo_kappa_gb, ihi_kappa_gb, jlo_kappa_gb,
                          jhi_kappa_gb, klo_kappa_gb, khi_kappa_gb,
                          phi0.flatten(),
                          ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                          jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                          phi_x.flatten(), phi_y.flatten(), phi_z.flatten(),
                          grad_phi_mag.flatten(),
                          ilo_grad_phi_gb, ihi_grad_phi_gb,
                          jlo_grad_phi_gb, jhi_grad_phi_gb,
                          klo_grad_phi_gb, khi_grad_phi_gb,
                          ilo_kappa_fb, ihi_kappa_fb, jlo_kappa_fb,
                          jhi_kappa_fb, klo_kappa_fb, khi_kappa_fb,
                          nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

    return kappa.reshape(shape)



def computeSignedUnitNormal(phi0, phi_x, phi_y, phi_z, gbNormalLims,
                            gbGradPhiLims, gbPhiLims, fbLims, dx=1., order=2):
    r"""
      Computes the signed unit normal
      vector (sgn(phi)*normal) to the interface from :math:`\nabla \phi`
      using the following smoothed sgn function

      .. math::

        sgn(\phi) = \phi / \sqrt{ \phi^2 + |\nabla \phi|^2 * dx^2 }


      :Parameters:

       - `phi_*`:         components of :math:`\nabla \phi`
       - `phi0`:           level set function
       - `dx`:    grid spacing
       - `*_gb`:          index range for ghostbox
       - `*_fb`:          index range for fillbox

      :Returns:

        - `normal_*`:     components of unit normal vector

      :Notes:

        - When :math:`| \nabla \phi |`  is close to zero, the unit normal is arbitrarily set to be :math:`(1.0, 0.0, 0.0)`.
        - Note: this is really inefficiently implemented currently and can deffo be improved.
    """
    nx, ny, nz, dx, dy, dz, shape, phi0 = get3dShape(phi0, dx, order)
    ilo_normal_gb = gbNormalLims[0]
    ihi_normal_gb = gbNormalLims[1]
    jlo_normal_gb = gbNormalLims[2]
    jhi_normal_gb = gbNormalLims[3]
    klo_normal_gb = gbNormalLims[4]
    khi_normal_gb = gbNormalLims[5]

    ilo_grad_phi_gb = gbGradPhiLims[0]
    ihi_grad_phi_gb = gbGradPhiLims[1]
    jlo_grad_phi_gb = gbGradPhiLims[2]
    jhi_grad_phi_gb = gbGradPhiLims[3]
    klo_grad_phi_gb = gbGradPhiLims[4]
    khi_grad_phi_gb = gbGradPhiLims[5]

    ilo_phi_gb = gbPhiLims[0]
    ihi_phi_gb = gbPhiLims[1]
    jlo_phi_gb = gbPhiLims[2]
    jhi_phi_gb = gbPhiLims[3]
    klo_phi_gb = gbPhiLims[4]
    khi_phi_gb = gbPhiLims[5]

    ilo_fb = fbLims[0]
    ihi_fb = fbLims[1]
    jlo_fb = fbLims[2]
    jhi_fb = fbLims[3]
    klo_fb = fbLims[4]
    khi_fb = fbLims[5]

    normal_x, normal_y, normal_z = lsm3dcomputesignedunitnormal(phi0.flatten(),
                                phi_x.flatten(),
                                phi_y.flatten(), phi_z.flatten(),
                                ilo_normal_gb, ihi_normal_gb, jlo_normal_gb,
                                jhi_normal_gb ,klo_normal_gb, khi_normal_gb,
                                ilo_grad_phi_gb, ihi_grad_phi_gb, jlo_grad_phi_gb,
                                jhi_grad_phi_gb, klo_grad_phi_gb, khi_grad_phi_gb,
                                ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                                jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                                ilo_fb, ihi_fb, jlo_fb, jhi_fb, klo_fb, khi_fb, nx=nx, ny=ny, nz=1, dx=dx, dy=dy, dz=1.)

    return normal_x.reshape((ny,nx)), normal_y.reshape((ny,nx)), normal_z.reshape((ny,nx))


def centralGradOrder4(phi0, gbGradPhiLims, gbPhiLims, fbLims, dx=1.,
                        order=2):
    r"""
    Computes the fourth-order, central,
    finite difference approximation to the gradient of :math:`\phi`
    using the formula:

    .. math::

       \left( \frac{\partial \phi}{\partial x} \right)_i \approx
          \frac{ -\phi_{i+2} + 8 \phi_{i+1} - 8 \phi_{i-1} + \phi_{i-2} }
               { 12 dx }



    :Arguments:
        - `\phi`:         :math:`\phi`
        - `dx`, `dy`, `dz`:  grid cell size
        - `*_gb`:        index range for ghostbox
        - `*_fb`:        index range for fillbox

    :Returns:
        - `phi_*`:      components of  :math:`\nabla \phi`
    """

    nx, ny, nz, dx, dy, dz, shape, phi0 = get3dShape(phi0, dx, order)

    ilo_grad_phi_gb = gbGradPhiLims[0]
    ihi_grad_phi_gb = gbGradPhiLims[1]
    jlo_grad_phi_gb = gbGradPhiLims[2]
    jhi_grad_phi_gb = gbGradPhiLims[3]
    klo_grad_phi_gb = gbGradPhiLims[4]
    khi_grad_phi_gb = gbGradPhiLims[5]

    ilo_phi_gb = gbPhiLims[0]
    ihi_phi_gb = gbPhiLims[1]
    jlo_phi_gb = gbPhiLims[2]
    jhi_phi_gb = gbPhiLims[3]
    klo_phi_gb = gbPhiLims[4]
    khi_phi_gb = gbPhiLims[5]

    ilo_fb = fbLims[0]
    ihi_fb = fbLims[1]
    jlo_fb = fbLims[2]
    jhi_fb = fbLims[3]
    klo_fb = fbLims[4]
    khi_fb = fbLims[5]

    phi_x, phi_y, phi_z = lsm3dcentralgradorder4(ilo_grad_phi_gb,
                ihi_grad_phi_gb,
                jlo_grad_phi_gb, jhi_grad_phi_gb,
                klo_grad_phi_gb, khi_grad_phi_gb,
                phi0.flatten(),
                ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                ilo_fb, ihi_fb, jlo_fb,
                jhi_fb, klo_fb, khi_fb,
                nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

    return phi_x.reshape(shape), phi_y.reshape(shape), phi_z.reshape(shape)


def computeDistanceFunction(phi0, dx=1., order=2):
    r"""

    Solves

    .. math::

       |\nabla \phi| = 1 \;\; \text{with} \;\; \phi=0 \;\; \text{where} \;\; \phi_0=0

    for :math:`\phi` using the zero level set of :math:`\phi_0`.

    :Parameters:

      - `phi0`: array of positive and negative values locating the
        zero level set of :math:`\phi`, shape determines the
        dimension.
      - `dx`: the cell dimensions
      - `order`: order of the computational stencil, either 1 or 2

    :Returns:

      - the calculated distance function, :math:`\phi`

    """
    nx, ny, nz, dx, dy, dz, shape, phi0 = get3dShape(phi0, dx, order)
    return computeDistanceFunction3d_(phi0.flatten(), nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, order=order).reshape(shape)

def computeExtensionFields(phi0, speed, dx=1., order=2, mask=None, ext_mask=None, ):
    r"""

    Solves

    .. math::

      \nabla \phi \cdot \nabla u = 0 \;\; \text{with} \;\; u=u_0 \;\; \text{where} \;\; \phi_0=0

    for multiple extension fields :math:`u` and

    .. math::

      |\nabla \phi| = 1 \;\; \text{with} \;\; \phi=0 \;\; \text{where} \;\; \phi_0=0

    for :math:`\phi` using the zero level set of :math:`\phi_0`.

    :Parameters:

      - `phi0`: array of positive and negative values locating the
        zero level set of :math:`\phi`, shape determines the
        dimension.
      - `speed`: array of values used as the initial condition to
        calculate the extension fields, :math:`u0`; shape is either
        `phi0.shape` or `(N,) + phi0.shape`, where `N` is the number
        of fields to extend
      - `mask`:
      - `ext_mask`: array of Boolean values determining which values of
        :math:`u` are calculated; `True` values set :math:`u=u_0`,
        `False` values set :math:`u=u` determined from the eikonal
        equation; shape is `phi0.shape`
      - `dx`: the cell dimensions
      - `order`: order of the computational stencil, either 1 or 2

    :Returns:

      - a tuple containing (:math:`\phi`, :math:`u`)

    """

    nx, ny, dx, dy, shape, phi0 = getShape(phi0, dx, order)
    u0 = toFloatArray(speed)
    u0shape = u0.shape

    if u0shape == shape:
        u0 = np.reshape(u0, (1,) + u0shape)

    if shape != u0[0].shape:
        raise ValueError, "phi and speed must have the same shape"

    if mask is None:
        mask = np.empty((0,), dtype=float)
    else:
        mask = toFloatArray(mask)

        assert shape == mask.shape, '`phi` and `mask` have incompatible shapes'

    if ext_mask is None:
        ext_mask = np.empty((0,), dtype=float)
    else:
        ext_mask = toFloatArray(ext_mask)
        ext_mask = -ext_mask * 2. + 1.
        assert shape == ext_mask.shape, '`phi` and `extensionMask` have incompatible shapes'

    N = u0.shape[0]
    phi, u = computeExtensionFields3d_(phi0.flatten(),
                                       u0.reshape((N, -1)),
                                       mask=mask.flatten(),
                                       extension_mask=ext_mask.flatten(),
                                       nx=nx, ny=ny, dx=dx, dy=dy, order=order)

    return phi.reshape(shape), u.reshape(u0shape)

def solveEikonalEquation(phi0, speed, dx=1., order=2):
    r"""

    Calculates :math:`\phi` based on the speed function, :math:`F`,
    using

    .. math::

      |\nabla \phi| F = 1 \;\; \text{with} \;\; \phi=0 \;\; \text{where} \;\; \phi_0=0

    :Parameters:

      - `phi0`: array of positive and negative values locating the
        zero level set of :math:`\phi`, shape determines the
        dimension.
      - `speed`: speed function, :math:`F`, shape is `phi0.shape`
      - `dx`: the cell dimensions
      - `order`: order of the computational stencil, either 1 or 2

    :Returns:

      - the field calculated by solving the eikonal equation,
        :math:`\phi`

    """

    nx, ny, dx, dy, shape, phi0 = getShape(phi0, dx, 1)
    speed = toFloatArray(speed)

    if shape != speed.shape:
        raise ValueError, "phi and speed must have the same shape"

    return solveEikonalEquation3d_(phi0.flatten(), speed.flatten(), nx=nx, ny=ny, dx=dx, dy=dy, order=order).reshape(shape)

def travel_time(phi0, speed, dx=1., order=2):
    return abs(solveEikonalEquation(phi0, speed, dx, order))

def testing():
    r"""

    **1D Test**

    >>> print np.allclose(computeDistanceFunction((-1., -1., -1., -1., 1., 1., 1., 1.), dx=.5),
    ...                   (-1.75, -1.25, -.75, -0.25, 0.25, 0.75, 1.25, 1.75))
    True

    Small dimensions.

    >>> dx = 1e-10
    >>> print np.allclose(computeDistanceFunction((-1., -1., -1., -1., 1., 1., 1., 1.), dx=dx),
    ...                   np.arange(8) * dx - 3.5 * dx)
    True

    **Bug Fix**

    A 2D test case to test trial values for a pathological case.

    >>> dx = 1.
    >>> dy = 2.
    >>> vbl = -dx * dy / np.sqrt(dx**2 + dy**2) / 2.
    >>> vbr = dx / 2
    >>> vml = dy / 2.
    >>> crossProd = dx * dy
    >>> dsq = dx**2 + dy**2
    >>> top = vbr * dx**2 + vml * dy**2
    >>> sqrt = crossProd**2 *(dsq - (vbr - vml)**2)
    >>> sqrt = np.sqrt(max(sqrt, 0))
    >>> vmr = (top + sqrt) / dsq
    >>> print np.allclose(computeDistanceFunction(((-1., 1., -1.), (1., 1., 1.)), dx=(dx, dy), order=1),
    ...                   ((vbl, vml, vbl), (vbr, vmr, vbr)))
    True

    **Test Extension Field Calculation**

    >>> tmp = 1 / np.sqrt(2)
    >>> phi = np.array([[-1., 1.], [1., 1.]])
    >>> phi, ext =  computeExtensionFields(phi,
    ...                                    [[-1, .5], [2., -1.]],
    ...                                    ext_mask=phi < 0,
    ...                                    dx=1., order=1)
    >>> print np.allclose(phi, ((-tmp / 2, 0.5), (0.5, 0.5 + tmp)))
    True
    >>> print np.allclose(ext, [[1.25, .5], [2., 1.25]])
    True

    >>> phi = np.array(((-1., 1., 1.), (1., 1., 1.), (1., 1., 1.)))
    >>> phi, ext = computeExtensionFields(phi,
    ...                                   ((-1., 2., -1.),
    ...                                    (.5, -1., -1.),
    ...                                    (-1., -1., -1.)),
    ...                                   ext_mask=phi < 0,
    ...                                   order=1)
    >>> v1 = 0.5 + tmp
    >>> v2 = 1.5
    >>> tmp1 = (v1 + v2) / 2 + np.sqrt(2. - (v1 - v2)**2) / 2
    >>> tmp2 = tmp1 + 1 / np.sqrt(2)
    >>> print np.allclose(phi, ((-tmp / 2, 0.5, 1.5),
    ...                         (0.5, 0.5 + tmp, tmp1),
    ...                         (1.5, tmp1, tmp2)))
    True
    >>> print np.allclose(ext, ((1.25, 2., 2.),
    ...                         (.5, 1.25, 1.5456),
    ...                         (.5, 0.9544, 1.25)),
    ...                   rtol = 1e-4)
    True

    **Bug Fix**

    Test case for a bug that occurs when initializing the distance
    variable at the interface. Currently it is assumed that adjacent
    cells that are opposite sign neighbors have perpendicular normal
    vectors. In fact the two closest cells could have opposite
    normals.

    >>> print np.allclose(computeDistanceFunction((-1., 1., -1.)), (-0.5, 0.5, -0.5))
    True

    Testing second order. This example failed with Scikit-fmm.

    >>> phi = ((-1., -1., 1., 1.),
    ...        (-1., -1., 1., 1.),
    ...        (1., 1., 1., 1.),
    ...        (1., 1., 1., 1.))
    >>> answer = ((-1.30473785, -0.5, 0.5, 1.49923009),
    ...           (-0.5, -0.35355339, 0.5, 1.45118446),
    ...           (0.5, 0.5, 0.97140452, 1.76215286),
    ...           (1.49923009, 1.45118446, 1.76215286, 2.33721352))
    >>> print np.allclose(computeDistanceFunction(phi),
    ...                   answer,
    ...                   rtol=1e-9)
    True

    **A test for a bug in both LSMLIB and Scikit-fmm**

    The following test gives different results depending on whether
    LSMLIB_ or Scikit-fmm_ is used. This issue occurs when calculating
    second order accurate distance functions. When a value becomes
    "known" after previously being a "trial" value it updates its
    neighbors' values. In a second order scheme the neighbors one step
    away also need to be updated (if the cell between the new "known"
    cell and the cell required for second order accuracy also happens
    to be "known"), but are not updated in either package.  By luck
    (due to trial values having the same value), the values calculated
    in Scikit-fmm_ for the following example are correct although an
    example that didn't work for Scikit-fmm_ could also be
    constructed.

    >>> phi = computeDistanceFunction([[-1, -1, -1, -1],
    ...                                [ 1,  1, -1, -1],
    ...                                [ 1,  1, -1, -1],
    ...                                [ 1,  1, -1, -1]], order=2)
    >>> phi = computeDistanceFunction(phi, order=2)

    The following values come form Scikit-fmm_.

    >>> answer = [[-0.5,        -0.58578644, -1.08578644, -1.85136395],
    ...           [ 0.5,         0.29289322, -0.58578644, -1.54389939],
    ...           [ 1.30473785,  0.5,        -0.5,        -1.5       ],
    ...           [ 1.49547948,  0.5,        -0.5,        -1.5       ]]

    The 3rd and 7th element are different for LSMLIB_. This is because
    the 15th element is not "known" when the "trial" value for the 7th
    element is calculated. Scikit-fmm_ calculates the values in a
    slightly different order so gets a seemingly better answer, but
    this is just chance.

    >>> print np.allclose(phi, answer, rtol=1e-9)
    False

    **Circle Example**

    Solve the level set equation in two dimensions for a circle.

    The 2D level set equation can be written,

    .. math::

        |\nabla \phi| = 1

    and the boundary condition for a circle is given by, :math:`\phi = 0` at
    :math:`(x - L / 2)^2 + (y - L / 2)^2 = (L / 4)^2`.

    The solution to this problem will be demonstrated in the following
    script. Firstly, setup the parameters.

    >>> def mesh(nx=1, ny=1, dx=1., dy=1.):
    ...     y, x = np.mgrid[0:nx,0:ny]
    ...     x = x * dx + dx / 2
    ...     y = y * dy + dy / 2
    ...     return x, y

    >>> dx = 1.
    >>> N = 11
    >>> L = N * dx
    >>> x, y = mesh(nx=N, ny=N, dx=dx, dy=dx)
    >>> phi = -np.ones(N * N, 'd')
    >>> phi[(x.flatten() - L / 2.)**2 + (y.flatten() - L / 2.)**2 < (L / 4.)**2] = 1.
    >>> phi = np.reshape(phi, (N, N))
    >>> phi = computeDistanceFunction(phi, dx=dx, order=1).flatten()

    >>> dX = dx / 2.
    >>> m1 = dX * dX / np.sqrt(dX**2 + dX**2)
    >>> def evalCell(phix, phiy, dx):
    ...     aa = dx**2 + dx**2
    ...     bb = -2 * ( phix * dx**2 + phiy * dx**2)
    ...     cc = dx**2 * phix**2 + dx**2 * phiy**2 - dx**2 * dx**2
    ...     sqr = np.sqrt(bb**2 - 4. * aa * cc)
    ...     return ((-bb - sqr) / 2. / aa,  (-bb + sqr) / 2. / aa)
    >>> v1 = evalCell(-dX, -m1, dx)[0]
    >>> v2 = evalCell(-m1, -dX, dx)[0]
    >>> v3 = evalCell(m1,  m1,  dx)[1]
    >>> v4 = evalCell(v3, dX, dx)[1]
    >>> v5 = evalCell(dX, v3, dx)[1]
    >>> MASK = -1000.
    >>> trialValues = np.array((
    ...     MASK,  MASK, MASK, MASK, MASK, MASK, MASK, MASK, MASK, MASK, MASK,
    ...     MASK,  MASK, MASK, MASK,-3*dX,-3*dX,-3*dX, MASK, MASK, MASK, MASK,
    ...     MASK,  MASK, MASK,   v1,  -dX,  -dX,  -dX,   v1, MASK, MASK, MASK,
    ...     MASK,  MASK,   v2,  -m1,   m1,   dX,   m1,  -m1,   v2, MASK, MASK,
    ...     MASK, -dX*3,  -dX,   m1,   v3,   v4,   v3,   m1,  -dX,-dX*3, MASK,
    ...     MASK, -dX*3,  -dX,   dX,   v5, MASK,   v5,   dX,  -dX,-dX*3, MASK,
    ...     MASK, -dX*3,  -dX,   m1,   v3,   v4,   v3,   m1,  -dX,-dX*3, MASK,
    ...     MASK,  MASK,   v2,  -m1,   m1,   dX,   m1,  -m1,   v2, MASK, MASK,
    ...     MASK,  MASK, MASK,   v1,  -dX,  -dX,  -dX,   v1, MASK, MASK, MASK,
    ...     MASK,  MASK, MASK, MASK,-3*dX,-3*dX,-3*dX, MASK, MASK, MASK, MASK,
    ...     MASK,  MASK, MASK, MASK, MASK, MASK, MASK, MASK, MASK, MASK, MASK), 'd')

    >>> phi[trialValues == MASK] = MASK
    >>> print np.allclose(phi, trialValues)
    True

    **Square Example**

    Here we solve the level set equation in two dimensions for a square. The equation is
    given by:

    .. math::

       |\nabla \phi| &= 1 \\
       \phi &= 0 \qquad \text{at} \qquad \begin{cases}
           x = \left( L / 3, 2 L / 3 \right)
           & \text{for $L / 3 \le y \le 2 L / 3$} \\
           y = \left( L / 3, 2 L / 3 \right)
           & \text{for $L / 3 \le x \le 2 L / 3$}
       \end{cases}

    >>> dx = 0.5
    >>> dy = 2.
    >>> nx = 5
    >>> ny = 5
    >>> Lx = nx * dx
    >>> Ly = ny * dy

    >>> x, y = mesh(nx=nx, ny=ny, dx=dx, dy=dy)
    >>> x = x.flatten()
    >>> y = y.flatten()
    >>> phi = -np.ones(nx * ny, 'd')
    >>> phi[((Lx / 3. < x) & (x < 2. * Lx / 3.)) & ((Ly / 3. < y) & (y < 2. * Ly / 3))] = 1.
    >>> phi = np.reshape(phi, (nx, ny))
    >>> phi = computeDistanceFunction(phi, dx=(dy, dx), order=1).flatten()

    >>> def evalCell(phix, phiy, dx, dy):
    ...     aa = dy**2 + dx**2
    ...     bb = -2 * ( phix * dy**2 + phiy * dx**2)
    ...     cc = dy**2 * phix**2 + dx**2 * phiy**2 - dx**2 * dy**2
    ...     sqr = np.sqrt(bb**2 - 4. * aa * cc)
    ...     return ((-bb - sqr) / 2. / aa,  (-bb + sqr) / 2. / aa)
    >>> val = evalCell(-dy / 2., -dx / 2., dx, dy)[0]
    >>> v1 = evalCell(val, -3. * dx / 2., dx, dy)[0]
    >>> v2 = evalCell(-3. * dy / 2., val, dx, dy)[0]
    >>> v3 = evalCell(v2, v1, dx, dy)[0]
    >>> v4 = dx * dy / np.sqrt(dx**2 + dy**2) / 2
    >>> arr = np.array((
    ...     v3           , v2      , -3. * dy / 2.   , v2      , v3,
    ...     v1           , val     , -dy / 2.        , val     , v1           ,
    ...     -3. * dx / 2., -dx / 2., v4              , -dx / 2., -3. * dx / 2.,
    ...     v1           , val     , -dy / 2.        , val     , v1           ,
    ...     v3           , v2      , -3. * dy / 2.   , v2      , v3           ))
    >>> print np.allclose(arr, phi)
    True

    **Assertion Errors**

    >>> computeDistanceFunction([[-1, 1],[1, 1]], dx=(1, 2, 3))
    Traceback (most recent call last):
      ...
    ValueError: dx must be of length len(phi.shape)
    >>> computeExtensionFields([[-1, 1],[1, 1]], speed=[1, 1])
    Traceback (most recent call last):
      ...
    ValueError: phi and speed must have the same shape

    **Test for 1D equality between `distance` and `travel_time`**

    >>> phi = np.arange(-5, 5) + 0.499
    >>> d = distance(phi)
    >>> t = travel_time(phi, speed=np.ones_like(phi))
    >>> ##np.testing.assert_allclose(t, np.abs(d))

    **Tests taken from FiPy**

    >>> phi = np.array(((-1, -1, 1, 1),
    ...                 (-1, -1, 1, 1),
    ...                 (1, 1, 1, 1),
    ...                 (1, 1, 1, 1)))
    >>> o1 = distance(phi, order=1)
    >>> dw_o1 =   [[-1.20710678, -0.5,         0.5,         1.5],
    ...            [-0.5,        -0.35355339,  0.5,         1.5],
    ...            [ 0.5,         0.5,         1.20710678,  2.04532893],
    ...            [ 1.5,         1.5,         2.04532893,  2.75243571]]
    >>> np.testing.assert_allclose(o1, dw_o1)

    >>> phi = np.array(((-1, -1, 1, 1),
    ...                 (-1, -1, 1, 1),
    ...                 (1, 1, 1, 1),
    ...                 (1, 1, 1, 1)))
    >>> o1 = travel_time(phi, np.ones_like(phi), order=1)
    >>> dw_o1 =   [[-1.20710678, -0.5,         0.5,         1.5],
    ...            [-0.5,        -0.35355339,  0.5,         1.5],
    ...            [ 0.5,         0.5,         1.20710678,  2.04532893],
    ...            [ 1.5,         1.5,         2.04532893,  2.75243571]]
    >>> ##np.testing.assert_allclose(o1, np.abs(dw_o1))

    >>> phi = np.array(((-1, -1, 1, 1),
    ...                (-1, -1, 1, 1),
    ...                (1, 1, 1, 1),
    ...                (1, 1, 1, 1)))
    >>> o2 = distance(phi)
    >>> dw_o2 = [[-1.30473785,  -0.5,          0.5,         1.49923009],
    ...          [-0.5,        -0.35355339,    0.5,         1.45118446],
    ...          [ 0.5,         0.5,         0.97140452,  1.76215286],
    ...          [ 1.49923009,  1.45118446,  1.76215286,  2.33721352]]

    >>> np.testing.assert_allclose(o2, dw_o2)
    >>> phi = np.array(((-1, -1, 1, 1),
    ...                (-1, -1, 1, 1),
    ...                (1, 1, 1, 1),
    ...                (1, 1, 1, 1)))
    >>> o2 = travel_time(phi, np.ones_like(phi))
    >>> dw_o2 = [[-1.30473785,  -0.5,          0.5,         1.49923009],
    ...          [-0.5,        -0.35355339,    0.5,         1.45118446],
    ...          [ 0.5,         0.5,         0.97140452,  1.76215286],
    ...          [ 1.49923009,  1.45118446,  1.76215286,  2.33721352]]
    >>> ##np.testing.assert_allclose(o2, np.abs(dw_o2))

    >>> distance([-1,1], order=0)
    Traceback (most recent call last):
      ...
    ValueError: order must be 1 or 2

    >>> distance([-1,1], order=3)
    Traceback (most recent call last):
      ...
    ValueError: order must be 1 or 2

    **Extension velocity tests**

    Test 1d extension constant.

    >>> phi =   [-1,-1,-1,1,1,1]
    >>> speed = [1,1,1,1,1,1]
    >>> d, f_ext  = extension_velocities(phi, speed)
    >>> np.testing.assert_allclose(speed, f_ext)

    Test the 1D extension block.

    >>> phi =   np.ones(10)
    >>> phi[0] =- 1
    >>> speed = np.ones(10)
    >>> speed[0:3] = 5
    >>> d, f_ext  = extension_velocities(phi, speed)
    >>> np.testing.assert_allclose(f_ext, 5)

    Test that a uniform speed value is preserved.

    >>> N     = 50
    >>> X, Y  = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> r     = 0.25
    >>> dx    = 2.0 / (N - 1)
    >>> phi   = (X) ** 2 + (Y) ** 2 - r ** 2
    >>> speed = np.ones_like(phi)
    >>> d, f_ext = extension_velocities(phi, speed, dx)
    >>> np.testing.assert_allclose(f_ext, 1.0)

    Constant value marchout test

    >>> speed[abs(Y)<0.3] = 10.0
    >>> d, f_ext = extension_velocities(phi, speed, dx)
    >>> np.testing.assert_allclose(f_ext, 10.0)

    Test distance from extenstion

    >>> speed = np.ones_like(phi)
    >>> d, f_ext = extension_velocities(phi, speed, dx)
    >>> d2 = distance(phi, dx)
    >>> np.testing.assert_allclose(d, d2)

    Test for extension velocity bug

    >>> N     = 150
    >>> X, Y  = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> r     = 0.5
    >>> dx    = 2.0 / (N - 1)
    >>> phi   = (X) ** 2 + (Y) ** 2 - r ** 2
    >>> speed = np.ones_like(phi)
    >>> speed[X>0.25] = 3.0
    >>> d2, f_ext = extension_velocities(phi, speed, dx)

    >>> assert (f_ext <= 3.0000001).all()
    >>> assert (f_ext >= 1).all()

    >>> np.testing.assert_almost_equal(f_ext[137, 95], 1, 3)
    >>> np.testing.assert_almost_equal(f_ext[103, 78], 1, 2)
    >>> np.testing.assert_almost_equal(f_ext[72, 100], 3, 3)
    >>> np.testing.assert_almost_equal(f_ext[72, 86], 3, 3)
    >>> np.testing.assert_almost_equal(f_ext[110, 121], 3, 3)

    Simple two point tests

    >>> np.testing.assert_array_equal(distance([-1, 1]),
    ...                                        [-0.5, 0.5])
    >>> np.testing.assert_allclose(distance([-1, -1, -1, 1, 1, 1]),
    ...                                     [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    >>> np.testing.assert_allclose(distance([1, 1, 1, -1, -1, -1]),
    ...                                     [2.5, 1.5, 0.5, -0.5, -1.5, -2.5])

    Three point test case

    >>> np.testing.assert_array_almost_equal(distance([-1, 0, 1]),           [-1, 0, 1])
    >>> np.testing.assert_array_almost_equal(distance([-1, 0, 1], dx=[2]),   [-2, 0, 2])
    >>> np.testing.assert_array_almost_equal(distance([-1, 0, 1], dx=2),     [-2, 0, 2])
    >>> np.testing.assert_array_almost_equal(distance([-1, 0, 1], dx=2.0),   [-2, 0, 2])
    >>> np.testing.assert_array_equal(travel_time([1, 0, -1], [1, 1, 1]),
    ...                                           [1, 0, 1])
    >>> np.testing.assert_array_equal(travel_time([-1, 0, 1], [1, 1, 1]),
    ...                                           [1, 0, 1])
    >>> ##np.testing.assert_array_almost_equal(travel_time([1, 0, -1], [1, 1, 1], dx=2), [2, 0, 2])
    >>> ##np.testing.assert_array_equal(travel_time([1, 0, -1], [1, 1, 1], dx=[2]), [2, 0, 2])
    >>> ##np.testing.assert_array_equal(travel_time([1, 0, -1], [1, 1, 1], dx=2.0), [2, 0, 2])

    Travel time tests 1

    >>> ##np.testing.assert_allclose(travel_time([0, 1, 1, 1, 1], [2, 2, 2, 2, 2]), [0, 0.5, 1.0, 1.5, 2.0])
    >>> ##np.testing.assert_array_equal(travel_time([1, 0, -1], [2, 2, 2]), [0.5, 0, 0.5])

    Travel time tests 2

    >>> ##phi   = [1, 1, 1, -1, -1, -1]
    >>> ##t     = travel_time(phi, np.ones_like(phi))
    >>> ##exact = [2.5, 1.5, 0.5, 0.5, 1.5, 2.5]
    >>> ##np.testing.assert_allclose(t, exact)

    Travel time tests 3

    >>> ##phi   = [-1, -1, -1, 1, 1, 1]
    >>> ##t     = travel_time(phi, np.ones_like(phi))
    >>> ##exact = [2.5, 1.5, 0.5, 0.5, 1.5, 2.5]
    >>> ##np.testing.assert_allclose(t, exact)

    Corner case

    >>> np.testing.assert_array_almost_equal(distance([0, 0]), [0, 0])
    >>> ##np.testing.assert_array_equal(travel_time([0, 0], [1, 1]), [0, 0])

    Test zero

    >>> distance([1, 0, 1, 1], 0)
    Traceback (most recent call last):
      ...
    ValueError: dx must be greater than zero

    Test dx shape

    >>> distance([0, 0, 1, 0, 0], [0, 0, 1, 0, 0])
    Traceback (most recent call last):
      ...
    ValueError: dx must be of length len(phi.shape)

    Test for small speeds

    Test catching speeds which are too small. Speeds less than the
    machine epsilon are masked off to avoid an overflow.

    >>> ##t = travel_time([-1, -1, 0, 1, 1], [1, 1, 1, 1, 0])
    >>> ##assert isinstance(t, np.ma.MaskedArray)
    >>> ##np.testing.assert_array_equal(t.data[:-1], [2, 1, 0, 1])
    >>> ##np.testing.assert_array_equal(t.mask, [False, False, False, False, True])

    >>> ##t2 = travel_time([-1, -1, 0, 1, 1], [1, 1, 1, 1, 1e-300])
    >>> ##np.testing.assert_array_equal(t, t2)

    Mask test

    Test that when the mask cuts off the solution, the cut off points
    are also masked.

    >>> ##ma    = np.ma.MaskedArray([1, 1, 1, 0], [False, True, False, False])
    >>> ##d     = distance(ma)
    >>> ##exact = np.ma.MaskedArray([0, 0, 1, 0], [True, True, False, False])
    >>> ##np.testing.assert_array_equal(d.mask, exact.mask)
    >>> ##np.testing.assert_array_equal(d, exact)

    Circular level set

    >>> N     = 50
    >>> X, Y  = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> r     = 0.5
    >>> dx    = 2.0 / (N - 1)
    >>> phi   = (X) ** 2 + (Y) ** 2 - r ** 2
    >>> d     = distance(phi, dx)
    >>> exact = np.sqrt(X ** 2 + Y ** 2) - r
    >>> np.testing.assert_allclose(d, exact, atol=dx)

    Planar level set

    >>> N         = 50
    >>> X, Y      = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> dx        = 2.0 / (N - 1)
    >>> phi       = np.ones_like(X)
    >>> phi[0, :] = -1
    >>> d         = distance(phi, dx)
    >>> exact     = Y + 1 - dx / 2.0
    >>> np.testing.assert_allclose(d, exact)

    Masked input

    >>> N         = 50
    >>> X, Y      = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> dx        = 2.0 / (N - 1)
    >>> phi       = np.ones_like(X)
    >>> phi[0, 0] = -1
    >>> mask      = np.logical_and(abs(X) < 0.25, abs(Y) < 0.25)
    >>> mphi      = np.ma.MaskedArray(phi.copy(), mask)
    >>> d0        = distance(phi, dx)
    >>> d         = distance(mphi, dx)
    >>> d0[mask]  = 0
    >>> d[mask]   = 0
    >>> shadow    = d0 - d
    >>> bsh       = abs(shadow) > 0.001
    >>> diff      = (bsh).sum()

    >>> ##assert diff > 635 and diff < 645

    Test eikonal solution

    >>> ##N     = 50
    >>> ##X, Y  = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> ##r     = 0.5
    >>> ##dx    = 2.0 / (N - 1)
    >>> ##phi   = (X) ** 2 + (Y) ** 2 - r ** 2
    >>> ##speed = np.ones_like(phi) * 2
    >>> ##t     = travel_time(phi, speed, dx)
    >>> ##exact = 0.5 * np.abs(np.sqrt(X ** 2 + Y ** 2) - 0.5)

    >>> ##np.testing.assert_allclose(t, exact, atol=dx)

    Test 1d

    >>> N          = 100
    >>> X          = np.linspace(-1.0, 1.0, N)
    >>> dx         = 2.0 / (N - 1)
    >>> phi        = np.zeros_like(X)
    >>> phi[X < 0] = -1
    >>> phi[X > 0] = 1
    >>> d          = distance(phi, dx)

    >>> np.testing.assert_allclose(d, X)

    Test 3d

    >>> N            = 15
    >>> X            = np.linspace(-1, 1, N)
    >>> Y            = np.linspace(-1, 1, N)
    >>> Z            = np.linspace(-1, 1, N)
    >>> phi          = np.ones((N, N, N))
    >>> phi[0, 0, 0] = -1.0
    >>> dx           = 2.0 / (N - 1)
    >>> d            = distance(phi, dx)
    >>> ##Traceback (most recent call last):
    >>> ##  ...
    >>> ##Exception: 3D meshes not yet implemented
    >>> exact        = np.sqrt((X + 1) ** 2 +
    ...                        (Y + 1)[:, np.newaxis] ** 2 +
    ...                        (Z + 1)[:, np.newaxis, np.newaxis] ** 2)

    >>> ## np.testing.assert_allclose(d, exact, atol=dx)

    Test default dx

    >>> ##N     = 50
    >>> ##X, Y  = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> ##r     = 0.5
    >>> ##phi   = (X) ** 2 + (Y) ** 2 - r ** 2
    >>> ##speed = np.ones_like(phi) * 2
    >>> ##out = travel_time(phi, speed)

    Test non-square grid and dx different in different directions

    >>> N      = 50
    >>> NX, NY = N, 5 * N
    >>> X, Y   = np.meshgrid(np.linspace(-1, 1, NY), np.linspace(-1, 1, NX))
    >>> r      = 0.5
    >>> phi    = X ** 2 + Y ** 2 - r ** 2
    >>> dx     = [2.0 / (NX - 1), 2.0 / (NY - 1)]
    >>> d      = distance(phi, dx, order=1)
    >>> exact  = np.sqrt(X ** 2 + Y ** 2) - r

    >>> np.testing.assert_allclose(d, exact, atol=1.1*max(dx))

    Shape mismatch test

    >>> travel_time([-1, 1], [2])
    Traceback (most recent call last):
      ...
    ValueError: phi and speed must have the same shape

    Speed wrong type test

    >>> travel_time([0, 0, 1, 1], 2)
    Traceback (most recent call last):
      ...
    ValueError: phi and speed must have the same shape

    dx mismatch test

    >>> travel_time([-1, 1], [2, 2], [2, 2, 2, 2])
    Traceback (most recent call last):
      ...
    ValueError: dx must be of length len(phi.shape)

    Test c error handling. Check array type test

    >>> distance(np.array(["a", "b"]))
    Traceback (most recent call last):
      ...
    ValueError: could not convert string to float: a


    **Testing new functions**

    Test ``get3dShape``

    >>> a = np.zeros((2,5))
    >>> exact = [5, 2, 1, 1.0, 1.0, 0.0, (2, 5),
    ...         np.array([[ 0.,  0.,  0.,  0.,  0.],
    ...                [ 0.,  0.,  0.,  0.,  0.]])]
    >>> print(get3dShape(a, 1., 2))
    (5, 2, 1, 1.0, 1.0, 0.0, (2, 5), array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]]))

    >>> b = np.random.rand(4,8,20)
    >>> nx,ny,nz,dx,dy,dz,_,c = get3dShape(b,.5,2)
    >>> print(nx,ny,nz)
    (20, 8, 4)
    >>> print(dx, dy, dz)
    (0.5, 0.5, 0.5)
    >>> np.testing.assert_allclose(b, c)

    >>> phi1 = np.array([[[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.]],
    ...                   [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.]],
    ...                   [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.]],
    ...                   [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.]],
    ...                   [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.]],
    ...                   [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.],
    ...                   [ 1.,  1., 1.,  1., -1., -1.]]])
    >>> phi1 = computeDistanceFunction(phi1, dx=1.)
    >>> ##print(phi1)
    >>> lim = np.array([0,5,0,5,0,5], dtype=np.intc)
    >>> fblim = np.array([2,3,2,3,2,3], dtype=np.intc)
    >>> phi2 = np.array([[[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [10., 10.,  8.,  0., -1., -1.],
    ...                   [11., 11., 10.,  0., -1., -1.],
    ...                   [ 5.,  3., 1.,  0., -10., -15.],
    ...                   [ 5.,  3., 1.,  0., -10., -15.]],
    ...                  [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [10., 10.,  8.,  0., -1., -1.],
    ...                   [11., 11., 10.,  0., -1., -1.],
    ...                   [ 5.,  3., 1.,  0., -10., -15.],
    ...                   [ 5.,  3., 1.,  0., -10., -15.]],
    ...                  [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [10., 10.,  8.,  0., -1., -1.],
    ...                   [11., 11., 10.,  0., -1., -1.],
    ...                   [ 5.,  3., 1.,  0., -10., -15.],
    ...                   [ 5.,  3., 1.,  0., -10., -15.]],
    ...                  [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 3.,  2.,  1.,  0., -0.1, -1.],
    ...                   [ 3.,  2.,  0.,  -1., -1., -1.],
    ...                   [ 3.,  2.,  0.,  -1., -1.9, -6.],
    ...                   [ 3.,  2.,  0.,  -1., -1.9, -6.]],
    ...                  [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 3.,  2.,  1.,  0., -0.1, -1.],
    ...                   [ 3.,  2.,  0.,  -1., -1., -1.],
    ...                   [ 3.,  2.,  0.,  -1., -1.9, -6.],
    ...                   [ 3.,  2.,  0.,  -1., -1.9, -6.]],
    ...                  [[-1., -1., -1., -1., -1., -1.],
    ...                   [-1., -1., -1., -1., -1., -1.],
    ...                   [ 3.,  2.,  1.,  0., -0.1, -1.],
    ...                   [ 3.,  2.,  0.,  -1., -1., -1.],
    ...                   [ 3.,  2.,  0.,  -1., -1.9, -6.],
    ...                   [ 3.,  2.,  0.,  -1., -1.9, -6.]]])
    >>> phi2 = computeDistanceFunction(phi2, dx=0.5)
    >>> ##print(phi2)

    ``centralGradOrder4``

    >>> ##phi_x, phi_y, phi_z = centralGradOrder4(phi1, lim, lim, fblim)
    >>> phi_x, phi_y, phi_z = pythonisedfns.gradPhi(phi1)
    >>> print(phi_x.shape)
    (6, 6, 6)
    >>> print(phi_y.shape)
    (6, 6, 6)
    >>> print(phi_z.shape)
    (6, 6, 6)
    >>> ##phi_x, phi_y, phi_z = centralGradOrder4(phi2, lim, lim, fblim, dx=0.5)
    >>> phi2_x, phi2_y, phi2_z = pythonisedfns.gradPhi(phi2, dx=0.5, dy=0.5, dz=0.5)
    >>> isSmall = abs(phi2_z) <= 0.6
    >>> print(isSmall.all())
    True

    ``surfaceAreaZeroLevelSet``

    This doesn't work - there is something up with the pointers as the surface area is passed by reference in ``lsmlib.pyx`` to the C function, but not updated by it. I strongly suspect it has something to do with the way the C function is called - I can change the ``#define`` statement in ``lsm_geometry3d.h`` so that it has a different name (e.g. ``lsm3dComputeSignedUnitNormal`` rather than ``lsm3dcomputesignedunitnormal_``), recompile and it will still run. If you look at the examples (e.g. ``curvature_model3d.c``), the C functions are all called using the CAPITAL_NAMES rather than the ones in the ``#define`` statements.

    >>> ##print(surfaceAreaZeroLevelSet(phi1,phix1,phix1,phix1,lim,lim,fblim))
    >>> norm_x, norm_y, norm_z = pythonisedfns.signedUnitNormal(phi1, phi_x, phi_y, phi_z)
    >>> print(pythonisedfns.altSurfaceAreaLevelSet(phi1,phi_x,phi_y,phi_z, norm_x, norm_y, norm_z))
    14.0910950538

    >>> ##print(surfaceAreaZeroLevelSet(phi2,phix2,phix2,phix2,lim,lim,fblim, dx=0.5, epsilon=2.))
    >>> norm2_x, norm2_y, norm2_z = pythonisedfns.signedUnitNormal(phi2, phi2_x, phi2_y, phi2_z, dx=0.5, dy=0.5, dz=0.5)
    >>> print(pythonisedfns.altSurfaceAreaLevelSet(phi2, phi2_x,phi2_y,phi2_z, norm2_x, norm2_y, norm2_z, np.ones_like(phi2, dtype=bool), dx=0.5, dy=0.5, dz=0.5, epsilon=2.))
    1.55662111361

    ``computeMeanCurvatureLocal``

    This doesn't work either, but does at least return different answers for the two tests, implying it's doing something.

    >>> kappa = np.zeros_like(phi1, dtype=np.float64)
    >>> ##isinstance(kappa[1,2,3], np.float64)
    >>> ##True
    >>> ##narrow_band = np.chararray(phi1.shape)
    >>> ##narrow_band[:] = '0'
    >>> ##narrow_band = narrow_band.tostring()
    >>> ##narrow_band = '0'
    >>> ##mark_fb = narrow_band
    >>> ##isinstance(lim[5], np.intc)
    >>> ##True
    >>> ##a = np.linspace(0,3,4, dtype=np.intc)
    >>> ##isinstance(a[3], np.intc)
    >>> ##True
    >>> ##print(computeMeanCurvatureLocal(phi1, phix1, phix1, phix1, kappa, phix1, lim, a, a, a, narrow_band, mark_fb, lim, lim, lim, lim))
    >>> kappa = pythonisedfns.meanCurvature(phi1, phi_x, phi_y, phi_z, fblim)
    >>> checkCurvy = np.ones_like(kappa, dtype=bool)
    >>> checkCurvy[:2,:,:] = (kappa[:2,:,:] == 0.)
    >>> checkCurvy[3:,:,:] = (kappa[3:,:,:] == 0.)
    >>> checkCurvy[2:4,2:4,2:4] = (kappa[2:4,2:4,2:4] != 0.)
    >>> print(checkCurvy.all())
    True

    >>> ##print(computeMeanCurvatureLocal(phi2, phix2, phix2, phix2, kappa,  phix2, lim, a, a, a, narrow_band, mark_fb, lim, lim, lim, lim, dx=0.5))
    >>> kappa = pythonisedfns.meanCurvature(phi2, phi2_x, phi2_y, phi2_z, fblim, dx=0.5, dy=0.5, dz=0.5)
    >>> checkCurvy = np.ones_like(kappa, dtype=bool)
    >>> checkCurvy[:2,:,:] = (kappa[:2,:,:] == 0.)
    >>> checkCurvy[3:,:,:] = (kappa[3:,:,:] == 0.)
    >>> checkCurvy[2:4,2:4,2:4] = (kappa[2:4,2:4,2:4] != 0.)
    >>> print(checkCurvy.all())
    True


    ``computeMeanCurvature``

    >>> ##print(computeMeanCurvature(phi1, phix1, phix1, phix1, kappa, phix1, lim, lim, lim, fblim))

    >>> ##print(computeMeanCurvature(phi2, phix2, phix2, phix2, kappa, phix2, lim, lim, lim, fblim, dx=0.5))

    ``computeGaussianCurvature``

    >>> ##print(computeGaussianCurvature(phi1, phix1, phix1, phix1, kappa, phix1, lim, lim, lim, fblim))

    >>> ##print(computeGaussianCurvature(phi2, phix2, phix2, phix2, kappa, phix2, lim, lim, lim, fblim, dx=0.5))


    ``computeSignedUnitNormal``

    This doesn't work either, but does at least return different answers for the two tests, implying it's doing something.

    >>> ##print(computeSignedUnitNormal(phi1,phix1, phix1, phix1,#lim, lim, lim, fblim))
    >>> norm_x, norm_y, norm_z = pythonisedfns.signedUnitNormal(phi1, phi_x, phi_y, phi_z)
    >>> checkSign = np.ones_like(kappa, dtype=bool)
    >>> checkSign = (np.sign(phi[:]) != np.sign(norm_x[:]))
    >>> print(checkSign)
    True

    >>> ##print(computeSignedUnitNormal(phi2,phix2, phix2, phix2,#lim, lim, lim, fblim, dx=0.5))
    >>> norm2_x, norm2_y, norm2_z = pythonisedfns.signedUnitNormal(phi2, phi2_x, phi2_y, phi2_z, dx=0.5, dy=0.5, dz=0.5)
    >>> checkSign = np.ones_like(kappa, dtype=bool)
    >>> checkSign = (np.sign(phi2[:]) != np.sign(norm2_x[:]))
    >>> checkSign[1,0,0] = True
    >>> print(checkSign.all())
    True

    ``strainRate``

    Define velocities:
    >>> u = np.ones_like(phi1)
    >>> u[4:,:,:] *= -1.
    >>> v = np.zeros_like(phi1)
    >>> w = np.zeros_like(phi1)

    >>> ##print(pythonisedfns.strainRate(phi1, phi_x, phi_y, phi_z, u, v, w))

    >>> ##print(pythonisedfns.strainRate(phi2, phi2_x, phi2_y, phi2_z, u, v, w, dx=0.5, dy=0.5, dz=0.5))

    ``laminarFlameSpeed``

    >>> marksteinLength = 0.2
    >>> sL0 = 3.

    >>> ##print(pythonisedfns.laminarFlameSpeed(phi1, sL0, marksteinLength, u, v, w, fblim))

    >>> ##print(pythonisedfns.laminarFlameSpeed(phi2, sL0, marksteinLength, u, v, w, fblim))

    **Test level set functions**

    ``locateLS1d``

    >>> phi1d = np.array([-1.,-1.,-1.,-0.7,1.,1.,1.])
    >>> phi1d = computeDistanceFunction(phi1d, dx=1.)
    >>> zeros, alpha = lsmfns.locateLS1d(phi1d)
    >>> ##print(phi1d)

    >>> ##print(zeros)

    >>> ##print(alpha)

    ``locateLS2d``

    >>> phi2d = np.array([[-1.,-1., -0.2, 1.,1.],
    ...                   [-1.,-1., -0.1,0.5,1.],
    ...                   [-1.,-1., 0.1,  1.,1.],
    ...                   [-1.,-0.5, 0.7,  1.,1.]])
    >>> phi2d = computeDistanceFunction(phi2d, dx=0.5)
    >>> phi2d_y, phi2d_x = np.gradient(phi2d, 0.5, 0.5)
    >>> norm_x, norm_y= pythonisedfns.signedUnitNormal2d(phi2d, phi2d_x, phi2d_y, dx=0.5, dy=0.5)
    >>> ##print(norm_x)

    >>> ##print(norm_y)

    >>> zeros, alpha = lsmfns.locateLS2d(phi2d, norm_x, norm_y, dx=0.5, dy=0.5)
    >>> ##print(phi2d)

    >>> ##print(zeros)

    >>> ##print(alpha)

    **Flame evolution tests**

    Does a circle stay circular?
    Evolving it here is non-trivial....

    >>> N     = 20
    >>> X, Y  = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    >>> r     = 0.5
    >>> dx    = 2.0 / (N - 1)
    >>> phi   = (X) ** 2 + (Y) ** 2 - r ** 2
    >>> phi   = computeDistanceFunction(phi, dx)
    >>> sL0 = 1.
    >>> marksteinLength = 0.01
    >>> u = np.zeros_like(phi)
    >>> iblims = np.array([2,N-3, 2, N-3])
    >>> sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u, iblims, dx=dx, dy=dx)
    >>> print(sL)

    """

    pass

distance = computeDistanceFunction
extension_velocities = computeExtensionFields

def test():
    r"""
    Run all the doctests available.
    """
    import doctest
    import pylsmlib
    doctest.testmod(pylsmlib)
