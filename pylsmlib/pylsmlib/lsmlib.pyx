import cython
import numpy as np
from numpy.core import intc
from numpy.compat import asbytes
from numpy import int, double, int32
from lsmlib cimport computeDistanceFunction3d
from lsmlib cimport computeExtensionFields3d
from lsmlib cimport solveEikonalEquation3d
from lsmlib cimport lsm3dcomputesignedunitnormal
from lsmlib cimport lsm3dcomputemeancurvatureorder2local
from libc.stdlib cimport malloc, free

def lsm3dcomputesignedunitnormal_(np.ndarray[double, ndim=1] phi,
                                  np.ndarray[double, ndim=1] phi_x,
                                  np.ndarray[double, ndim=1] phi_y,
                                  np.ndarray[double, ndim=1] phi_z,
                                  ilo_normal_gb, ihi_normal_gb, jlo_normal_gb,
                                  jhi_normal_gb ,klo_normal_gb, khi_normal_gb,
                                  ilo_grad_phi_gb, ihi_grad_phi_gb, jlo_grad_phi_gb,
                                  jhi_grad_phi_gb, klo_grad_phi_gb, khi_grad_phi_gb,
                                  ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                                  jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                                  ilo_fb, ihi_fb, jlo_fb, jhi_fb, klo_fb, khi_fb,
                                  nx=1, ny=1, nz=1, dx=1., dy=1., dz=1.):
"""
 * LSM3D_COMPUTE_SIGNED_UNIT_NORMAL() computes the signed unit normal
 * vector (sgn(phi)*normal) to the interface from \f$ \nabla \phi \f$
 * using the following smoothed sgn function
 *
 * \f[
 *
 *   sgn(\phi) = \phi / \sqrt{ \phi^2 + |\nabla \phi|^2 * dx^2 }
 *
 * \f]
 *
 * Arguments:
 *  - normal_* (out):     components of unit normal vector
 *  - phi_* (in):         components of \f$ \nabla \phi \f$
 *  - phi (in):           level set function
 *  - dx, dy, dz (in):    grid spacing
 *  - *_gb (in):          index range for ghostbox
 *  - *_fb (in):          index range for fillbox
 *
 * Return value:          none
 *
 * NOTES:
 * - When \f$ | \nabla \phi | \f$ is close to zero, the unit normal is
 *   arbitrarily set to be (1.0, 0.0, 0.0).
 *
"""
    cdef np.ndarray[double, ndim=1] normal_x = np.zeros((nx * ny * nz,))
    cdef np.ndarray[double, ndim=1] normal_y = np.zeros((nx * ny * nz,))
    cdef np.ndarray[double, ndim=1] normal_z = np.zeros((nx * ny * nz,))
    cdef int* _ilo_normal_gb = &ilo_normal_gb
    cdef int* _ihi_normal_gb = &ihi_normal_gb
    cdef int* _jlo_normal_gb = &jlo_normal_gb
    cdef int* _jhi_normal_gb = &jhi_normal_gb
    cdef int* _klo_normal_gb = &klo_normal_gb
    cdef int* _khi_normal_gb = &khi_normal_gb
    cdef int* _ilo_grad_phi_gb = &ilo_grad_phi_gb
    cdef int* _ihi_grad_phi_gb = &ihi_grad_phi_gb
    cdef int* _jlo_grad_phi_gb = &jlo_grad_phi_gb
    cdef int* _jhi_grad_phi_gb = &jhi_grad_phi_gb
    cdef int* _klo_grad_phi_gb = &klo_grad_phi_gb
    cdef int* _khi_grad_phi_gb = &khi_grad_phi_gb
    cdef int* _ilo_phi_gb = &ilo_phi_gb
    cdef int* _ihi_phi_gb = &ihi_phi_gb
    cdef int* _jlo_phi_gb = &jlo_phi_gb
    cdef int* _jhi_phi_gb = &jhi_phi_gb
    cdef int* _klo_phi_gb = &klo_phi_gb
    cdef int* _khi_phi_gb = &khi_phi_gb
    cdef int* _ilo_fb = &ilo_fb
    cdef int* _ihi_fb = &ihi_fb
    cdef int* _jlo_fb = &jlo_fb
    cdef int* _jhi_fb = &jhi_fb
    cdef int* _klo_fb = &klo_fb
    cdef int* _khi_fb = &khi_fb
    cdef double* _dx = &dx
    cdef double* _dy = &dy
    cdef double* _dz = &dz

    lsm3dcomputesignedunitnormal(<double *> normal_x, <double *> normal_y, <double *> normal_z,
	       <int *> _ilo_normal_gb.data, <int *> _ihi_normal_gb.data,
         <int *> _jlo_normal_gb.data,  <int *> _jhi_normal_gb.data,
         <int *> _klo_normal_gb.data, <int *> _khi_normal_gb.data,
    	   <double *> phi_x.data, <double *> phi_y.data, <double *> phi_z.data,
    	   <int *> _ilo_grad_phi_gb.data, <int *> _ihi_grad_phi_gb.data,
         <int *> _jlo_grad_phi_gb.data, <int *> _jhi_grad_phi_gb.data,
         <int *> _klo_grad_phi_gb.data, <int *> _khi_grad_phi_gb.data,
	       <double *> phi.data,
	       <int *> _ilo_phi_gb.data, <int *> _ihi_phi_gb.data,
         <int *> _jlo_phi_gb.data, <int *> _jhi_phi_gb.data,
         <int *> _klo_phi_gb.data, <int *> _khi_phi_gb.data,
    	   <int *> _ilo_fb.data, <int *> _ihi_fb.data,
         <int *> _jlo_fb.data, <int *> _jhi_fb.data,
         <int *> _klo_fb.data, <int *> _khi_fb.data,
	       <double *> _dx.data, <double *> _dy.data, <double *> _dz.data)

    return normal_x, normal_y, normal_z

def lsm3dsurfaceareazerolevelset_(np.ndarray[double, ndim=1] phi,
                                  np.ndarray[double, ndim=1] phi_x,
                                  np.ndarray[double, ndim=1] phi_y,
                                  np.ndarray[double, ndim=1] phi_z,
                                  ilo_grad_phi_gb, ihi_grad_phi_gb, jlo_grad_phi_gb,
                                  jhi_grad_phi_gb, klo_grad_phi_gb, khi_grad_phi_gb,
                                  ilo_phi_gb, ihi_phi_gb, jlo_phi_gb,
                                  jhi_phi_gb, klo_phi_gb, khi_phi_gb,
                                  ilo_ib, ihi_ib, jlo_ib, jhi_ib, klo_ib, khi_ib,
                                  nx=1, ny=1, nz=1, dx=1., dy=1., dz=1.,
                                  epsilon=1.e-10):
     """
     * LSM3D_SURFACE_AREA_ZERO_LEVEL_SET() computes the surface area of the
     * surface defined by the zero level set.
     *
     * Arguments:
     *  - area (out):            area of the surface defined by the zero level
     *                           set
     *  - phi (in):              level set function
     *  - phi_* (in):            components of \f$ \nabla \phi \f$
     *  - dx, dy, dz (in):       grid spacing
     *  - epsilon (in):          width of numerical smoothing to use for
     *                           Heaviside function
     *  - *_gb (in):             index range for ghostbox
     *  - *_ib (in):             index range for interior box
     *
     * Return value:         none
     *
     """

     cdef np.ndarray[double, ndim=1] surface_area = np.zeros((nx * ny * nz,))
     cdef int* _ilo_grad_phi_gb = &ilo_grad_phi_gb
     cdef int* _ihi_grad_phi_gb = &ihi_grad_phi_gb
     cdef int* _jlo_grad_phi_gb = &jlo_grad_phi_gb
     cdef int* _jhi_grad_phi_gb = &jhi_grad_phi_gb
     cdef int* _klo_grad_phi_gb = &klo_grad_phi_gb
     cdef int* _khi_grad_phi_gb = &khi_grad_phi_gb
     cdef int* _ilo_phi_gb = &ilo_phi_gb
     cdef int* _ihi_phi_gb = &ihi_phi_gb
     cdef int* _jlo_phi_gb = &jlo_phi_gb
     cdef int* _jhi_phi_gb = &jhi_phi_gb
     cdef int* _klo_phi_gb = &klo_phi_gb
     cdef int* _khi_phi_gb = &khi_phi_gb
     cdef int* _ilo_ib = &ilo_ib
     cdef int* _ihi_ib = &ihi_ib
     cdef int* _jlo_ib = &jlo_ib
     cdef int* _jhi_ib = &jhi_ib
     cdef int* _klo_ib = &klo_ib
     cdef int* _khi_ib = &khi_ib
     cdef double* _dx = &dx
     cdef double* _dy = &dy
     cdef double* _dz = &dz
     cdef double* _epsilon = &epsilon

     lsm3dsurfaceareazerolevelset(
        <double *> surface_area.data,
        <double *> phi.data,
        <int *> _ilo_phi_gb.data, <int *> _ihi_phi_gb.data, <int *> _jlo_phi_gb.data,
        <int *> _jhi_phi_gb.data, <int *> _klo_phi_gb.data, <int *> _khi_phi_gb.data,
        <double *> phi_x.data, <double *> phi_y.data, <double *> phi_z.data,
        <int *> _ilo_grad_phi_gb.data, <int *> _ihi_grad_phi_gb.data,
        <int *> _jlo_grad_phi_gb.data,
        <int *> _jhi_grad_phi_gb.data, <int *> _klo_grad_phi_gb.data,
        <int *> _khi_grad_phi_gb.data,
        <int *> _ilo_ib.data, <int *> _ihi_ib.data,
        <int *> _jlo_ib.data, <int *> _jhi_ib.data,
        <int *> _klo_ib.data, <int *> _khi_ib.data,
        <double *> _dx.data, <double *> _dy.data, <double *> _dz.data,
        <double *> _epsilon.data);

      return


def lsm3dcomputemeancurvatureorder2local_():
  """
  *
  *  LSM3D_COMPUTE_MEAN_CURVATURE_ORDER2_LOCAL() computes mean curvature
  *  kappa = div ( grad_phi / |grad_phi|)
  *  kappa = ( phi_xx*phi_y^2 + phi_yy*phi_x^2 - 2*phi_xy*phi_x*phi_y +
  *            phi_xx*phi_z^2 + phi_zz*phi_x^2 - 2*phi_xz*phi_x*phi_z +
  *            phi_yy*phi_z^2 + phi_zz*phi_y^2 - 2*phi_yz*phi_y*phi_z )/
  *          ( | grad phi | ^ 3 )
  *  Note that this value is technically twice the mean curvature.
  *  Standard centered 27 point stencil, second order differencing used.
  *  First order derivatives assumed precomputed.
  c
  *  Arguments:
  *    kappa     (in/out): curvature data array
  *    phi          (in):  level set function
  *    phi_*        (in):  first order derivatives of phi
  *    grad_phi_mag (in):  gradient magnitude of phi
  *    *_gb        (in):   index range for ghostbox
  *    dx, dy      (in):   grid spacing
  *    index_[xyz]  (in):  [xyz] coordinates of local (narrow band) points
  *    n*_index    (in):  index range of points to loop over in index_*
  *    narrow_band(in):   array that marks voxels outside desired fillbox
  *    mark_fb(in):      upper limit narrow band value for voxels in
  *                      fillbox
  *
  """

  cdef np.ndarray[double, ndim=1] surface_area = np.zeros((nx * ny * nz,))
  cdef int* _ilo_grad_phi_gb = &ilo_grad_phi_gb
  cdef int* _ihi_grad_phi_gb = &ihi_grad_phi_gb
  cdef int* _jlo_grad_phi_gb = &jlo_grad_phi_gb
  cdef int* _jhi_grad_phi_gb = &jhi_grad_phi_gb
  cdef int* _klo_grad_phi_gb = &klo_grad_phi_gb
  cdef int* _khi_grad_phi_gb = &khi_grad_phi_gb
  cdef int* _ilo_phi_gb = &ilo_phi_gb
  cdef int* _ihi_phi_gb = &ihi_phi_gb
  cdef int* _jlo_phi_gb = &jlo_phi_gb
  cdef int* _jhi_phi_gb = &jhi_phi_gb
  cdef int* _klo_phi_gb = &klo_phi_gb
  cdef int* _khi_phi_gb = &khi_phi_gb
  cdef int* _ilo_nb_gb = &ilo_nb_gb
  cdef int* _ihi_nb_gb = &ihi_nb_gb
  cdef int* _jlo_nb_gb = &jlo_nb_gb
  cdef int* _jhi_nb_gb = &jhi_nb_gb
  cdef int* _klo_nb_gb = &klo_nb_gb
  cdef int* _khi_nb_gb = &khi_nb_gb
  cdef int* _ilo_kappa_gb = &ilo_kappa_gb
  cdef int* _ihi_kappa_gb = &ihi_kappa_gb
  cdef int* _jlo_kappa_gb = &jlo_kappa_gb
  cdef int* _jhi_kappa_gb = &jhi_kappa_gb
  cdef int* _klo_kappa_gb = &klo_kappa_gb
  cdef int* _khi_kappa_gb = &khi_kappa_gb
  cdef double* _dx = &dx
  cdef double* _dy = &dy
  cdef double* _dz = &dz
  cdef double* _epsilon = &epsilon
  cdef int* _index_x = &index_x
  cdef int* _index_y = &index_y
  cdef int* _index_z = &index_z


    lsm3dcomputemeancurvatureorder2local(<double *> kappa.data,
        <int *> _ilo_kappa_gb.data, <int *> _ihi_kappa_gb.data,
        <int *> _jlo_kappa_gb.data, <int *> _jhi_kappa_gb.data,
        <int *> _klo_kappa_gb.data, <int *> _khi_kappa_gb.data,
        <double *> phi.data,
        <int *> _ilo_phi_gb.data, <int *> _ihi_phi_gb.data,
        <int *> _jlo_phi_gb.data, <int *> _jhi_phi_gb.data,
        <int *> _klo_phi_gb.data, <int *> _khi_phi_gb.data,
        <double *> phi_x.data, <double *> phi_y.data, <double *> phi_z.data,
        <double *> grad_phi_mag.data,
        <int *> _ilo_grad_phi_gb.data, <int *> _ihi_grad_phi_gb.data,
        <int *> _jlo_grad_phi_gb.data, <int *> _jhi_grad_phi_gb.data,
        <int *> _klo_grad_phi_gb.data, <int *> _khi_grad_phi_gb.data,
        <double *> _dx.data, <double *> _dy.data, <double *> _dz.data,
        <int *> _index_x.data, <int *> _index_y.data, <int *> _index_z.data,
        <int *> _nlo_index.data, <int *> _nhi_index.data,
        <char *> narrow_band.data,
        <int *> _ilo_nb_gb.data, <int *> _ihi_nb_gb.data,
        <int *> _jlo_nb_gb.data, <int *> _jhi_nb_gb.data,
        <int *> _klo_nb_gb.data, <int *> _khi_nb_gb.data,
        <char *> _mark_fb.data)

    return


def computeDistanceFunction3d_(np.ndarray[double, ndim=1] phi, nx=1, ny=1, nz=1, dx=1., dy=1., dz=1., order=2):
    cdef np.ndarray[double, ndim=1] distance_function = np.zeros((nx * ny * nz,))
    cdef int spatial_derivative_order = order
    cdef np.ndarray[int, ndim=1] grid_dims = np.array((nx, ny, nz), dtype=int32)
    cdef np.ndarray[double, ndim=1] _dx = np.array((dx, dy, dz))

    cdef double *maskdata=NULL

    error = computeDistanceFunction3d(
		  <double *> distance_function.data,
    	<double *> phi.data,
    	<double *> maskdata,
    	spatial_derivative_order,
    	<int *> grid_dims.data,
    	<double *> _dx.data)

    return distance_function

def computeExtensionFields3d_(np.ndarray[double, ndim=1] phi, np.ndarray[double, ndim=2] extensionFields,
                              np.ndarray[double, ndim=1] mask=np.empty((0,), dtype=double),
                              np.ndarray[double, ndim=1] extension_mask=np.empty((0,), dtype=double),
                              nx=1,  ny=1, nz=1, dx=1., dy=1., dz=1., order=2):

    cdef int num_ext_fields = extensionFields.shape[0]

    cdef np.ndarray[double, ndim=1] distance_function = np.zeros((nx * ny * nz,))
    cdef int spatial_derivative_order = order

    cdef np.ndarray[int, ndim=1] grid_dims = np.array((nx, ny, nz), dtype=int32)
    cdef np.ndarray[double, ndim=1] _dx = np.array((dx, dy, dz))

    cdef double **ext_fields = <double **> malloc(num_ext_fields * sizeof(double*))
    cdef double **source_fields = <double **> malloc(num_ext_fields * sizeof(double*))
    cdef np.ndarray[double, ndim=2] extReturnFields = np.zeros((num_ext_fields, nx * ny * nz))

    cdef double *maskdata=NULL
    cdef double *extension_maskdata=NULL

    if len(mask) > 0:
        maskdata = <double *> mask.data

    if len(extension_mask) > 0:
        extension_maskdata = <double *> extension_mask.data


    for i in range(num_ext_fields):
        ext_fields[i] = &extReturnFields[i,0]
        source_fields[i] = &extensionFields[i,0]

    error = computeExtensionFields3d(
		      <double *> distance_function.data,
		      <double **> ext_fields,
    	    <double *> phi.data,
    	    <double *> maskdata,
		      <double **> source_fields,
		      <double *> extension_maskdata,
		      num_ext_fields,
    	    spatial_derivative_order,
    	    <int *> grid_dims.data,
    	    <double *> _dx.data)

    free(ext_fields)
    free(source_fields)

    return distance_function, extReturnFields

def solveEikonalEquation3d_(np.ndarray[double, ndim=1] phi,
                            np.ndarray[double, ndim=1] speed,
                            nx=1, ny=1, nz=1, dx=1., dy=1., dz=1., order=2):

    cdef np.ndarray[double, ndim=1] mask = np.zeros((nx * ny * nz,))
    cdef int spatial_derivative_order = order
    cdef np.ndarray[int, ndim=1] grid_dims = np.array((nx, ny, nz), dtype=int32)
    cdef np.ndarray[double, ndim=1] _dx = np.array((dx, dy, dz))

    cdef double *maskdata=NULL

    minphi = min(phi)
    phi[phi > 0] = minphi - 1.
    phi[:] = phi - minphi

    error = solveEikonalEquation3d(
		<double *> phi.data,
    	<double *> speed.data,
    	<double *> maskdata,
    	spatial_derivative_order,
    	<int *> grid_dims.data,
    	<double *> _dx.data)

    phi[:] = phi + minphi

    return phi
