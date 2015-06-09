import numpy as np
cimport numpy as np

cdef extern int computeDistanceFunction3d(
	double *distance_function,
	double *phi,
	double *mask,
	int spatial_derivative_order,
	int *grid_dims,
	double *dx)

cdef extern int computeExtensionFields3d(
 	double *distance_function,
    double **extension_fields,
	double *phi,
	double *mask,
	double **source_fields,
	double *extension_mask,
	int num_ext_fields,
	int spatial_derivative_order,
	int *grid_dims,
	double *dx)

cdef extern int solveEikonalEquation3d(
	double *phi,
	double *speed,
	double *mask,
	int spatial_derivative_order,
	int *grid_dims,
	double *dx)

cdef extern	void lsm3dsurfaceareazerolevelset_(
	double *surface_area,
	double *phi,
	const int *ilo_phi_gb,
	const int *ihi_phi_gb,
	const int *jlo_phi_gb,
	const int *jhi_phi_gb,
	const int *klo_phi_gb,
	const int *khi_phi_gb,
	double *phi_x,
	double *phi_y,
	double *phi_z,
	const int *ilo_grad_phi_gb,
	const int *ihi_grad_phi_gb,
	const int *jlo_grad_phi_gb,
	const int *jhi_grad_phi_gb,
	const int *klo_grad_phi_gb,
	const int *khi_grad_phi_gb,
	const int *ilo_ib,
	const int *ihi_ib,
	const int *jlo_ib,
	const int *jhi_ib,
	const int *klo_ib,
	const int *khi_ib,
	double *dx,
	double *dy,
	double *dz,
	double *epsilon)

cdef extern void lsm3dcomputesignedunitnormal_(
	double *normal_x,
	double *normal_y,
	double *normal_z,
	const int *ilo_normal_gb,
	const int *ihi_normal_gb,
	const int *jlo_normal_gb,
	const int *jhi_normal_gb,
	const int *klo_normal_gb,
	const int *khi_normal_gb,
	const double *phi_x,
	const double *phi_y,
	const double *phi_z,
	const int *ilo_grad_phi_gb,
	const int *ihi_grad_phi_gb,
	const int *jlo_grad_phi_gb,
	const int *jhi_grad_phi_gb,
	const int *klo_grad_phi_gb,
	const int *khi_grad_phi_gb,
	const double *phi,
	const int *ilo_phi_gb,
	const int *ihi_phi_gb,
	const int *jlo_phi_gb,
	const int *jhi_phi_gb,
	const int *klo_phi_gb,
	const int *khi_phi_gb,
	const int *ilo_fb,
	const int *ihi_fb,
	const int *jlo_fb,
	const int *jhi_fb,
	const int *klo_fb,
	const int *khi_fb,
	const double *dx,
	const double *dy,
	const double *dz)

cdef extern void lsm3dcomputemeancurvatureorder2local_(
	double *kappa,
	const int *ilo_kappa_gb,
	const int *ihi_kappa_gb,
	const int *jlo_kappa_gb,
	const int *jhi_kappa_gb,
	const int *klo_kappa_gb,
	const int *khi_kappa_gb,
	double *phi,
	const int *ilo_phi_gb,
	const int *ihi_phi_gb,
	const int *jlo_phi_gb,
	const int *jhi_phi_gb,
	const int *klo_phi_gb,
	const int *khi_phi_gb,
	double *phi_x,
	double *phi_y,
	double *phi_z,
	double *grad_phi_mag,
	const int *ilo_grad_phi_gb,
	const int *ihi_grad_phi_gb,
	const int *jlo_grad_phi_gb,
	const int *jhi_grad_phi_gb,
	const int *klo_grad_phi_gb,
	const int *khi_grad_phi_gb,
	double *dx,
	double *dy,
	double *dz,
	const int *index_x,
	const int *index_y,
	const int *index_z,
	const int *nlo_index,
	const int *nhi_index,
	const unsigned char *narrow_band,
	const int *ilo_nb_gb,
	const int *ihi_nb_gb,
	const int *jlo_nb_gb,
	const int *jhi_nb_gb,
	const int *klo_nb_gb,
	const int *khi_nb_gb,
	const unsigned char *mark_fb)


cdef extern void lsm3dcomputemeancurvatureorder2_(
	double *kappa,
	const int *ilo_kappa_gb,
	const int *ihi_kappa_gb,
	const int *jlo_kappa_gb,
	const int *jhi_kappa_gb,
	const int *klo_kappa_gb,
	const int *khi_kappa_gb,
	double *phi,
	const int *ilo_phi_gb,
	const int *ihi_phi_gb,
	const int *jlo_phi_gb,
	const int *jhi_phi_gb,
	const int *klo_phi_gb,
	const int *khi_phi_gb,
	double *phi_x,
	double *phi_y,
	double *phi_z,
	double *grad_phi_mag,
	const int *ilo_grad_phi_gb,
	const int *ihi_grad_phi_gb,
	const int *jlo_grad_phi_gb,
	const int *jhi_grad_phi_gb,
	const int *klo_grad_phi_gb,
	const int *khi_grad_phi_gb,
	const int *ilo_kappa_fb,
	const int *ihi_kappa_fb,
	const int *jlo_kappa_fb,
	const int *jhi_kappa_fb,
	const int *klo_kappa_fb,
	const int *khi_kappa_fb,
	double *dx,
	double *dy,
	double *dz
)

cdef extern void lsm3dcomputegaussiancurvatureorder2_(
	double *kappa,
	const int *ilo_kappa_gb,
	const int *ihi_kappa_gb,
	const int *jlo_kappa_gb,
	const int *jhi_kappa_gb,
	const int *klo_kappa_gb,
	const int *khi_kappa_gb,
	double *phi,
	const int *ilo_phi_gb,
	const int *ihi_phi_gb,
	const int *jlo_phi_gb,
	const int *jhi_phi_gb,
	const int *klo_phi_gb,
	const int *khi_phi_gb,
	double *phi_x,
	double *phi_y,
	double *phi_z,
	double *grad_phi_mag,
	const int *ilo_grad_phi_gb,
	const int *ihi_grad_phi_gb,
	const int *jlo_grad_phi_gb,
	const int *jhi_grad_phi_gb,
	const int *klo_grad_phi_gb,
	const int *khi_grad_phi_gb,
	const int *ilo_kappa_fb,
	const int *ihi_kappa_fb,
	const int *jlo_kappa_fb,
	const int *jhi_kappa_fb,
	const int *klo_kappa_fb,
	const int *khi_kappa_fb,
	double *dx,
	double *dy,
	double *dz
)

cdef extern void lsm3dcentralgradorder4_(
	double *phi_x,
	double *phi_y,
	double *phi_z,
	const int *ilo_grad_phi_gb,
	const int *ihi_grad_phi_gb,
	const int *jlo_grad_phi_gb,
	const int *jhi_grad_phi_gb,
	const int *klo_grad_phi_gb,
	const int *khi_grad_phi_gb,
	const double *phi,
	const int *ilo_phi_gb,
	const int *ihi_phi_gb,
	const int *jlo_phi_gb,
	const int *jhi_phi_gb,
	const int *klo_phi_gb,
	const int *khi_phi_gb,
	const int *ilo_fb,
	const int *ihi_fb,
	const int *jlo_fb,
	const int *jhi_fb,
	const int *klo_fb,
	const int *khi_fb,
	const double *dx,
	const double *dy,
	const double *dz
)
