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

cdef extern void lsm3dcomputesignedunitnormal(
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
