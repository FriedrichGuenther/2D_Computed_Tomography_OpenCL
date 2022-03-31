## 2D Computed Tomography in OpenCL ##

This repository contains functions for the inverse problem based approach to 2D computed tomography with parallel geometry, i.e. functions for the forward projection (using image rotation with bilinear interpolation rather than raytracing), RAM-LAK, Shepp-Logan, and Cosine filters for the sinogram and a function for the discrete back projection. 

The program is coded using the C++ Wrapper for OpenCL, using the old OpenCL 1.2 standard, and there are versions of the functions for both buffers and images2d_t objects.  
