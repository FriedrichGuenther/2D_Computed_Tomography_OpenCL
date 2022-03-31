__kernel void compute_sinogram(__global const float* image, __global float* sinogram, int res, int scans, int underscans, int rots, float pi, float h)
{
	int x,y; // Cartesian coordinates of image
	float x_rot, y_rot; // Cartesian coordinates of source image
	
	int i_rotf, j_rotf;
	
	float rot, sinw, cosw, tmp;
	int i,j,k;
	
	j = get_global_id(0); // Row of output image; only go through every underscan'th row {0, underscan, 2underscan, ...}
	k = get_global_id(1);
	y = res/2-(j*underscans); 
	
	
	tmp = 0.0;
	rot = (float) k/rots;
	#pragma unroll
	for(i=0; i<res; i++) // Calculated the rotation of the j-th row of image by rot*pi
	{
		x = i - res/2; 
		sinw = sincos(-rot*pi, &cosw);
	
		x_rot = cosw*x - sinw*y + (float) res/2;
		y_rot = sinw*x + cosw*y + (float) res/2;
		
		i_rotf = round(x_rot);
		j_rotf = round(y_rot);
		
		if(i_rotf < 0 || j_rotf < 0	|| i_rotf > res-1 || j_rotf > res-1) // Out-of-bounds check
		{
			continue;
		}
		
		else 
		{
			tmp += image[i_rotf*res+j_rotf];
		}
	}

	// Because first and last pixel of each row are black anyways, we don't need to substract anything at start and end

	sinogram[k*scans+j] = h*tmp;	
}