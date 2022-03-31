__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

__kernel void compute_sinogram(read_only image2d_t phantom, write_only image2d_t sinogram, int res, int scans, int underscans, int rots, float pi, float h)
{
	int x,y; // Cartesian coordinates of image
	float x_rot, y_rot; // Cartesian coordinates of source image
	float pixel;
	
	float rot, sinw, cosw, tmp;
	int i,j,k,scale;
	
	j = get_global_id(0); // In {0,...,scans}
	k = get_global_id(1); // In {0,...,rots}
	y = res/2-(j*underscans); // Fixed row index
	
	tmp = 0.0;
	scale = 2;
	rot = pi/2 - (pi*k)/rots;
	#pragma unroll
	for(i=0; i<res; i++) // Calculate the rotation of the j-th row of image by rot*pi
	{
		x = i - res/2; 
		
		sinw = sincos(rot, &cosw); // Rotation back to source, hence "-"
			
		x_rot = (cosw*x - sinw*y + (float) res/2)/res;
		y_rot = (sinw*x + cosw*y + (float) res/2)/res;
		
		pixel = read_imagef(phantom, sampler, (float2) (x_rot, y_rot)).x;
		tmp += pixel; // Due to the sampler settings, the TMUs will bilinearly interpolate over the square in which (x_rot, y_rot) lies
	}
	// Because first and last pixel of each row are black anyways, we don't need to substract anything at start and end
	write_imagef(sinogram, (int2) (j,k), (float4) ((h)*tmp, 0,0,0));
}