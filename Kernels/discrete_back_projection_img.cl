__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;

__kernel void discrete_back_projection(read_only image2d_t sinogram, __global float* reconstruction, int res, int half_scans, int scans, int rots, float scale, float pi, float h)
{
	int l = get_global_id(0); // In {0,...,res}
	int i = get_global_id(1); // In {0,...,res}
	float omegax, omegay, rot;
	
	int j,k;
	float x,y,s,u,tmp,pixel_1,pixel_2;
	
	x = -1+(h/2)+l*h; // Column
	y = 1-(h/2)-i*h; // Row
	
	#pragma unroll
	for(j=0; j<rots; j++)
	{
		rot = (float) j/rots;
		omegay = sincos(rot*pi, &omegax);
			
		tmp = 0.0;
		s = (float) (x*omegax + y*omegay)*half_scans + half_scans;
		k = floor(s);
		u = s-k;
		pixel_1 = read_imagef(sinogram, sampler, (int2) (k,j)).x;
		pixel_2 = read_imagef(sinogram, sampler, (int2) (k+1,j)).x;	
		reconstruction[i*res+l] += scale*((1-u)*pixel_1 + u*pixel_2);
	}
}