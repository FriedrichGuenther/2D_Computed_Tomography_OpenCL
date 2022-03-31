__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;

__kernel void fast_shepp_logan_filter(read_only image2d_t sinogram, write_only image2d_t filtered, int scans, int rots, float pi, float h)
{
	int j;
	float nom,tmp,pixel;
	
	float scale = (float) 1/(pi*pi*h);
	int i = get_global_id(0); // In {0,...,rots}
	int k = get_global_id(1); // In {0,...,scans}
	
	tmp = 0.0;
	#pragma unroll
	for(j=0; j<scans; j++)
	{
		nom = (float) 1/(1-(4*(k-j)*(k-j)));
		pixel = read_imagef(sinogram, sampler, (int2) (j,i)).x;
		tmp += nom*pixel;
	}
	write_imagef(filtered, (int2) (k,i), (float4) (scale*tmp,0,0,0));
}