__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;

__kernel void fast_cosine_filter(read_only image2d_t sinogram, write_only image2d_t filtered, int scans, int rots, float pi, float h)
{
	int j, sign, par;
	float nom,tmp,x,pixel;
	
	float scale = (float) 1/(2*h*pi*pi);
	int i = get_global_id(0); // In {0,...,rots}
	int k = get_global_id(1); // In {0,...,scans}
	
	par = k&1;
	sign = 1 - (par)*2; // k even: sign = 1; k uneven: sign = -1
	tmp = 0.0;
	#pragma unroll
	for(j=0; j<scans; j++)
	{
		x = 4*((k-j)*(k-j));
		pixel = read_imagef(sinogram, sampler, (int2) (j,i)).x;
		tmp += ( sign*pi*( 1-x ) - 2*( 1+x ) )*( (float) 1/( (x-1)*(x-1) ) )*pixel;
		sign *= (-1);
	}
	write_imagef(filtered, (int2) (k,i), (float4) (scale*tmp,0,0,0));
}
