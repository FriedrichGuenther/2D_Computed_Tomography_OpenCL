__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;

__kernel void shepp_logan_filter(read_only image2d_t sinogram, write_only image2d_t filtered, int scans, int rots, float pi, float h, float epsilon, float b)
{
	int j;
	float tmp,s,pixel,cond;
	
	float scale = (b*b*h)/(2*pi*pi*pi);
	int i = get_global_id(0); // In {0,...,rots}
	int k = get_global_id(1); // In {0,...,scans}
	
	tmp = 0.0;
	#pragma unroll
	for(j=0; j<scans; j++)
	{
		s = (k-j)*h;
		pixel = read_imagef(sinogram, sampler, (int2) (j,i)).x;
		cond = (float) b*s - 0.5*pi;
		if (fabs(cond) > epsilon)
		{
			tmp += ( ((0.5*pi) - (b*s*sin((float) b*s))) )/( (0.25*pi*pi) - (b*b*s*s) )*pixel;
		}
		else 
		{
			tmp += ((float) 1/pi)*pixel;
		}
	}
	write_imagef(filtered, (int2) (k,i), (float4) (scale*tmp,0,0,0));
}