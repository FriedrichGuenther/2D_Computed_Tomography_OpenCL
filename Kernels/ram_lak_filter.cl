__kernel void ram_lak_filter(__global float* sinogram, __global float* filtered, int scans, int rots, float pi, float h, float epsilon, float b)
{
	int j;
	float tmp,s,pixel,sins,sins2;
	
	float scale = (float) (b*b*h)/(4*pi*pi);
	int i = get_global_id(0); // In {0,...,rots}
	int k = get_global_id(1); // In {0,...,scans}
	
	tmp = 0.0;
	#pragma unroll
	for(j=0; j<scans; j++)
	{
		s = b*(k-j)*h;
		sins = sin((float) s);
		sins2 = sin((float) 0.5*s);
		pixel = sinogram[i*scans+j];
		if (fabs(s) > epsilon)
		{
			tmp += ((sins/s) - 0.5*(sins2*sins2)/(0.25*s*s))*pixel;
		}
		else 
		{
			tmp += 0.5*pixel;
		}
	}
	filtered[i*scans+k] = scale*tmp;
}
