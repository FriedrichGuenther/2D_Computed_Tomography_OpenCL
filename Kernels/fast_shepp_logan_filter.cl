__kernel void fast_shepp_logan_filter(__global const float* sinogram, __global float* filtered, int scans, int rots, float pi, float h)
{
	int j;
	float nom,tmp;
	
	float scale = (float) 1/(pi*pi*h);
	int i = get_global_id(0); // In {0,...,rots}
	int k = get_global_id(1); // In {0,...,scans}
	
	tmp = 0.0;
	#pragma unroll
	for(j=0; j<scans; j++)
	{
		nom = (float) 1/(1-(4*(k-j)*(k-j)));
		tmp += nom*sinogram[i*scans+j];
	}
	filtered[i*scans+k] = scale*tmp;
}