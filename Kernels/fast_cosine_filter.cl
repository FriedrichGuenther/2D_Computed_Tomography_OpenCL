__kernel void fast_cosine_filter(__global const float* sinogram, __global float* filtered, int scans, int rots, float pi, float h)
{
	int j, sign, par;
	float nom,tmp,x;
	
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
		tmp += ( sign*pi*( 1-x ) - 2*( 1+x ) )*( (float) 1/( (x-1)*(x-1) ) )*sinogram[i*scans+j];
		sign *= (-1);
	}
	filtered[i*scans+k] = scale*tmp;
}
