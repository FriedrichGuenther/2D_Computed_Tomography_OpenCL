__kernel void fast_cosine_filter(__global const float* sinogram, __global float* filtered, int scans, int rots, float pi, float h)
{
	int k,j;
	int sign, par;
	float nom,tmp,x;
	
	float scale = (float) 1/(2*h*pi*pi);
	int i = get_global_id(0);
	
	for(k=0; k<scans; k++)
	{
		par = k%2;
		sign = 1 - (par)*2; // k even: sign = 1; k uneven: sign = -1
		tmp = 0.0;
		for(j=0; j<scans; j++)
		{
			x = 4*((k-j)*(k-j));
			tmp += ( sign*pi*( 1-x ) - 2*( 1+x ) )*( (float) 1/( (x-1)*(x-1) ) )*sinogram[i*rots+j];
			sign *= (-1);
		}
		filtered[i*rots+k] = scale*tmp;
	}
}
