__kernel void fast_shepp_logan_filter(__global const float* sinogram, __global float* filtered, int scans, int rots, float pi, float h)
{
	int k,j;
	float nom,tmp;
	
	float scale = (float) 1/(pi*pi*h);
	int i = get_global_id(0);
	
	for(k=0; k<scans; k++)
	{
		tmp = 0.0;
		for(j=0; j<scans; j++)
		{
			nom = (float) 1/(1-(4*(k-j)*(k-j)));
			tmp += nom*sinogram[i*rots+j];
		}
		filtered[i*rots+k] = scale*tmp;
	}
}