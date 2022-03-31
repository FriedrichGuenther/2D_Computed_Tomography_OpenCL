__kernel void fast_ram_lak_filter(__global const float* sinogram, __global float* filtered, int scans, int rots, float pi, float h)
{
	int j,shift;
	float nom,tmp;
	
	float scale = (-1)*((float) 1/(2*h*pi*pi)); // Only used for uneven indeces
	int i = get_global_id(0); // In {0,...,rots}
	int k = get_global_id(1); // In {0,...,scans}
	
	tmp=0.0;
	shift = k&1; // If k is (un)even, go through the un(even) indices in {0,...,res-1}
	tmp += ( (float) 1/(8*h))*sinogram[i*scans+k]; // Summand for k=j
	#pragma unroll
	for(j=1-shift; j<scans; j+=2) // Indix shift not necessary because we look at differences; because we only end up with uneven differences, we never divide by zero
	{
		nom = (float) 1/((k-j)*(k-j));
 		tmp += nom*scale*sinogram[i*scans+j];
	}
	filtered[i*scans+k] = tmp;
}