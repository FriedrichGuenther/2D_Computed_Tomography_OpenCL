__kernel void cosine_filter(__global float* sinogram, __global float* filtered, int scans, int rots, float pi, float h, float epsilon, float b)
{
	int j,shift;
	float wb,s1,s2,tmp,pixel;
	
	float scale = (b*b*h)/(8*pi*pi); 
	int i = get_global_id(0); // In {0,...,rots}
	int k = get_global_id(1); // In {0,...,scans}
	
	tmp=0.0;
	#pragma unroll
	for(j=0; j<scans; j++)
	{
		wb = 0.0;
		s1 = pi/2 - b*h*(k-j);
		s2 = pi/2 + b*h*(k-j);
		pixel = sinogram[i*scans+j];
		if (fabs(s1) > epsilon && fabs(s2) > epsilon)
		{
			wb += (sin((float)s1)/s1 - 0.5*( ( sin((float)s1/2)*sin((float)s1/2) )/( (s1/2)*(s1/2) ) ) );
			wb += (sin(s2)/s2 - 0.5*( ( sin((float)s2/2)*sin((float)s2/2) )/( (s2/2)*(s2/2) ) ) );
			wb *= scale;
		}
		else if ( fabs(s1) < epsilon && fabs(s2) > epsilon) 
		{
			wb += 1/2;
			wb += (sin((float)s2)/s2 - 0.5*( ( sin((float)s2/2)*sin((float)s2/2) )/( (s2/2)*(s2/2) ) ) );
			wb *= scale;
		}
		else if ( fabs(s1) > epsilon && fabs(s2) < epsilon )
		{
			wb += (sin((float)s1)/s1 - 0.5*( ( sin((float)s1/2)*sin((float)s1/2) )/( (s1/2)*(s1/2) ) ) );
			wb += 1/2;
			wb *= scale;
		}
 		tmp += wb*pixel;
	}
	filtered[i*scans+k] = tmp;
}