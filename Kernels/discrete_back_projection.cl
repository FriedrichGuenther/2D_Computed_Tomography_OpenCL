__kernel void discrete_back_projection(__global const float* sinogram, __global float* phantom, int res, int half_scans, int scans, int rots, float scale, float pi, float h)
{
	int l = get_global_id(0);
	float omegax, omegay, rot;
	
	int i,j,k;
	float x,y,u,s,tmp,layer;
	
	x = -1+(h/2)+l*h; // Column
	y = 1-(h/2); // Row
	for(i=0; i<res; i++)
	{
		y -= h;
		for(j=0; j<rots; j++)
		{
			rot = (float) j/rots;
			omegax = cos(pi-rot*pi);
			omegay = sin(pi-rot*pi);
			
			tmp = 0.0;
			s = (float) (x*omegax + y*omegay)*half_scans;
			k = floor(s);
			u = s-k;
			k+=half_scans;
			if(k > -1 && k < scans-2)
			{
				phantom[i*res+l] += scale*((1-u)*sinogram[j*scans+k] + u*sinogram[j*scans+(k+1)]);
			}
			else 
			{
				continue;
			}	
		}
	}
}