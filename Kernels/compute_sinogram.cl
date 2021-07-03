__kernel void compute_sinogram(__global const float* image, __global float* sinogram, int res, int scans, int underscans, int rots, float pi, float h)
{
	int x,y; // Cartesian coordinates of image
	float x_rot, y_rot; // Cartesian coordinates of source image
	
	int i_rotf, i_rotc, j_rotf, j_rotc; // indices of rotated point after floor and ceiling
	float dx, dy; // Difference to neighbouring pixels
	float tln, trn, bln, brn; // 4 neighbouring pixels
	float ti, bi, bil; // Variables for interpolation values
	
	float rot, sinw, cosw, tmp;
	int step[4] = {1,0,-1,0};
	int i,j,k,scale;
	
	j = get_global_id(0); // Row of output image; only go through every underscan'th row {0, underscan, 2underscan, ...}
	y = res/2-(j*underscans); 
	
	for(k=0; k<rots; k++) // Goes through the rotations
	{
		tmp = 0.0;
		scale = 2;
		rot = (float) k/rots;
		for(i=0; i<res; i++) // Calculated the rotation of the j-th row of image by rot*pi
		{
			scale += step[i%4];
			
			x = i - res/2; 
		
			if (x==0 && y==0)
			{
				tmp += scale*image[(j*underscans)*res+i]; 
				continue;
			}
			
			sinw = sin(-rot*pi); // Rotation back to source, hence "-"
			cosw = cos(-rot*pi);
			
			x_rot = cosw*x - sinw*y + (float) res/2;
			y_rot = sinw*x + cosw*y + (float) res/2;
		
			i_rotf = floor(x_rot);
			i_rotc = ceil(x_rot);
			j_rotf = floor(y_rot);
			j_rotc = ceil(y_rot);
		
			if(i_rotf < 0 || i_rotc < 0 || j_rotf < 0 || j_rotc < 0 || i_rotf > res || i_rotc > res || j_rotf > res || j_rotf > res) // Out-of-bounds check
			{
				continue;
			}
		
			else 
			{
				dx = x_rot - (float) i_rotf;
				dy = y_rot - (float) j_rotf;	
					
				tln = image[i_rotf*res+j_rotf];
				trn = image[i_rotc*res+j_rotf];
				bln = image[i_rotf*res+j_rotc];
				brn = image[i_rotc*res+j_rotc];
					
				ti = (1-dx)*tln + dx*trn;
				bi = (1-dx)*bln + dx*brn;
				bil = (1-dy)*ti + dy*bi;
					
				clamp(bil, 0.0f, 255.0f);
				tmp += scale*bil;
			}
		}
		
		// Because first and last pixel of each row are black anyways, we don't need to substract anything at start and end
		
		tmp *= (3*h/8);
		sinogram[k*scans+j] = tmp;
	}	
}