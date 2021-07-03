#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS // Allows for better error catching/handling
#if defined(__APPLE__)  // C++ Wrapper for OpenCL
#include <OpenCL/cl2.hpp>
#else 
#include <CL/cl2.hpp>
#endif
#include <iostream> // IO Streams (cout, cin, ...)
#include <fstream> // File streams
#include <vector> // Vectors
#include <string> // String library
#include <sstream> // Streams
#include <iomanip> // Output formating
#include <cmath> // Math functions
#include <random> // Random number generator
#include <chrono>
#include "CL_err_helper.hpp"


using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// OpenCL host functions
void set_up_platform_and_devices(cl::Platform& cpu_platform, cl::Device& cpu_device);
cl::Program CL_create_program_source(cl::Context context, string filename);
vector<float> CL_compute_sinogram(cl::Device device, cl::Context context, vector<float>& image, unsigned int res, unsigned int rots, unsigned int underscans, unsigned int scans);
vector<float> CL_fast_ram_lak_filter(cl::Device device, cl::Context context, vector<float>& sinogram, unsigned int rots, unsigned int scans);
vector<float> CL_fast_shepp_logan_filter(cl::Device device, cl::Context context, vector<float>& sinogram, unsigned int rots, unsigned int scans);
vector<float> CL_fast_cosine_filter(cl::Device device, cl::Context context, vector<float>& sinogram, unsigned int rots, unsigned int scans);
vector<float> CL_discrete_back_projection(cl::Device device, cl::Context context, vector<float> &sinogram, unsigned int res, unsigned int rots, unsigned int scans);

// CT functions
unsigned int set_resolution(unsigned int m, unsigned int n);
vector<float> salt_and_pepper_noise(unsigned int res, float amp);

// IO functions and miscellaneous
vector<float> add_matrices(vector<float>& A, vector<float>& B);
vector<float> subtract_matrices(vector<float>& A, vector<float>& B);

bool try_reading_float(string& input, float& output);
bool try_reading_uint(string& input, unsigned int& output);
void split_string(string& input, vector<string>& output);
void print_matrix_float(vector<float> matrix, const unsigned int& m, const unsigned int& n);
void write_matrix_float_to_file(string filename, vector<float> matrix, const unsigned int& m, const unsigned int& n);
void write_matrix_float_to_pgm(string filename, vector<float> matrix, const unsigned int& m, const unsigned int& n);
void write_matrix_float_to_pgm_normalised(string filename, vector<float> matrix, const unsigned int& m, const unsigned int& n);
vector<float> read_matrix_float(string filename, unsigned int& m, unsigned int& n);
vector<float> read_pgm_to_matrix_float(string filename, unsigned int& m, unsigned int& n);

int main(void)
{
	// OpenCL host
	try 
	{
		cl::Platform platform;
		cl::Device device;
		set_up_platform_and_devices(platform, device);
		cl::Context context(device);
		
		// Constants 
		unsigned int res;
		unsigned int m,n,rots,scans,underscans;
		
		string infile;
		string outfile;
		
		infile = "./SLP/SLP_2048.pgm";
		vector<float> phantom = read_pgm_to_matrix_float(infile,m,n);
		cout << "Phantom read..." << endl;
		res = set_resolution(m,n);
		
		rots = res;
		underscans = 1;
		scans = res/underscans;
		duration<double, std::milli> t;
        
		// Main program

		auto t1 = high_resolution_clock::now();
		vector<float> sinogram = CL_compute_sinogram(device, context, phantom, res, rots, underscans, scans);
		outfile = "./Output/sinogram_res_"+to_string(res)+"_rots_"+to_string(rots)+"_underscans_"+to_string(underscans)+".pgm"; //"_normalised.pgm";
		write_matrix_float_to_pgm(outfile, sinogram, rots, scans);
		auto t2 = high_resolution_clock::now();
		duration<double, std::milli> compute = t2-t1;
		cout << "Sinogram computed in " << compute.count() << "ms..." << endl;
		t = t2-t1;
		
		/*t1 = high_resolution_clock::now();
		vector<float> rlf = CL_fast_ram_lak_filter(device, context, sinogram, rots, scans);
		outfile = "./Output/rlf_res_"+to_string(res)+"_rots_"+to_string(rots)+"_scans_"+to_string(scans)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, rlf, rots, scans);
		t2 = high_resolution_clock::now();
		compute = t2-t1;
		cout << "Fast Ram-Lak filter applied in " << compute.count() << "ms..." << endl;
		t += t2-t1;
		
		t1 = high_resolution_clock::now();
		vector<float> slf = CL_fast_shepp_logan_filter(device, context, sinogram, rots, scans);
		outfile = "./Output/slf_res_"+to_string(res)+"_rots_"+to_string(rots)+"_scans_"+to_string(scans)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, slf, rots, scans);
		t2 = high_resolution_clock::now();
		compute = t2-t1;
		cout << "Fast Shepp-Logan filter applied in " << compute.count() << "ms..." << endl;
		t += t2-t1;*/
		
		t1 = high_resolution_clock::now();
		vector<float> cf = CL_fast_cosine_filter(device, context, sinogram, rots, scans);
		outfile = "./Output/cf_res_"+to_string(res)+"_rots_"+to_string(rots)+"_scans_"+to_string(scans)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, cf, rots, scans);
		t2 = high_resolution_clock::now();
		compute = t2-t1;
		cout << "Fast Cosine filter applied in " << compute.count() << "ms..." << endl;
		t += t2-t1;
		
		/*t1 = high_resolution_clock::now();
		vector<float> dbp = CL_discrete_back_projection(device, context, sinogram, res, rots, scans);
		outfile = "./Output/dbp_res_"+to_string(res)+"_rots_"+to_string(rots)+"_scans_"+to_string(scans)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp, res, res);
		t2 = high_resolution_clock::now();
		compute = t2-t1;
		cout << "Discrete back projection of sinogram performed in " << compute.count() << "ms..." << endl;
		t += t2-t1;
		
		t1 = high_resolution_clock::now();
		vector<float> dbp_1 = CL_discrete_back_projection(device, context, rlf, res, rots, scans);
		outfile = "./Output/dbp_rlf_res_"+to_string(res)+"_rots_"+to_string(rots)+"_scans_"+to_string(scans)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp_1, res, res);
		t2 = high_resolution_clock::now();
		compute = t2-t1;
		cout << "Discrete back projection of Ram-Lak filtered sinogram performed in " << compute.count() << "ms..." << endl;
		t += t2-t1;
		
		t1 = high_resolution_clock::now();
		vector<float> dbp_2 = CL_discrete_back_projection(device, context, slf, res, rots, scans);
		outfile = "./Output/dbp_slf_res_"+to_string(res)+"_rots_"+to_string(rots)+"_scans_"+to_string(scans)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp_2, res, res);
		t2 = high_resolution_clock::now();
		compute = t2-t1;
		cout << "Discrete back projection of Shepp-Logan filtered sinogram performed in " << compute.count() << "ms..." << endl;
		t += t2-t1;*/
		
		t1 = high_resolution_clock::now();
		vector<float> dbp_3 = CL_discrete_back_projection(device, context, cf, res, rots, scans);
		outfile = "./Output/dbp_cf_res_"+to_string(res)+"_rots_"+to_string(rots)+"_scans_"+to_string(scans)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp_3, res, res);
		t2 = high_resolution_clock::now();
		compute = t2-t1;
		cout << "Discrete back projection of Cosine filtered sinogram performed in " << compute.count() << "ms..." << endl;
		t += t2-t1;
        
		cout << "Total compute time: " << t.count() << "ms" << endl;
		
		return EXIT_SUCCESS;
	} 
	catch(const cl::Error& clExp) 
	{
		cout << "cl Exception: " << clExp.what() << " with error code " << opencl_err_to_str(clExp.err()) << " (" << clExp.err() << ")" << endl;
	} 
	catch(const std::exception& e) 
	{
		cout << "other exception: " << e.what() << endl;
	}
}

// --------------------------------------------------------------------------------------------------------------------------------


void set_up_platform_and_devices(cl::Platform& cpu_platform, cl::Device& cpu_device)
{
	vector<cl::Platform> platforms;
	vector<cl::Device> devices_per_platform;
	vector<cl::Device> gpu_devices;
	
	unsigned int num_platforms;
	unsigned int num_target_device=0;
	unsigned int num_devices;
	unsigned int total_gpus;
	string input;
	
	cl::Platform::get(&platforms);
	num_platforms = platforms.size();
	
	for(int i=0; i<num_platforms; i++)
	{
		platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_per_platform);
		num_devices = devices_per_platform.size();
		for(int j=0; j<num_devices; j++)
		{
			if(devices_per_platform[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU)
			{
				gpu_devices.push_back(devices_per_platform[j]);
			}
			devices_per_platform.clear();
		}
	}
	total_gpus = gpu_devices.size();
	
	if(total_gpus == 0) 
	{
		throw std::runtime_error("No GPU devices available! Cannot carry on.");
	}

	if(total_gpus == 1)
	{
		cpu_device = gpu_devices[0];
		cpu_platform = gpu_devices[0].getInfo<CL_DEVICE_PLATFORM>();
		cout << "Selected " << cpu_device.getInfo<CL_DEVICE_NAME>() << " on platform " << cpu_platform.getInfo<CL_PLATFORM_NAME>() << " as GPU device." << endl;
	}
	
	if(total_gpus > 1)
	{
		cout << "Available GPU devices: " << endl;
		for(int i=0; i<total_gpus; i++)
		{
			cout << "["<<i+1<<"] " << gpu_devices[i].getInfo<CL_DEVICE_NAME>() << endl;
		}
		cout << "Please enter target GPU device: ";
		getline(cin, input); 
		while(!try_reading_uint(input, num_target_device) or num_target_device < 1 or num_target_device > total_gpus)
		{	
			cout << "Please enter target GPU device: ";
			getline(cin, input);
		}
		num_target_device--;
		cpu_device = gpu_devices[num_target_device];
		cpu_platform = cpu_device.getInfo<CL_DEVICE_PLATFORM>();
		cout << "Selected " << cpu_device.getInfo<CL_DEVICE_NAME>() << " on platform " << cpu_platform.getInfo<CL_PLATFORM_NAME>() << " as GPU device." << endl;
		num_target_device=0;
		cin.clear();
	}
	
}

cl::Program CL_create_program_from_source(cl::Context context, cl::Device device, string filename)
{
	ifstream program_file(filename);
	string program_string(istreambuf_iterator<char>(program_file), (istreambuf_iterator<char>()));
	cl::Program::Sources source { program_string };
	cl::Program program(context, source);
	try
	{
		program.build();
	}
	catch(cl::Error e)
	{
		cout << e.what() << " ; Error code " << e.err() << endl;
		string build_log;
		build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
		cout << build_log << endl;
	}
	
	return program;
}

vector<float> CL_compute_sinogram(cl::Device device, cl::Context context, vector<float>& image, unsigned int res, unsigned int rots, unsigned int underscans, unsigned int scans)
{
	vector<float> rotated(rots*res, 0.0);
	vector<float> sinogram(rots*scans, 0.0);
	vector<int> scales(res, 0.0);
	
	float h = (float) 2/res;
	double compute_time=0;	
	
	float pi = 4*atan(1);
	
	cl::Program program = CL_create_program_from_source(context, device, "./Kernels/compute_sinogram.cl");
	cl::Kernel kernel(program, "compute_sinogram");
	cl::CommandQueue queue(context, device); // , CL_QUEUE_PROFILING_ENABLE);
	
	cl::Buffer Image(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, res*res*sizeof(float));
	cl::Buffer Sinogram(context, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));

	queue.enqueueWriteBuffer(Image, CL_TRUE, 0, res*res*sizeof(float), &image[0]);
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	
	kernel.setArg(0, Image);
	kernel.setArg(1, Sinogram);
	kernel.setArg(2, res);
	kernel.setArg(3, scans);
	kernel.setArg(4, underscans);
	kernel.setArg(5, rots);
	kernel.setArg(6, pi);
	kernel.setArg(7, h);
	
	// cl::Event compute;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(scans), cl::NullRange, NULL); // &compute); 
	queue.enqueueReadBuffer(Sinogram, CL_TRUE, 0, scans*rots*sizeof(float), &sinogram[0]);
	queue.finish();
	
	/*cl_ulong start = compute.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong stop = compute.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	compute_time += (double) (stop - start)/1000000;*/	
	
	return sinogram;
}

vector<float> CL_fast_ram_lak_filter(cl::Device device, cl::Context context, vector<float>& sinogram, unsigned int rots, unsigned int scans)
{
	vector<float> filtered(rots*scans, 0.0);
	float h = (float) 2/scans;
	
	double compute_time=0;	
	
	float pi = 4*atan(1);
	
	cl::Program program = CL_create_program_from_source(context, device, "./Kernels/fast_ram_lak_filter.cl");
	cl::Kernel kernel(program, "fast_ram_lak_filter");
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));

	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	
	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);

	cl::Event compute;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots), cl::NullRange, NULL, &compute); 
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	queue.finish();

	cl_ulong start = compute.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong stop = compute.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	compute_time += (double) (stop - start)/1000000;	
	
	// cout << "Fast Ram-Lak filter applied to sinogram in " << compute_time << " ms..." << endl;
	
	return filtered;
}

vector<float> CL_fast_shepp_logan_filter(cl::Device device, cl::Context context, vector<float>& sinogram, unsigned int rots, unsigned int scans)
{
	vector<float> filtered(rots*scans, 0.0);
	float h = (float) 2/scans;
	
	double compute_time=0;	
	
	float pi = 4*atan(1);
	
	cl::Program program = CL_create_program_from_source(context, device, "./Kernels/fast_shepp_logan_filter.cl");
	cl::Kernel kernel(program, "fast_shepp_logan_filter");
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));

	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	
	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	
	cl::Event compute;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots), cl::NullRange, NULL, &compute); 
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	queue.finish();

	cl_ulong start = compute.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong stop = compute.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	compute_time += (double) (stop - start)/1000000;	
	
	// cout << "Fast Shepp-Logan filter applied to sinogram in " << compute_time << " ms..." << endl;
	
	return filtered;
}

vector<float> CL_fast_cosine_filter(cl::Device device, cl::Context context, vector<float>& sinogram, unsigned int rots, unsigned int scans)
{
	vector<float> filtered(rots*scans, 0.0);
	float h = (float) 2/scans;
	
	double compute_time=0;	
	
	float pi = 4*atan(1);
	
	cl::Program program = CL_create_program_from_source(context, device, "./Kernels/fast_cosine_filter.cl");
	cl::Kernel kernel(program, "fast_cosine_filter");
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));

	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	
	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	
	cl::Event compute;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots), cl::NullRange, NULL, &compute); 
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	queue.finish();

	cl_ulong start = compute.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong stop = compute.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	compute_time += (double) (stop - start)/1000000;	
	
	// cout << "Fast Cosine filter applied to sinogram in " << compute_time << " ms..." << endl;
	
	return filtered;
}

vector<float> CL_discrete_back_projection(cl::Device device, cl::Context context, vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans)
{
	vector<float> reconstruction(res*res, 0.0);
	float h = (float) 2/res;
	float pi = 4*atan(1);
	float scale = (float) 2*pi/rots;
	int half_scans=scans/2;
	
	double compute_time=0.0;
	
	cl::Program program = CL_create_program_from_source(context, device, "./Kernels/discrete_back_projection.cl");
	cl::Kernel kernel(program, "discrete_back_projection");
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, rots*scans*sizeof(float));
	cl::Buffer Reconstruction(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, res*res*sizeof(float));

	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	queue.enqueueWriteBuffer(Reconstruction, CL_TRUE, 0, res*res*sizeof(float), &reconstruction[0]);
	
	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Reconstruction);
	kernel.setArg(2, res);
	kernel.setArg(3, half_scans);
	kernel.setArg(4, scans);
	kernel.setArg(5, rots);
	kernel.setArg(6, scale);
	kernel.setArg(7, pi);
	kernel.setArg(8, h);
	
	cl::Event compute;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res), cl::NullRange, NULL, &compute); 
	queue.enqueueReadBuffer(Reconstruction, CL_TRUE, 0, res*res*sizeof(float), &reconstruction[0]);
	queue.finish();

	cl_ulong start = compute.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong stop = compute.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	compute_time += (double) (stop - start)/1000000;	
		
	// cout << "Discrete back projection applied to sinogram in " << compute_time << " ms..." << endl;
	
	return reconstruction;
}



// --------------------------------------------------------------------------------------------------------------------------------

// CT functions

unsigned int set_resolution(unsigned int m, unsigned int n)
{
	if (m == n)
	{
		return m;
	}
	else 
	{
		cout << "Non-square phantoms not supported at this point, aborting" << endl;
		exit(-1);	
	}
}

vector<float> salt_and_pepper_noise(unsigned int res, float amp) // Expects amplitude between 0 and 1
{
	vector<float> image(res*res, 0);
	
	random_device dev;
    default_random_engine rng(dev());
    uniform_int_distribution<int> dist(-127,127);
	
	for(int i=0; i<res; i++)
	{
		for(int j=0; j<res; j++)
		{
			image[i*res+j] = amp*dist(rng);
		}
	}
	
	return image;
}

// --------------------------------------------------------------------------------------------------------------------------------

vector<float> add_matrices(vector<float>& A, vector<float>& B)
{	
	if (A.size() == B.size())
	{
		vector<float> C(A.size(), 0.0);
		
		for(int i=0; i<A.size(); i++)
		{
			C[i] = A[i]+B[i];
		}
		
		return C;
	}
	else 
	{
		cout << "Can't add matrices that aren't the same size!" << endl;	
		exit(-1);
	}
}
vector<float> subtract_matrices(vector<float>& A, vector<float>& B)
{
	if (A.size() == B.size())
	{
		vector<float> C(A.size(), 0.0);
		
		for(int i=0; i<A.size(); i++)
		{
			C[i] = A[i] - B[i];
		}
		
		return C;
	}
	else 
	{
		cout << "Can't add matrices that aren't the same size!" << endl;
		exit(-1);	
	}
}

// --------------------------------------------------------------------------------------------------------------------------------

// Input reading and output writing

bool try_reading_float(string& input, float& output)
{
	try
	{
		output = stof(input);
	}
	catch(invalid_argument)
	{
		return false;
	}
	return true;
}

bool try_reading_uint(string& input, unsigned int& output)
{
	try
	{
		output = stoul(input);
	}
	catch(invalid_argument)
	{
		return false;
	}
	return true;
}

void split_string(string& input, vector<string>& output)
{
	stringstream ss(input);       // Insert the string into a stream
	string buffer;
    while (ss >> buffer)
	{
		output.push_back(buffer);
	}
}

vector<float> read_matrix_float(string filename, unsigned int& m, unsigned int& n)
{
	vector<float> matrix;
	
	vector<string> dimensions_string;
	vector<unsigned int> dimensions(2);
	vector<string> line_values_string;
	vector<float> line_values;
	float tmp;
	string input_buffer;
	
	ifstream data_file(filename);
	if(!data_file.is_open())
	{
		cout << "File couln't be opened!" << endl;
		exit(-1);
	}
	
	// Read first line and check for compatibility
	getline(data_file, input_buffer);
	split_string(input_buffer, dimensions_string);
	if (dimensions_string.size() == 2)
	{
		if(try_reading_uint(dimensions_string[0],dimensions[0]) and try_reading_uint(dimensions_string[1], dimensions[1]))
		{
			m = dimensions[0];
			n = dimensions[1];
			for(int i=0; i<m; i++)
			{
				getline(data_file, input_buffer);
				split_string(input_buffer, line_values_string);
				if(line_values_string.size()==n)
				{
					for(int k=0; k<n; k++)
					{
						if (try_reading_float(line_values_string[k], tmp))
						{
							line_values.push_back(tmp);
						}
						else 
						{
							cout << "File does not meet requirements!" << endl;
							exit(-1);
						}
					}
					matrix.insert(matrix.end(),line_values.begin(), line_values.end());
					
					// Empty buffers!
					
					input_buffer.clear();
					line_values_string.clear();
					line_values.clear();
				}
				else 
				{
					cout << "File does not meet requirements!" << endl;
					exit(-1);	
				}
			}
			return matrix;
		}
		else 
		{
			cout << "File does not meet requirements!" << endl;
			exit(-1);
		}
	}	
	else 
	{
		cout << "File does not meet requirements!" << endl;
		exit(-1);
	}
}

void print_matrix_float(vector<float> matrix, const unsigned int& m, const unsigned int& n)
{
	if (matrix.size() != m*n)
	{
		cout << "Matrix size: " << matrix.size() << ", Dimension: "<<m<<"*"<<n<<"="<<m*n<<endl;
		cout << "Corrupt matrix!" << endl;
		exit(-1);
	}
	else 
	{
		for(int i=0; i<m; i++)
		{
			for(int j=0; j<n; j++)
			{
				cout << fixed;
				cout.precision(3);
				cout << setw(10) << matrix[i*n+j]; 
			}
			cout << endl;
		}	
	}
}

void write_matrix_float_to_file(string filename, vector<float> matrix, const unsigned int& m, const unsigned int& n)
{
	if (matrix.size() != m*n)
	{
		cout << "Attempted to write corrupt matrix to file " << filename << ", aborting!" << endl;
		exit(-1);
	}
	
	else 
	{
		ofstream output_file(filename);
		if (output_file.is_open())
		{
			for(int i=0; i<m; i++)
			{
				for(int j=0; j<n; j++)
				{
					output_file << fixed;
					output_file.precision(3);
					output_file << setw(10) << matrix[i*n+j]; 
				}
				output_file << endl;
			}	 
		}
		else 
		{
			cout << "Couldn't create file!" << endl;
			exit(-1);	
		}
	}	
}

void write_matrix_float_to_pgm(string filename, vector<float> matrix, const unsigned int& m, const unsigned int& n) // m = rows of matrix, n = columns of matrix
{
	// Expects matrix with float values in the range of 0 and 255. Conversion necessary for matrices with different gray value spectra
	
	float conversion; // Dummy variable that is converted to unsigned char and clipped to 0..255
	unsigned char output_pixel;
	
	if (matrix.size() != m*n)
	{
		cout << "Attempted to write corrupt matrix to " << filename << ", aborting!" << endl;
		exit(-1);
	}
	
	ofstream output_file(filename, ios::out | ios::binary);
	if (output_file.is_open())
	{
		output_file << "P5" << endl;
		output_file << n << " " << m << endl; // Switch n and m for usual resolution statement of picture
		output_file << "255" << endl; // States maximum grey value
		
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
			{
				conversion = matrix[i*n+j] +0.5; // For rounding
				if (conversion < 0)
				{
					output_pixel = (unsigned char) 0;
					output_file << output_pixel; 
				}
				if (conversion > 255)
				{
					output_pixel= (unsigned char) 255;
					output_file << output_pixel;
				}
				else 
				{
					output_pixel = (unsigned char) conversion; 
					output_file << output_pixel;
				}
				
			}
			// output_file << endl; // Terminate pixel row 
		}
		
	}
	else 
	{
		cout << "Couldn't create file!" << endl;
		exit(-1);
	}
}

void write_matrix_float_to_pgm_normalised(string filename, vector<float> matrix, const unsigned int& m, const unsigned int& n) // m = rows of matrix, n = columns of matrix
{
	// Expects matrix with float values in the range of 0 and 255. Conversion necessary for matrices with different gray value spectra
	
	float min,max,conversion,scale; // Dummy variable that is converted to unsigned char and clipped to 0..255
	vector<float> output(m*n, 0);
	unsigned char output_pixel;
	
	if (matrix.size() != m*n)
	{
		cout << "Attempted to write corrupt matrix to " << filename << ", aborting!" << endl;
		exit(-1);
	}
	
	min = matrix[0]; // Initialise to value that actually is in the image
	max = matrix[0];
	
	for(int i=0; i<m; i++)
	{
		for(int j=0; j<n; j++)
		{
			if (matrix[i*n+j] > max)
			{
				max = matrix[i*n+j];
			}
			if (matrix[i*n+j] < min)
			{
				min = matrix[i*n+j];
			}
		}
	}
	
	// Normalise to {0,...,255}
	
	scale = 255/(max - min);
	
	for(int i=0; i<m; i++)
	{
		for(int j=0; j<m; j++)
		{
			output[i*m+j] = scale*(matrix[i*m+j]-min);
		}
	}
	
	ofstream output_file(filename, ios::out | ios::binary);
	if (output_file.is_open())
	{
		output_file << "P5" << endl;
		output_file << n << " " << m << endl; // Switch n and m for usual resolution statement of picture
		output_file << "255" << endl; // States maximum grey value
		
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
			{
				conversion = output[i*n+j] +0.5; // For rounding
				if (conversion > 255)
				{
					output_pixel= (unsigned char) 255;
					output_file << output_pixel;
				}
				else 
				{
					output_pixel = (unsigned char) conversion; 
					output_file << output_pixel;
				}
			}
		}
	}
	else 
	{
		cout << "Couldn't create file!" << endl;
		exit(-1);
	}
}

vector<float> read_pgm_to_matrix_float(string filename, unsigned int& m, unsigned int& n)
{
	string input_line;
	vector<string> line_values_string;
	vector<float> matrix;
	unsigned char* tmp;
	float dummy;
	
	ifstream input_file(filename, ios::binary);
	if (input_file.is_open())
	{
		getline(input_file, input_line);
		if(input_line.compare("P5") != 0)
		{
			cout << "Unknown file format in file " << filename << endl;
			exit(-1);
		}
		else 
		{
			getline(input_file, input_line);
			while(input_line.compare(0,1,"#")==0)
			{	
				input_line.clear(); // Line read was comment, discard
				getline(input_file, input_line);
			}	
			split_string(input_line, line_values_string);
			if (line_values_string.size()==2)
			{
				if(try_reading_uint(line_values_string[0], m) and try_reading_uint(line_values_string[1], n))
				{
					getline(input_file, input_line); // Read maximum grey value and discard
					input_line.clear();
						
					tmp = (unsigned char*) new unsigned char[m*n];
							
					for(int i=0; i<m*n; i++)
					{
						input_file.read(reinterpret_cast<char*>(tmp), m*n*sizeof(unsigned char));
					}
				
					for(int i=0; i<m; i++)
					{
						for(int j=0; j<n; j++)
						{
							dummy = (float) tmp[i*n+j];
							matrix.push_back(dummy);
						}
					}
				}
				else 
				{
					cout << "Corrupt file " << filename << endl;	
					exit(-1);
				}	
			}
			else 
			{
				cout << "Corrput file " << filename << endl;
				exit(-1);	
			}
		}
	}		
	else 
	{
		cout << "Couldn't open file " << filename << endl;
		exit(-1);	
	}
	return matrix;
}
