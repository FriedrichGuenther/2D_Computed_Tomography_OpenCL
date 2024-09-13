// ----------------------------------------------------------------------------------
// Filtered Backprojection [OpenCL], written by Friedrich Guenther
// Rev 9 [04.08.2021]
// ----------------------------------------------------------------------------------

// OpenCL includes
#define CL_HPP_TARGET_OPENCL_VERSION 120	// OpenCL 2.0 and higher are not widely supported anyways
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS 				// Allows for better error catching/handling
#if defined(__APPLE__)  					// C++ Wrapper for OpenCL
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif
#include "CL_err_helper.hpp"				// Translates OpenCL error codes to human readable format

// C++ includes
#include <iostream> 	// IO Streams (cout, cin, ...)
#include <fstream> 		// File streams
#include <vector> 		// Vectors
#include <string> 		// String library
#include <sstream> 		// Streams
#include <iomanip> 		// Output formating
#include <cmath> 		// Math functions
#include <random> 		// Random number generator
#include <chrono> 		// Timing

// Type savings for clocks
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// Actual macros
#define KERNEL_PATH(x) "./Kernels/"#x		// Path to kernel folder
#define DGPU 1								// Using host pointers for buffers and images if DGPU is set to 0
#define EXPERIMENTAL_IMAGES	1				// Using single channel floating point images if EXPERIMENTAL_IMAGES is 1, four channel floating point images otherwise
#define PINNED_BUFFERS 1					// Using pinned buffers and pinned images if PINNED_BUFFERS is 1; only has an effect if DGPUs is set to 1
#define LOCAL_RANGE cl::NullRange			// Global controll for local size, i.e. grouping of work items for execution in work groups
											// cl::NullRange lets runtime choose "appropriate local size",

#if PINNED_BUFFERS==1
#define PINNED_FLAG(x) x|CL_MEM_ALLOC_HOST_PTR
#endif
#if PINNED_BUFFERS==0
#define PINNED_FLAG(x) x
#endif

// --------------------------------------------------------------------------------------------------------------------------------
// Function declarations
// --------------------------------------------------------------------------------------------------------------------------------

// Phantom
void raster_ellipse(std::vector<float>& image, int res, std::vector<float> focus, float semimajor, float semiminor, float gray_value);
void raster_ellipse_rotated(std::vector<float>& image, int res, std::vector<float> focus, float semimajor, float semiminor, float gray_value, float angle);
std::vector<float> raster_shepp_logan(unsigned int res);

// OpenCL host functions
void set_up_platform_and_devices(cl::Platform& cpu_platform, cl::Device& cpu_device);
cl::Program CL_create_program_source(cl::Context context, std::string filename);

// Buffer-Crunchers
std::vector<float> CL_compute_sinogram(cl::Device device, cl::Context context, std::vector<float>& phantom, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_fast_ram_lak_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_fast_shepp_logan_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_fast_cosine_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_discrete_back_projection(cl::Device device, cl::Context context, std::vector<float> &sinogram, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_ram_lak_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, float b, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_shepp_logan_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, float b, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_cosine_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, float b, unsigned int res, unsigned int rots, unsigned int scans);

// Image-Crunchers
std::vector<float> CL_compute_sinogram_img(cl::Device device, cl::Context context, std::vector<float>& phantom_img, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_fast_ram_lak_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_fast_shepp_logan_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_fast_cosine_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_discrete_back_projection_img(cl::Device device, cl::Context context, std::vector<float> &sinogram_img, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_ram_lak_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, float b, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_shepp_logan_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, float b, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> CL_cosine_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, float b, unsigned int res, unsigned int rots, unsigned int scans);

// Miscellaneous for buffer <-> image2d_t
std::vector<float> create_image2d_type(std::vector<float>& image); 				// Needed for conversion to 4 channel float images
std::vector<float> reduce_to_buffer(std::vector<float>& image); 				// Needed for saving of 4 channel float image
void remove_artifacts(std::vector<float>& reconstruction, unsigned int res);	// Sets pixels outside the unit disk to zero
unsigned int set_resolution(unsigned int m, unsigned int n);					// Used for reading phantoms

// Execution functions
std::vector<duration<double, std::milli>> bench_for_execution_time(std::vector<float> (*crunch)(cl::Device, cl::Context, std::vector<float>&, unsigned int, unsigned int, unsigned int), unsigned int iterations, cl::Device device, cl::Context context, std::vector<float>& image, unsigned int res, unsigned int rots, unsigned int scans);
std::vector<float> run_function(std::vector<float> (*crunch)(cl::Device, cl::Context, std::vector<float>&, unsigned int, unsigned int, unsigned int), cl::Device device, cl::Context context, std::vector<float>& image, unsigned int res, unsigned int rots, unsigned int scans);
// Benchmark and image producers
void run_benchmark(std::vector<float>& phantom, unsigned int iterations, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans);
void produce_images(std::vector<float>& phantom, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans);
void noisy_data(std::vector<float>& phantom, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans);
void run_benchmark_img(std::vector<float>& phantom_img, unsigned int iterations, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans);
void produce_images_img(std::vector<float>& phantom_img, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans);
void noisy_data_img(std::vector<float>& phantom_img, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans);

// CT functions
unsigned int set_resolution(unsigned int m, unsigned int n);
std::vector<float> salt_and_pepper_noise(unsigned int res, float amp);
std::vector<float> add_matrices(std::vector<float>& A, std::vector<float>& B);
std::vector<float> subtract_matrices(std::vector<float>& A, std::vector<float>& B);
std::vector<float> absolute_value_of_matrix_difference(std::vector<float>& A, std::vector<float>& B);

// IO functions and miscellaneous
bool try_reading_float(std::string& input, float& output);
bool try_reading_uint(std::string& input, unsigned int& output);
void split_string(std::string& input, std::vector<std::string>& output);
void print_matrix_float(std::vector<float> matrix, const unsigned int& m, const unsigned int& n);
void write_matrix_float_to_file(std::string filename, std::vector<float> matrix, const unsigned int& m, const unsigned int& n);
void write_matrix_float_to_pgm(std::string filename, std::vector<float> matrix, const unsigned int& m, const unsigned int& n);
void write_matrix_float_to_pgm_normalised(std::string filename, std::vector<float> matrix, const unsigned int& m, const unsigned int& n);
std::vector<float> read_matrix_float(std::string filename, unsigned int& m, unsigned int& n);
std::vector<float> read_pgm_to_matrix_float(std::string filename, unsigned int& m, unsigned int& n);

// --------------------------------------------------------------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------------------------------------------------------------

int main(void)
{
	try // OpenCL host
	{
		// Setting up GPU platform and device
		cl::Platform platform;
		cl::Device device;
		set_up_platform_and_devices(platform, device);
		cl::Context context(device);

		// Constants
		std::vector<unsigned int> resolutions = {1024};
		unsigned int res,m,n,rots,scans,underscans,iterations;
		bool use_buffers = false;

		for(int i=0; i<resolutions.size(); i++)
		{
			res = resolutions[i];

			rots = res;
			underscans = 1;
			scans = res/underscans;
			iterations = 5;

			std::cout << "Rastering Shepp-Logan phantom..." << std::endl;
			std::vector<float> phantom = raster_shepp_logan(res);
			std::cout << "Working on Shepp-Logan phantom in "<< res << "x" << res << " pixels..." << std::endl;

			if(use_buffers)
			{
				run_benchmark(phantom, iterations, device, context, res, rots, scans);
				produce_images(phantom, device, context, res, rots, scans);
				// noisy_data(phantom, device, context, res, rots, scans);
			}

			if(!use_buffers) // At some point, one could use this switch
			{
				#if EXPERIMENTAL_IMAGES==1
				std::cout << "Using experimental images..." << std::endl;
 				run_benchmark_img(phantom, iterations, device, context, res, rots, scans);
				produce_images_img(phantom, device, context, res, rots, scans);
				//noisy_data_img(phantom, device, context, res, rots, scans);
				#endif

				#if EXPERIMENTAL_IMAGES==0
				std::cout << "Using standard images..." << std::endl;
				std::vector<float> phantom_img = create_image2d_type(phantom);
				run_benchmark_img(phantom_img, iterations, device, context, res, rots, scans);
				//produce_images_img(phantom_img, device, context, res, rots, scans);
				// noisy_data_img(phantom_img, device, context, res, rots, scans);
				#endif
			}
		}

		return EXIT_SUCCESS;
	}
	catch(const cl::Error& clExp)
	{
		std::cout << "OpenCL Exception: " << clExp.what() << " with error code " << opencl_err_to_str(clExp.err()) << " (" << clExp.err() << ")" << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cout << "Other exception: " << e.what() << std::endl;
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------------------------
// OpenCL Host
// --------------------------------------------------------------------------------------------------------------------------------

void set_up_platform_and_devices(cl::Platform& gpu_platform, cl::Device& gpu_device)
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices_per_platform;
	std::vector<cl::Device> gpu_devices;

	unsigned int num_platforms;
	unsigned int num_target_device=0;
	unsigned int num_devices;
	unsigned int total_gpus;
	std::string input;

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
		throw std::runtime_error("No GPU devices available! Check OpenCL installation!");
	}

	if(total_gpus == 1)
	{
		gpu_device = gpu_devices[0];
		gpu_platform = gpu_devices[0].getInfo<CL_DEVICE_PLATFORM>();
		std::cout << "Selected " << gpu_device.getInfo<CL_DEVICE_NAME>() << " on platform " << gpu_platform.getInfo<CL_PLATFORM_NAME>() << " as GPU device." << std::endl;
	}

	if(total_gpus > 1)
	{
		std::cout << "Available GPU devices: " << std::endl;
		for(int i=0; i<total_gpus; i++)
		{
			std::cout << "[" << i+1 << "] " << gpu_devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		}
		std::cout << "Please enter target GPU device: ";
		getline(std::cin, input);
		while(!try_reading_uint(input, num_target_device) or num_target_device < 1 or num_target_device > total_gpus)
		{
			std::cout << "Please enter target GPU device: ";
			std::getline(std::cin, input);
		}
		num_target_device--;
		gpu_device = gpu_devices[num_target_device];
		gpu_platform = gpu_device.getInfo<CL_DEVICE_PLATFORM>();
		std::cout << "Selected " << gpu_device.getInfo<CL_DEVICE_NAME>() << " on platform " << gpu_platform.getInfo<CL_PLATFORM_NAME>() << " as GPU device." << std::endl;
		num_target_device=0;
		std::cin.clear();
	}

}

cl::Program CL_create_program_from_source(cl::Context context, cl::Device device, std::string filename)
{
	std::ifstream program_file(filename);
	std::string program_string(std::istreambuf_iterator<char>(program_file), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source { program_string };
	cl::Program program(context, source);
	try
	{
		program.build("-cl-std=CL1.2 -cl-single-precision-constant -cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only"); //-cl-fast-relaxed-math
	}
	catch(cl::Error e)
	{
		std::cout << e.what() << " ; Error code " << e.err() << std::endl;
		std::string build_log;
		build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
		std::cout << build_log << std::endl;
	}

	return program;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Buffer crunchers
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_compute_sinogram(cl::Device device, cl::Context context, std::vector<float>& image, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> sinogram(rots*scans, 0.0);

	int underscans = res/scans;
	float h = (float) 2/res;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(compute_sinogram.cl));
	cl::Kernel kernel(program, "compute_sinogram");
	cl::CommandQueue queue(context, device);

	#if DGPU==1
	cl::Buffer Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), res*res*sizeof(float)); // Pinned host memory for a little extra speed
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_WRITE_ONLY), rots*scans*sizeof(float));
	queue.enqueueWriteBuffer(Image, CL_TRUE, 0, res*res*sizeof(float), &image[0]);
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	#if DGPU==0
	cl::Buffer Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, res*res*sizeof(float), &image[0]);
	cl::Buffer Sinogram(context, CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	kernel.setArg(0, Image);
	kernel.setArg(1, Sinogram);
	kernel.setArg(2, res);
	kernel.setArg(3, scans);
	kernel.setArg(4, underscans);
	kernel.setArg(5, rots);
	kernel.setArg(6, pi);
	kernel.setArg(7, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(scans, rots), LOCAL_RANGE, NULL); // cl::NDRange(16,16)

	#if DGPU==1
	queue.enqueueReadBuffer(Sinogram, CL_TRUE, 0, scans*rots*sizeof(float), &sinogram[0]);
	#endif
	queue.finish();

	return sinogram;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Fast filters
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_fast_ram_lak_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> filtered(rots*scans, 0.0);
	float h = (float) 2/scans;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(fast_ram_lak_filter.cl));
	cl::Kernel kernel(program, "fast_ram_lak_filter");
	cl::CommandQueue queue(context, device);

	#if DGPU==1
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_READ_ONLY), rots*scans*sizeof(float));
	cl::Buffer Filtered(context, PINNED_FLAG(CL_MEM_READ_WRITE), rots*scans*sizeof(float));
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	#if DGPU==0
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &filtered[0]);
	#endif

	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL); // cl::NDRange(16,16)
	#if DGPU==1
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	#endif
	queue.finish();

	return filtered;
}

std::vector<float> CL_fast_shepp_logan_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> filtered(rots*scans, 0.0);
	float h = (float) 2/scans;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(fast_shepp_logan_filter.cl));
	cl::Kernel kernel(program, "fast_shepp_logan_filter");
	cl::CommandQueue queue(context, device);

	#if DGPU==1
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_READ_ONLY), rots*scans*sizeof(float));
	cl::Buffer Filtered(context, PINNED_FLAG(CL_MEM_READ_WRITE), rots*scans*sizeof(float));
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	#if DGPU==0
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &filtered[0]);
	#endif

	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	#endif
	queue.finish();

	return filtered;
}

std::vector<float> CL_fast_cosine_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> filtered(rots*scans, 0.0);
	float h = (float) 2/scans;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(fast_cosine_filter.cl));
	cl::Kernel kernel(program, "fast_cosine_filter");
	cl::CommandQueue queue(context, device);

	#if DGPU==1
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_READ_ONLY), rots*scans*sizeof(float));
	cl::Buffer Filtered(context, PINNED_FLAG(CL_MEM_READ_WRITE), rots*scans*sizeof(float));
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	#if DGPU==0
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &filtered[0]);
	#endif

	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	#endif
	queue.finish();

	return filtered;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Discrete Backprojection
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_discrete_back_projection(cl::Device device, cl::Context context, std::vector<float>& sinogram, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> reconstruction(res*res, 0.0);
	float h = (float) 2/res;
	float pi = M_PI;
	float scale = (float) 2*pi/rots;
	int half_scans=scans/2;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(discrete_back_projection.cl));
	cl::Kernel kernel(program, "discrete_back_projection");
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	#if DGPU==1
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_READ_ONLY), rots*scans*sizeof(float));
	cl::Buffer Reconstruction(context, PINNED_FLAG(CL_MEM_READ_WRITE), res*res*sizeof(float));
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	queue.enqueueWriteBuffer(Reconstruction, CL_TRUE, 0, res*res*sizeof(float), &reconstruction[0]);
	#endif

	#if DGPU==0
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	cl::Buffer Reconstruction(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, res*res*sizeof(float), &reconstruction[0]);
	#endif

	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Reconstruction);
	kernel.setArg(2, res);
	kernel.setArg(3, half_scans);
	kernel.setArg(4, scans);
	kernel.setArg(5, rots);
	kernel.setArg(6, scale);
	kernel.setArg(7, pi);
	kernel.setArg(8, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res, res), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadBuffer(Reconstruction, CL_TRUE, 0, res*res*sizeof(float), &reconstruction[0]);
	#endif
	queue.finish();

	return reconstruction;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Slow filters
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_ram_lak_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, float b, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> filtered(rots*scans, 0.0);

	float h = (float) 2/scans;
	float pi = M_PI;
	float epsilon = pow(10,-8);

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(ram_lak_filter.cl));
	cl::Kernel kernel(program, "ram_lak_filter");
	cl::CommandQueue queue(context, device);

	#if DGPU==1
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_READ_ONLY), rots*scans*sizeof(float));
	cl::Buffer Filtered(context, PINNED_FLAG(CL_MEM_READ_WRITE), rots*scans*sizeof(float));
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	#if DGPU==0
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &filtered[0]);
	#endif

	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	kernel.setArg(6, epsilon);
	kernel.setArg(7, b);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	#endif
	queue.finish();

	return filtered;
}

std::vector<float> CL_shepp_logan_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, float b, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> filtered(rots*scans, 0.0);

	float h = (float) 2/scans;
	float pi = M_PI;
	float epsilon = pow(10,-8);

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(shepp_logan_filter.cl));
	cl::Kernel kernel(program, "shepp_logan_filter");
	cl::CommandQueue queue(context, device);

	#if DGPU==1
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_READ_ONLY), rots*scans*sizeof(float));
	cl::Buffer Filtered(context, PINNED_FLAG(CL_MEM_READ_WRITE), rots*scans*sizeof(float));
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	#if DGPU==0
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &filtered[0]);
	#endif

	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	kernel.setArg(6, epsilon);
	kernel.setArg(7, b);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	#endif
	queue.finish();

	return filtered;
}

std::vector<float> CL_cosine_filter(cl::Device device, cl::Context context, std::vector<float>& sinogram, float b, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> filtered(rots*scans, 0.0);

	float h = (float) 2/scans;
	float pi = M_PI;
	float epsilon = pow(10,-8);

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(cosine_filter.cl));
	cl::Kernel kernel(program, "cosine_filter");
	cl::CommandQueue queue(context, device);

	#if DGPU==1
	cl::Buffer Sinogram(context, PINNED_FLAG(CL_MEM_READ_ONLY), rots*scans*sizeof(float));
	cl::Buffer Filtered(context, PINNED_FLAG(CL_MEM_READ_WRITE), rots*scans*sizeof(float));
	queue.enqueueWriteBuffer(Sinogram, CL_TRUE, 0, rots*scans*sizeof(float), &sinogram[0]);
	#endif

	#if DGPU==0
	cl::Buffer Sinogram(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &sinogram[0]);
	cl::Buffer Filtered(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rots*scans*sizeof(float), &filtered[0]);
	#endif

	kernel.setArg(0, Sinogram);
	kernel.setArg(1, Filtered);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	kernel.setArg(6, epsilon);
	kernel.setArg(7, b);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadBuffer(Filtered, CL_TRUE, 0, rots*scans*sizeof(float), &filtered[0]);
	#endif
	queue.finish();

	return filtered;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Image crunchers
// --------------------------------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------------------------
// Sinogram
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_compute_sinogram_img(cl::Device device, cl::Context context, std::vector<float>& phantom_img, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> sinogram_img(rots*scans, 0.0);
	#endif

	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> sinogram_img(rots*scans*4, 0.0);
	#endif

	int underscans = res/scans;
	float h = (float) 2/res;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(compute_sinogram_img.cl));
	cl::Kernel kernel(program, "compute_sinogram");
	cl::CommandQueue queue(context, device);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Phantom_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, res, res);
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_WRITE_ONLY), rgb, scans, rots);

	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {res, res, 1};
	std::array<cl::size_type, 3> region_dst {scans, rots, 1};

	queue.enqueueWriteImage(Phantom_Image, CL_TRUE, origin, region, 0, 0, &phantom_img[0]);
	#endif

	#if DGPU==0
	cl::Image2D Phantom_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, res, res, 0, &phantom_img[0]);
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	#endif

	kernel.setArg(0, Phantom_Image);
	kernel.setArg(1, Sinogram_Image);
	kernel.setArg(2, res);
	kernel.setArg(3, scans);
	kernel.setArg(4, underscans);
	kernel.setArg(5, rots);
	kernel.setArg(6, pi);
	kernel.setArg(7, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(scans, rots), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadImage(Sinogram_Image, CL_TRUE, origin, region_dst, 0, 0, &sinogram_img[0]);
	#endif
	queue.finish();

	return sinogram_img;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Fast filters
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_fast_ram_lak_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> filtered_img(rots*scans, 0.0);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> filtered_img(rots*scans*4, 0.0);
	#endif

	float h = (float) 2/scans;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(fast_ram_lak_filter_img.cl));
	cl::Kernel kernel(program, "fast_ram_lak_filter");
	cl::CommandQueue queue(context, device);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, scans, rots);
	cl::Image2D Filtered_Image(context, PINNED_FLAG(CL_MEM_READ_WRITE), rgb, scans, rots);
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {scans, rots, 1};
	queue.enqueueWriteImage(Sinogram_Image, CL_TRUE, origin, region, 0, 0, &sinogram_img[0]);
	#endif

	#if DGPU==0
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	cl::Image2D Filtered_Image(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &filtered_img[0]);
	#endif

	kernel.setArg(0, Sinogram_Image);
	kernel.setArg(1, Filtered_Image);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadImage(Filtered_Image, CL_TRUE, origin, region, 0, 0, &filtered_img[0]);
	#endif
	queue.finish();

	return filtered_img;
}

std::vector<float> CL_fast_shepp_logan_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> filtered_img(rots*scans, 0.0);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> filtered_img(rots*scans*4, 0.0);
	#endif

	float h = (float) 2/scans;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(fast_shepp_logan_filter_img.cl));
	cl::Kernel kernel(program, "fast_shepp_logan_filter");
	cl::CommandQueue queue(context, device);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, scans, rots);
	cl::Image2D Filtered_Image(context, PINNED_FLAG(CL_MEM_READ_WRITE), rgb, scans, rots);
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {scans, rots, 1};
	queue.enqueueWriteImage(Sinogram_Image, CL_TRUE, origin, region, 0, 0, &sinogram_img[0]);
	#endif

	#if DGPU==0
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	cl::Image2D Filtered_Image(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &filtered_img[0]);
	#endif

	kernel.setArg(0, Sinogram_Image);
	kernel.setArg(1, Filtered_Image);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadImage(Filtered_Image, CL_TRUE, origin, region, 0, 0, &filtered_img[0]);
	#endif
	queue.finish();

	return filtered_img;
}

std::vector<float> CL_fast_cosine_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> filtered_img(rots*scans, 0.0);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> filtered_img(rots*scans*4, 0.0);
	#endif

	float h = (float) 2/scans;
	float pi = M_PI;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(fast_cosine_filter_img.cl));
	cl::Kernel kernel(program, "fast_cosine_filter");
	cl::CommandQueue queue(context, device);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, scans, rots);
	cl::Image2D Filtered_Image(context, PINNED_FLAG(CL_MEM_READ_WRITE), rgb, scans, rots);
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {scans, rots, 1};
	queue.enqueueWriteImage(Sinogram_Image, CL_TRUE, origin, region, 0, 0, &sinogram_img[0]);
	#endif

	#if DGPU==0
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	cl::Image2D Filtered_Image(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &filtered_img[0]);
	#endif

	kernel.setArg(0, Sinogram_Image);
	kernel.setArg(1, Filtered_Image);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadImage(Filtered_Image, CL_TRUE, origin, region, 0, 0, &filtered_img[0]);
	#endif
	queue.finish();

	return filtered_img;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Discrete Backprojection
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_discrete_back_projection_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> reconstruction(res*res, 0.0);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> reconstruction(res*res, 0.0);
	#endif

	float h = (float) 2/res;
	float pi = M_PI;
	float scale = (float) 2*pi/rots;
	int half_scans=scans/2;

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(discrete_back_projection_img.cl));
	cl::Kernel kernel(program, "discrete_back_projection");
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, scans, rots);
	cl::Buffer Reconstruction(context, PINNED_FLAG(CL_MEM_READ_WRITE), res*res*sizeof(float));
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {scans, rots, 1};
	queue.enqueueWriteImage(Sinogram_Image, CL_TRUE, origin, region, 0, 0, &sinogram_img[0]);
	queue.enqueueWriteBuffer(Reconstruction, CL_TRUE, 0, res*res*sizeof(float), &reconstruction[0]);
	#endif

	#if DGPU==0
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	cl::Buffer Reconstruction(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, res*res*sizeof(float), &reconstruction[0]);
	#endif

	kernel.setArg(0, Sinogram_Image);
	kernel.setArg(1, Reconstruction);
	kernel.setArg(2, res);
	kernel.setArg(3, half_scans);
	kernel.setArg(4, scans);
	kernel.setArg(5, rots);
	kernel.setArg(6, scale);
	kernel.setArg(7, pi);
	kernel.setArg(8, h);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res, res), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadBuffer(Reconstruction, CL_TRUE, 0, res*res*sizeof(float), &reconstruction[0]);
	#endif
	queue.finish();

	return reconstruction;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Slow filters
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> CL_ram_lak_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, float b, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> filtered_img(rots*scans, 0.0);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> filtered_img(rots*scans*4, 0.0);
	#endif

	float h = (float) 2/scans;
	float pi = M_PI;
	float epsilon = pow(10,-8);

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(ram_lak_filter_img.cl));
	cl::Kernel kernel(program, "ram_lak_filter");
	cl::CommandQueue queue(context, device);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, scans, rots);
	cl::Image2D Filtered_Image(context, PINNED_FLAG(CL_MEM_READ_WRITE), rgb, scans, rots);
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {scans, rots, 1};
	queue.enqueueWriteImage(Sinogram_Image, CL_TRUE, origin, region, 0, 0, &sinogram_img[0]);
	#endif

	#if DGPU==0
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	cl::Image2D Filtered_Image(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &filtered_img[0]);
	#endif

	kernel.setArg(0, Sinogram_Image);
	kernel.setArg(1, Filtered_Image);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	kernel.setArg(6, epsilon);
	kernel.setArg(7, b);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadImage(Filtered_Image, CL_TRUE, origin, region, 0, 0, &filtered_img[0]);
	#endif
	queue.finish();

	return filtered_img;
}

std::vector<float> CL_shepp_logan_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, float b, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> filtered_img(rots*scans, 0.0);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> filtered_img(rots*scans*4, 0.0);
	#endif

	float h = (float) 2/scans;
	float pi = M_PI;
	float epsilon = pow(10,-8);

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(shepp_logan_filter_img.cl));
	cl::Kernel kernel(program, "shepp_logan_filter");
	cl::CommandQueue queue(context, device);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, scans, rots);
	cl::Image2D Filtered_Image(context, PINNED_FLAG(CL_MEM_READ_WRITE), rgb, scans, rots);
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {scans, rots, 1};
	queue.enqueueWriteImage(Sinogram_Image, CL_TRUE, origin, region, 0, 0, &sinogram_img[0]);
	#endif

	#if DGPU==0
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	cl::Image2D Filtered_Image(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &filtered_img[0]);
	#endif

	kernel.setArg(0, Sinogram_Image);
	kernel.setArg(1, Filtered_Image);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	kernel.setArg(6, epsilon);
	kernel.setArg(7, b);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadImage(Filtered_Image, CL_TRUE, origin, region, 0, 0, &filtered_img[0]);
	#endif
	queue.finish();

	return filtered_img;
}

std::vector<float> CL_cosine_filter_img(cl::Device device, cl::Context context, std::vector<float>& sinogram_img, float b, unsigned int res, unsigned int rots, unsigned int scans)
{
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> filtered_img(rots*scans, 0.0);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> filtered_img(rots*scans*4, 0.0);
	#endif

	float h = (float) 2/scans;
	float pi = M_PI;
	float epsilon = pow(10,-8);

	cl::Program program = CL_create_program_from_source(context, device, KERNEL_PATH(cosine_filter_img.cl));
	cl::Kernel kernel(program, "cosine_filter");
	cl::CommandQueue queue(context, device);

	#if EXPERIMENTAL_IMAGES==1
	cl::ImageFormat rgb(CL_R, CL_FLOAT);
	#endif
	#if EXPERIMENTAL_IMAGES==0
	cl::ImageFormat rgb(CL_RGBA, CL_FLOAT);
	#endif

	#if DGPU==1
	cl::Image2D Sinogram_Image(context, PINNED_FLAG(CL_MEM_READ_ONLY), rgb, scans, rots);
	cl::Image2D Filtered_Image(context, PINNED_FLAG(CL_MEM_READ_WRITE), rgb, scans, rots);
	std::array<cl::size_type, 3> origin {0,0,0};
	std::array<cl::size_type, 3> region {scans, rots, 1};
	queue.enqueueWriteImage(Sinogram_Image, CL_TRUE, origin, region, 0, 0, &sinogram_img[0]);
	#endif

	#if DGPU==0
	cl::Image2D Sinogram_Image(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &sinogram_img[0]);
	cl::Image2D Filtered_Image(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, rgb, scans, rots, 0, &filtered_img[0]);
	#endif

	kernel.setArg(0, Sinogram_Image);
	kernel.setArg(1, Filtered_Image);
	kernel.setArg(2, scans);
	kernel.setArg(3, rots);
	kernel.setArg(4, pi);
	kernel.setArg(5, h);
	kernel.setArg(6, epsilon);
	kernel.setArg(7, b);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rots, scans), LOCAL_RANGE, NULL);
	#if DGPU==1
	queue.enqueueReadImage(Filtered_Image, CL_TRUE, origin, region, 0, 0, &filtered_img[0]);
	#endif
	queue.finish();

	return filtered_img;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Miscellaneous for buffer <-> image2d_t
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<float> create_image2d_type(std::vector<float>& image)
{
	int k = image.size();
	std::vector<float> image_img(k*4, 0.0);
	for(int i=0; i<k; i++)
		image_img[4*i] = image[i];

	return image_img;
}

std::vector<float> reduce_to_buffer(std::vector<float>& image)
{
	int k = image.size();
	if ( (k&3) == 0)
	{
		int j = k/4;
		std::vector<float> output(j, 0.0);
		for(int i=0; i<j; i++)
		{
			output[i] = image[4*i];
		}
		return output;
	}
	else
		throw std::runtime_error("Corrupt image in >>reduce_to_buffer<<!");

}

void remove_artifacts(std::vector<float>& reconstruction, unsigned int res)
{
	for(int i=0; i<res; i++)
	{
		for(int j=0; j<res; j++)
		{
			if( sqrt( (i-res/2)*(i-res/2) + (j-res/2)*(j-res/2) ) > res/2)
			{
				reconstruction[i*res+j] = 0;
			}
		}
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// Execution functions
// --------------------------------------------------------------------------------------------------------------------------------

std::vector<duration<double, std::milli>> bench_for_execution_time(std::vector<float> (*crunch)(cl::Device, cl::Context, std::vector<float>&, unsigned int, unsigned int, unsigned int), unsigned int iterations, cl::Device device, cl::Context context, std::vector<float>& image, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<duration<double, std::milli>> times(3); // Total duration, average, standard deviation
	std::vector<duration<double, std::milli>> individual_runs(iterations);
	std::vector<float> dummy;
	double standard_deviation=0;

	auto t1 = high_resolution_clock::now();
	auto t2 = high_resolution_clock::now();
	times[0] = t1-t1; // Set first component to zero
	for(int i=0; i<iterations; i++)
	{
		t1 = high_resolution_clock::now();
		dummy = crunch(device, context, image, res, rots, scans);
		t2 = high_resolution_clock::now();
		times[0] += t2-t1;
		individual_runs[i] = t2-t1;
	}
	times[1] = duration<double, std::milli>(times[0].count()/iterations);
	for(int i=0; i<iterations; i++)
	{
		standard_deviation += (individual_runs[i].count() - times[1].count())*(individual_runs[i].count() - times[1].count());
	}
	standard_deviation *= (double) 1/iterations;
	standard_deviation = sqrt(standard_deviation);
	times[2] = duration<double, std::milli>(standard_deviation);

	return times;
}

std::vector<float> run_function(std::vector<float> (*crunch)(cl::Device, cl::Context, std::vector<float>&, unsigned int, unsigned int, unsigned int), cl::Device device, cl::Context context, std::vector<float>& image, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<float> output;
	output = crunch(device, context, image, res, rots, scans);
	return output;
}

void run_benchmark(std::vector<float>& phantom, unsigned int iterations, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<duration<double, std::milli>> times;
	std::vector<float> sinogram;

	std::cout << "Running benchmark using buffers..." << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
	std::cout << "| Function                        | total [ms] | avg [ms] | stdev [ms] |  N |" << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
	std::cout.precision(2);
	std::cout << std::fixed;

	auto function = CL_compute_sinogram;
	times = bench_for_execution_time(function, iterations, device, context, phantom, res, rots, scans);
	std::cout << std::right << "| CL_compute_sinogram             | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	if(rots*scans != res*res)
	{
		phantom = std::vector<float>(rots*scans,0.0);
	}

	function = CL_fast_ram_lak_filter;
	times = bench_for_execution_time(function, iterations, device, context, phantom, res, rots, scans);
	std::cout << std::right << "| CL_fast_ram_lak_filter          | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	function = CL_fast_shepp_logan_filter;
	times = bench_for_execution_time(function, iterations, device, context, phantom, res, rots, scans);
	std::cout << std::right << "| CL_fast_shepp_logan_filter      | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	function = CL_fast_cosine_filter;
	times = bench_for_execution_time(function, iterations, device, context, phantom, res, rots, scans);
	std::cout << std::right << "| CL_fast_cosine_filter           | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	function = CL_discrete_back_projection;
	times = bench_for_execution_time(function, iterations, device, context, phantom, res, rots, scans);
	std::cout << std::right << "| CL_discrete_back_projection     | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
}

void produce_images(std::vector<float>& phantom, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans)
{
	auto function = CL_compute_sinogram;
	std::vector<float> sinogram = run_function(function, device, context, phantom, res, rots, scans);
	std::cout << "Computed sinogram..." << std::endl;

	function = CL_fast_ram_lak_filter;
	std::vector<float> rlf = run_function(function, device, context, sinogram, res, rots, scans);
	std::cout << "Filtered sinogram with fast Ram-Lak filter..." << std::endl;

	function = CL_fast_shepp_logan_filter;
	std::vector<float> slf = run_function(function, device, context, sinogram, res, rots, scans);
	std::cout << "Filtered sinogram with fast Shepp-Logan filter..." << std::endl;

	function = CL_fast_cosine_filter;
	std::vector<float> cf = run_function(function, device, context, sinogram, res, rots, scans);
	std::cout << "Filtered sinogram with fast Cosine filter..." << std::endl;

	function = CL_discrete_back_projection; // Attention: Discrete back projection returns buffer, not image2d_t!
	std::vector<float> dbp = run_function(function, device, context, sinogram, res, rots, scans);
	std::cout << "Performed back projection of sinogram..." << std::endl;

	std::vector<float> dbp_rlf = run_function(function, device, context, rlf, res, rots, scans);
	std::cout << "Performed back projection of Ram-Lak filtered sinogram..." << std::endl;

	std::vector<float> dbp_slf = run_function(function, device, context, slf, res, rots, scans);
	std::cout << "Performed back projection of Shepp-Logan filtered sinogram..." << std::endl;

	function = CL_discrete_back_projection;
	std::vector<float> dbp_cf = run_function(function, device, context, cf, res, rots, scans);
	std::cout << "Performed back projection of Cosine filtered sinogram... " << std::endl;

	remove_artifacts(dbp_rlf, res);
	remove_artifacts(dbp_slf, res);
	remove_artifacts(dbp_cf, res);
	std::vector<float> err_dbp = absolute_value_of_matrix_difference(phantom, dbp);
	std::vector<float> err_dbp_rlf = absolute_value_of_matrix_difference(phantom, dbp_rlf);
	std::vector<float> err_dbp_slf = absolute_value_of_matrix_difference(phantom, dbp_slf);
	std::vector<float> err_dbp_cf = absolute_value_of_matrix_difference(phantom, dbp_cf);

	std::string outfile = "./Output/sinogram_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm(outfile, sinogram, scans, rots);

	outfile = "./Output/sinogram_rlf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, rlf, scans, rots);
	outfile = "./Output/sinogram_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, slf, scans, rots);
	outfile = "./Output/sinogram_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, cf, scans, rots);

	outfile = "./Output/dbp_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp, res, res);
	outfile = "./Output/dbp_rlf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp_rlf, res, res);
	outfile = "./Output/dbp_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp_slf, res, res);
	outfile = "./Output/dbp_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp_cf, res, res);

	outfile = "./Output/err_dbp_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp, res, res);
	outfile = "./Output/err_dbp_rlf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp_rlf, res, res);
	outfile = "./Output/err_dbp_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp_slf, res, res);
	outfile = "./Output/err_dbp_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp_cf, res, res);

	std::cout << "Wrote results to disk." << std::endl;
}

void noisy_data(std::vector<float>& phantom, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans)
{
	float pi = M_PI;
	float h = (float) 2/res;

	std::vector<float> sinogram_standard = CL_compute_sinogram(device, context, phantom, res, rots, scans);
	std::cout << "Computed sinogram..." << std::endl;

	std::vector<float> salt_and_pepper = salt_and_pepper_noise(res, 0.4);
	std::vector<float> sinogram = add_matrices(sinogram_standard, salt_and_pepper);

	write_matrix_float_to_pgm_normalised("phantom.pgm", phantom, res, res);
	write_matrix_float_to_pgm_normalised("sinogram_standard.pgm", sinogram, res, res);

	for(int i=0; i<3; i++)
	{
		std::vector<float> slf = CL_shepp_logan_filter(device, context, sinogram, pi/(3*(2*i+1)*h), res, rots, scans);
		std::cout << "Computed Shepp-Logan filter..." << std::endl;
		std::vector<float> cf = CL_cosine_filter(device, context, sinogram, pi/(3*(2*i+1)*h), res, rots, scans);
		std::cout << "Computed Cosine filter..." << std::endl;
		std::vector<float> dbp_slf = CL_discrete_back_projection(device, context, slf, res, rots, scans);
		std::cout << "Discrete back projection of Shepp-Logan filtered sinogram performed..." << std::endl;
		std::vector<float> dbp_cf = CL_discrete_back_projection(device, context, cf, res, rots, scans);
		std::cout << "Discrete back projection of Cosine filtered sinogram performed..." << std::endl;

		std::string outfile = "./Output/sinogram_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, slf, scans, rots);
		outfile = "./Output/sinogram_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, cf, scans, rots);

		remove_artifacts(dbp_slf, res);
		remove_artifacts(dbp_cf, res);

		outfile = "./Output/dbp_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp_slf, res, res);
		outfile = "./Output/dbp_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp_cf, res, res);
		std::cout << "Results written to disk." << std::endl;

		std::vector<float> difference = absolute_value_of_matrix_difference(dbp_slf, dbp_cf);
		outfile = "./Output/difference_cf_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, difference, res, res);
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// Image execution
// --------------------------------------------------------------------------------------------------------------------------------

void run_benchmark_img(std::vector<float>& phantom_img, unsigned int iterations, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans)
{
	std::vector<duration<double, std::milli>> times;
	std::vector<float> sinogram_img;

	std::cout << "Running benchmark using image2d_t's..." << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
	std::cout << "| Function                        | total [ms] | avg [ms] | stdev [ms] |  N |" << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
	std::cout.precision(2);
	std::cout << std::fixed;

	auto function = CL_compute_sinogram_img;
	times = bench_for_execution_time(function, iterations, device, context, phantom_img, res, rots, scans);
	std::cout << std::right << "| CL_compute_sinogram_img         | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	if (rots * scans != res*res)
		phantom_img = std::vector<float>(rots*scans,0);

	function = CL_fast_ram_lak_filter_img;
	times = bench_for_execution_time(function, iterations, device, context, phantom_img, res, rots, scans);
	std::cout << std::right << "| CL_fast_ram_lak_filter_img      | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	function = CL_fast_shepp_logan_filter_img;
	times = bench_for_execution_time(function, iterations, device, context, phantom_img, res, rots, scans);
	std::cout << std::right << "| CL_fast_shepp_logan_filter_img  | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	function = CL_fast_cosine_filter_img;
	times = bench_for_execution_time(function, iterations, device, context, phantom_img, res, rots, scans);
	std::cout << std::right << "| CL_fast_cosine_filter_img       | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;

	function = CL_discrete_back_projection_img;
	times = bench_for_execution_time(function, iterations, device, context, phantom_img, res, rots, scans);
	std::cout << std::right << "| CL_discrete_back_projection_img | " << std::setw(10) << times[0].count() << " | " << std::setw(8) << times[1].count() << " | " << std::setw(10) << times[2].count() << " | " << std::setw(2) << iterations << " |" << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
}

void produce_images_img(std::vector<float>& phantom_img, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans)
{
	auto function = CL_compute_sinogram_img;
	std::vector<float> sinogram_img = run_function(function, device, context, phantom_img, res, rots, scans);
	std::cout << "Computed sinogram..." << std::endl;

	function = CL_fast_ram_lak_filter_img;
	std::vector<float> rlf_img = run_function(function, device, context, sinogram_img, res, rots, scans);
	std::cout << "Filtered sinogram with fast Ram-Lak filter..." << std::endl;

	function = CL_fast_shepp_logan_filter_img;
	std::vector<float> slf_img = run_function(function, device, context, sinogram_img, res, rots, scans);
	std::cout << "Filtered sinogram with fast Shepp-Logan filter..." << std::endl;

	function = CL_fast_cosine_filter_img;
	std::vector<float> cf_img = run_function(function, device, context, sinogram_img, res, rots, scans);
	std::cout << "Filtered sinogram with fast Cosine filter..." << std::endl;

	function = CL_discrete_back_projection_img; // Attention: Discrete back projection returns buffer, not image2d_t!
	std::vector<float> dbp = run_function(function, device, context, sinogram_img, res, rots, scans);
	std::cout << "Performed back projection of sinogram..." << std::endl;

	std::vector<float> dbp_rlf = run_function(function, device, context, rlf_img, res, rots, scans);
	std::cout << "Performed back projection of Ram-Lak filtered sinogram..." << std::endl;

	std::vector<float> dbp_slf = run_function(function, device, context, slf_img, res, rots, scans);
	std::cout << "Performed back projection of Shepp-Logan filtered sinogram..." << std::endl;

	std::vector<float> dbp_cf = run_function(function, device, context, cf_img, res, rots, scans);
	std::cout << "Performed back projection of Cosine filtered sinogram... " << std::endl;

	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> phantom = reduce_to_buffer(phantom_img);
	std::vector<float> sinogram = reduce_to_buffer(sinogram_img);
	std::vector<float> rlf = reduce_to_buffer(rlf_img);
	std::vector<float> slf = reduce_to_buffer(slf_img);
	std::vector<float> cf = reduce_to_buffer(cf_img);
	#endif
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> phantom = phantom_img;
	std::vector<float> sinogram = sinogram_img;
	std::vector<float> rlf = rlf_img;
	std::vector<float> slf = slf_img;
	std::vector<float> cf = cf_img;
	#endif
	remove_artifacts(dbp_rlf, res);
	remove_artifacts(dbp_slf, res);
	remove_artifacts(dbp_cf, res);
	std::vector<float> err_dbp = absolute_value_of_matrix_difference(phantom, dbp);
	std::vector<float> err_dbp_rlf = absolute_value_of_matrix_difference(phantom, dbp_rlf);
	std::vector<float> err_dbp_slf = absolute_value_of_matrix_difference(phantom, dbp_slf);
	std::vector<float> err_dbp_cf = absolute_value_of_matrix_difference(phantom, dbp_cf);

	std::string outfile = "./Output/sinogram_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm(outfile, sinogram, scans, rots);

	outfile = "./Output/sinogram_rlf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, rlf, scans, rots);
	outfile = "./Output/sinogram_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, slf, scans, rots);
	outfile = "./Output/sinogram_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, cf, scans, rots);

	outfile = "./Output/dbp_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp, res, res);
	outfile = "./Output/dbp_rlf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp_rlf, res, res);
	outfile = "./Output/dbp_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp_slf, res, res);
	outfile = "./Output/dbp_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, dbp_cf, res, res);

	outfile = "./Output/err_dbp_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp, res, res);
	outfile = "./Output/err_dbp_rlf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp_rlf, res, res);
	outfile = "./Output/err_dbp_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp_slf, res, res);
	outfile = "./Output/err_dbp_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+".pgm";
	write_matrix_float_to_pgm_normalised(outfile, err_dbp_cf, res, res);

	std::cout << "Wrote results to disk." << std::endl;
}

void noisy_data_img(std::vector<float>& phantom_img, cl::Device device, cl::Context context, unsigned int res, unsigned int rots, unsigned int scans)
{
	float pi = M_PI;
	float h = (float) 2/res;

	std::vector<float> sinogram_standard_img = CL_compute_sinogram_img(device, context, phantom_img, res, rots, scans);
	std::cout << "Computed sinogram..." << std::endl;

	std::vector<float> salt_and_pepper = salt_and_pepper_noise(res, 0.05);
	#if EXPERIMENTAL_IMAGES==0
	std::vector<float> salt_and_pepper_img = create_image2d_type(salt_and_pepper);
	std::vector<float> sinogram_img = add_matrices(sinogram_standard_img, salt_and_pepper_img);
	#endif
	#if EXPERIMENTAL_IMAGES==1
	std::vector<float> sinogram_img = add_matrices(sinogram_standard_img, salt_and_pepper);
	#endif
	for(int i=0; i<3; i++)
	{
		std::vector<float> slf_img = CL_shepp_logan_filter_img(device, context, sinogram_img, pi/(3*(2*i+1)*h), res, rots, scans);
		std::cout << "Computed Shepp-Logan filter..." << std::endl;
		std::vector<float> cf_img = CL_cosine_filter_img(device, context, sinogram_img, pi/(3*(2*i+1)*h), res, rots, scans);
		std::cout << "Computed Cosine filter..." << std::endl;
		std::vector<float> dbp_slf = CL_discrete_back_projection_img(device, context, slf_img, res, rots, scans);
		std::cout << "Discrete back projection of Shepp-Logan filtered sinogram performed..." << std::endl;
		std::vector<float> dbp_cf = CL_discrete_back_projection_img(device, context, cf_img, res, rots, scans);
		std::cout << "Discrete back projection of Cosine filtered sinogram performed..." << std::endl;

		#if EXPERIMENTAL_IMAGES==0
		std::vector<float> slf = reduce_to_buffer(slf_img);
		std::vector<float> cf = reduce_to_buffer(cf_img);
		#endif
		#if EXPERIMENTAL_IMAGES==1
		std::vector<float> slf = slf_img;
		std::vector<float> cf = cf_img;
		#endif

		std::string outfile = "./Output/sinogram_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, slf, scans, rots);
		outfile = "./Output/sinogram_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, cf, scans, rots);

		remove_artifacts(dbp_slf, res);
		remove_artifacts(dbp_cf, res);

		outfile = "./Output/dbp_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp_slf, res, res);
		outfile = "./Output/dbp_cf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, dbp_cf, res, res);
		std::cout << "Results written to disk." << std::endl;

		std::vector<float> difference = absolute_value_of_matrix_difference(dbp_slf, dbp_cf);
		outfile = "./Output/difference_cf_slf_"+std::to_string(res)+"_"+std::to_string(rots)+"_"+std::to_string(scans)+"_noise_"+std::to_string(i)+".pgm";
		write_matrix_float_to_pgm_normalised(outfile, difference, res, res);
	}
}


// --------------------------------------------------------------------------------------------------------------------------------
// General purpose functions
// --------------------------------------------------------------------------------------------------------------------------------

unsigned int set_resolution(unsigned int m, unsigned int n)
{
	if(m == n)
		return m;
	else
		throw std::runtime_error("Non-square phantoms not supported at this point, aborting");
}

std::vector<float> salt_and_pepper_noise(unsigned int res, float amp) // Expects amplitude between 0 and 1
{
	std::vector<float> image(res*res, 0);

	std::random_device dev;
    std::default_random_engine rng(dev());
    std::uniform_int_distribution<int> dist(-127,127);

	for(int i=0; i<res; i++)
		for(int j=0; j<res; j++)
			image[i*res+j] = amp*dist(rng);

	return image;
}

std::vector<float> add_matrices(std::vector<float>& A, std::vector<float>& B)
{
	if (A.size() == B.size())
	{
		std::vector<float> C(A.size(), 0.0);

		for(int i=0; i<A.size(); i++)
			C[i] = A[i]+B[i];

		return C;
	}
	else
		throw std::runtime_error("Can't add matrices that aren't the same size!");
}

std::vector<float> subtract_matrices(std::vector<float>& A, std::vector<float>& B)
{
	if (A.size() == B.size())
	{
		std::vector<float> C(A.size(), 0.0);

		for(int i=0; i<A.size(); i++)
			C[i] = A[i] - B[i];

		return C;
	}
	else
		throw std::runtime_error("Can't subtract matrices that aren't the same size!");
}

std::vector<float> absolute_value_of_matrix_difference(std::vector<float>& A, std::vector<float>& B)
{
	if (A.size() == B.size())
	{
		std::vector<float> C(A.size(), 0.0);

		for(int i=0; i<A.size(); i++)
			C[i] = fabs(A[i] - B[i]);

		return C;
	}
	else
		throw std::runtime_error("Can't subtract matrices that aren't the same size!");

}

// --------------------------------------------------------------------------------------------------------------------------------
// Input reading and output writing
// --------------------------------------------------------------------------------------------------------------------------------

bool try_reading_float(std::string& input, float& output)
{
	try
	{
		output = std::stof(input);
	}
	catch(std::invalid_argument)
	{
		return false;
	}
	return true;
}

bool try_reading_uint(std::string& input, unsigned int& output)
{
	try
	{
		output = std::stoul(input);
	}
	catch(std::invalid_argument)
	{
		return false;
	}
	return true;
}

void split_string(std::string& input, std::vector<std::string>& output)
{
	std::stringstream ss(input);       // Insert the string into a stream
	std::string buffer;
    while (ss >> buffer)
	{
		output.push_back(buffer);
	}
}

std::vector<float> read_matrix_float(std::string filename, unsigned int& m, unsigned int& n)
{
	std::vector<float> matrix;

	std::vector<std::string> dimensions_string;
	std::vector<unsigned int> dimensions(2);
	std::vector<std::string> line_values_string;
	std::vector<float> line_values;
	float tmp;
	std::string input_buffer;

	std::ifstream data_file(filename);
	if(!data_file.is_open())
		throw std::runtime_error("File couln't be opened!");

	// Read first line and check for compatibility
	std::getline(data_file, input_buffer);
	split_string(input_buffer, dimensions_string);
	if (dimensions_string.size() == 2)
	{
		if(try_reading_uint(dimensions_string[0],dimensions[0]) and try_reading_uint(dimensions_string[1], dimensions[1]))
		{
			m = dimensions[0];
			n = dimensions[1];
			for(int i=0; i<m; i++)
			{
				std::getline(data_file, input_buffer);
				split_string(input_buffer, line_values_string);
				if(line_values_string.size()==n)
				{
					for(int k=0; k<n; k++)
					{
						if (try_reading_float(line_values_string[k], tmp))
							line_values.push_back(tmp);
						else
							throw std::runtime_error("File does not meet requirements!");
					}
					matrix.insert(matrix.end(),line_values.begin(), line_values.end());

					// Empty buffers!

					input_buffer.clear();
					line_values_string.clear();
					line_values.clear();
				}
				else
					throw std::runtime_error("File does not meet requirements!");
			}
			return matrix;
		}
		else
			throw std::runtime_error("File does not meet requirements!");
	}
	else
		throw std::runtime_error("File does not meet requirements!");
}

void print_matrix_float(std::vector<float> matrix, const unsigned int& m, const unsigned int& n)
{
	if (matrix.size() != m*n)
	{
		std::cout << "Matrix size: " << matrix.size() << ", Dimension: " << m << "*" << n << "=" << m*n << std::endl;
		throw std::runtime_error("Corrupt matrix!");
	}
	else
	{
		for(int i=0; i<m; i++)
		{
			for(int j=0; j<n; j++)
			{
				std::cout << std::fixed;
				std::cout.precision(3);
				std::cout << std::setw(10) << matrix[i*n+j];
			}
			std::cout << std::endl;
		}
	}
}

void write_matrix_float_to_file(std::string filename, std::vector<float> matrix, const unsigned int& m, const unsigned int& n)
{
	if (matrix.size() != m*n)
	{
		std::string error_message = "Attempted to write corrupt matrix to file "+filename+", aborting!";
		throw std::runtime_error(error_message);
	}

	else
	{
		std::ofstream output_file(filename);
		if (output_file.is_open())
		{
			for(int i=0; i<m; i++)
			{
				for(int j=0; j<n; j++)
				{
					output_file << std::fixed;
					output_file.precision(3);
					output_file << std::setw(10) << matrix[i*n+j];
				}
				output_file << std::endl;
			}
		}
		else
			throw std::runtime_error("Couldn't create file!");
	}
}

void write_matrix_float_to_pgm(std::string filename, std::vector<float> matrix, const unsigned int& m, const unsigned int& n) // m = rows of matrix, n = columns of matrix
{
	float conversion; // Dummy variable that is converted to unsigned char and clipped to 0..255
	unsigned char output_pixel;

	if (matrix.size() != m*n)
	{
		std::string error_message = "Attempted to write corrupt matrix to "+filename+", aborting!";
		throw std::runtime_error(error_message);
	}

	std::ofstream output_file(filename, std::ios::out | std::ios::binary);
	if (output_file.is_open())
	{
		output_file << "P5" << std::endl;
		output_file << n << " " << m << std::endl; // Switch n and m for usual resolution statement of picture
		output_file << "255" << std::endl; // States maximum grey value

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
			{
				conversion = matrix[i*n+j]+0.5; // For rounding
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
		}
	}
	else
		throw std::runtime_error("Couldn't create file!");
}

void write_matrix_float_to_pgm_normalised(std::string filename, std::vector<float> matrix, const unsigned int& m, const unsigned int& n) // m = rows of matrix, n = columns of matrix
{
	// Expects matrix with float values in the range of 0 and 255. Conversion necessary for matrices with different gray value spectra

	float min,max,conversion,scale; // Dummy variable that is converted to unsigned char and clipped to 0..255
	std::vector<float> output(m*n, 0);
	unsigned char output_pixel;

	if (matrix.size() != m*n)
	{
		std::string error_message = "Attempted to write corrupt matrix to "+filename+", aborting!";
		throw std::runtime_error(error_message);
	}

	min = matrix[0]; // Initialise to value that actually is in the image
	max = matrix[0];

	for(int i=0; i<m; i++)
	{
		for(int j=0; j<n; j++)
		{
			if (matrix[i*n+j] > max)
				max = matrix[i*n+j];
			if (matrix[i*n+j] < min)
				min = matrix[i*n+j];
		}
	}

	scale = 255/(max - min); // Normalise to {0,...,255}

	for(int i=0; i<m; i++)
	{
		for(int j=0; j<m; j++)
		{
			output[i*m+j] = scale*(matrix[i*m+j]-min);
		}
	}

	std::ofstream output_file(filename, std::ios::out | std::ios::binary);
	if (output_file.is_open())
	{
		output_file << "P5" << std::endl;
		output_file << n << " " << m << std::endl; // Switch n and m for usual resolution statement of picture
		output_file << "255" << std::endl; // States maximum grey value

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
			{
				conversion = output[i*n+j] +0.5; // For rounding
				if (conversion > 255)
				{
					output_pixel = (unsigned char) 255;
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
		throw std::runtime_error("Couldn't create file!");
}

std::vector<float> read_pgm_to_matrix_float(std::string filename, unsigned int& m, unsigned int& n)
{
	std::string input_line;
	std::vector<std::string> line_values_string;
	std::vector<float> matrix;
	unsigned char* tmp;
	float dummy;

	std::ifstream input_file(filename, std::ios::binary);
	if (input_file.is_open())
	{
		std::getline(input_file, input_line);
		if(input_line.compare("P5") != 0)
		{
			std::string error_message = "Unknown file format in file "+filename;
			throw std::runtime_error(error_message);
		}
		else
		{
			std::getline(input_file, input_line);
			while(input_line.compare(0,1,"#")==0)
			{
				input_line.clear(); // Line read was comment, discard
				std::getline(input_file, input_line);
			}
			split_string(input_line, line_values_string);
			if (line_values_string.size()==2)
			{
				if(try_reading_uint(line_values_string[0], m) and try_reading_uint(line_values_string[1], n))
				{
					std::getline(input_file, input_line); // Read maximum grey value and discard
					input_line.clear();

					tmp = (unsigned char*) new unsigned char[m*n];

					for(int i=0; i<m*n; i++)
						input_file.read(reinterpret_cast<char*>(tmp), m*n*sizeof(unsigned char));

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
					std::string error_message = "Corrupt file "+filename;
					throw std::runtime_error(error_message);
				}
			}
			else
			{
				std::string error_message = "Corrupt file "+filename;
				throw std::runtime_error(error_message);
			}
		}
	}
	else
	{
		std::string error_message = "Corrupt file "+filename;
		throw std::runtime_error(error_message);
	}

	return matrix;
}

//-------------------------------------------------------------------------------------
// Phantom
//-------------------------------------------------------------------------------------

void raster_ellipse(std::vector<float>& image, int res, std::vector<float> focus, float semimajor, float semiminor, float gray_value)
{
	float h = (float) 2/res;
	float x, y;

	if (focus.size() == 2)
	{
		x = focus[0];
		y = focus[1];
	}
	else
		throw std::runtime_error("Corrupt function call in raster_ellipsis.");

	float semimajor2 = semimajor*semimajor;
	float semiminor2 = semiminor*semiminor;

	float tmp;
	float x_tmp = -1 + (h / 2), y_tmp = 1 - (h / 2);

	for(int i = 1; i < res - 1; i++)
	{
		for(int j = 1; j < res - 1; j++)
		{
			tmp = (float) (x_tmp - x)*(x_tmp - x)/semimajor2 + (y_tmp - y)*(y_tmp - y)/semiminor2;
			if (1 - tmp > 0)
			{
				image[j*res+i] += gray_value;
			}
			y_tmp -= h;
		}
		y_tmp = 1-(h/2);
		x_tmp += h;
	}
}

void raster_ellipse_rotated(std::vector<float>& image, int res, std::vector<float> focus, float a, float b, float gray_value, float angle)
{
	float h = (float) 2/res;
	float x, y;

	if (focus.size() == 2)
	{
		x = focus[0];
		y = focus[1];
	}
	else
		throw std::runtime_error("Corrupt function call in raster_ellipsis.");

	float sinw = sin(angle);
	float cosw = cos(angle);

	float tmp;
	float x_tmp = -1 + (h / 2), y_tmp = 1 - (h / 2);

	for(int i = 1; i < res - 1; i++)
	{
		for(int j = 1; j < res - 1; j++)
		{
			tmp = ((x_tmp-x)*cosw + (y_tmp-y)*sinw)*((x_tmp-x)*cosw + (y_tmp-y)*sinw)/(a*a) + ((x_tmp-x)*sinw - (y_tmp-y)*cosw)*((x_tmp-x)*sinw - (y_tmp-y)*cosw)/(b*b);
			if (1 - tmp > 0)
			{
				image[j*res+i] += gray_value;
			}
			y_tmp -= h;
		}
		y_tmp = 1-(h/2);
		x_tmp += h;
	}
}

std::vector<float> raster_shepp_logan(unsigned int res)
{
	std::vector<float> phantom(res*res, 0);
	std::vector<float> focus(2, 0.0);
	focus[0] = 0;
	focus[1] = 0;
	raster_ellipse(phantom, res, focus, 0.69, 0.92, 255);
	focus[0] = 0;
	focus[1] = -0.0184;
	raster_ellipse(phantom, res, focus, 0.6624, 0.874, -203);
	focus[0] = 0.22;
	focus[1] = 0;
	raster_ellipse_rotated(phantom, res, focus, 0.11, 0.31, -52, -0.31);
	focus[0] = -0.22;
	focus[1] = 0;
	raster_ellipse_rotated(phantom, res, focus, 0.16, 0.41, -52, 0.31);
	focus[0] = 0;
	focus[1] = 0.35;
	raster_ellipse(phantom, res, focus, 0.21, 0.25, 25);
	focus[0] = 0;
	focus[1] = 0.1;
	raster_ellipse(phantom, res, focus, 0.046, 0.046, 25);
	focus[0] = 0;
	focus[1] = -0.1;
	raster_ellipse(phantom, res, focus, 0.046, 0.046, 25);
	focus[0] = -0.08;
	focus[1] = -0.605;
	raster_ellipse(phantom, res, focus, 0.046, 0.023, 25);
	focus[0] = 0;
	focus[1] = -0.605;
	raster_ellipse(phantom, res, focus, 0.023, 0.023, 25);
	focus[0] = 0.06;
	focus[1] = -0.605;
	raster_ellipse(phantom, res, focus, 0.023, 0.046, 25);

	return phantom;
}
