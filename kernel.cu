//This program is written by Abubakr Shafique (abubakr.shafique@gmail.com) 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Histogram.h"

__global__ void Histogram_CUDA(unsigned char* Image, int Channels, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red);

void Histogram_Calculation_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red){
	unsigned char* Dev_Image = NULL;
	int* Dev_Histogram_Blue = NULL;
	int* Dev_Histogram_Green = NULL;
	int* Dev_Histogram_Red = NULL;

	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram_Blue, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Green, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Red, 256 * sizeof(int));

	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Blue, Histogram_Blue, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Green, Histogram_Green, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Red, Histogram_Red, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Histogram_CUDA << <Grid_Image, 1 >> >(Dev_Image, Channels, Dev_Histogram_Blue, Dev_Histogram_Green, Dev_Histogram_Red);

	//copy memory back to CPU from GPU
	cudaMemcpy(Histogram_Blue, Dev_Histogram_Blue, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Histogram_Green, Dev_Histogram_Green, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Histogram_Red, Dev_Histogram_Red, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	
	//free up the memory of GPU
	cudaFree(Dev_Histogram_Blue);
	cudaFree(Dev_Histogram_Green);
	cudaFree(Dev_Histogram_Red);
	cudaFree(Dev_Image);
}

__global__ void Histogram_CUDA(unsigned char* Image, int Channels, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red){
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = (x + y * gridDim.x) * Channels;

	atomicAdd(&Histogram_Blue[Image[Image_Idx]], 1);
	atomicAdd(&Histogram_Green[Image[Image_Idx + 1]], 1);
	atomicAdd(&Histogram_Red[Image[Image_Idx + 2]], 1);
}
