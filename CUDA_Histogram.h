//This program is written by Abubakr Shafique (abubakr.shafique@gmail.com) 
#ifndef _CUDA_Histogram_
#define _CUDA_Histogram_

void Histogram_Calculation_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red);

#endif
