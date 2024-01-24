#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/resource.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

char *fileIN, *fileOUT;
unsigned char *image, *equalizedImage;
int width, height, pixelWidth; //meta info de la imagen
int histogram[768];
unsigned int PREC = 10000;

__global__ void HistoK(unsigned int N, unsigned char *image, int *h){
	__shared__ int h_private[768];
	int i = (blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	int j = threadIdx.x;
	int k = j;
	while(j < 768){
		h_private[j] = 0;
		j += 256;
	}
	__syncthreads();
	while (i < N) {
		unsigned char color = image[i];
		atomicAdd(&h_private[color + 256*(i%3)], 1);
		i = i + stride;
	}
	__syncthreads();
	while(k < 768){
		atomicAdd(&h[k], h_private[k]);
		k += 256;
	}
}

__global__ void Equalize(unsigned int N, unsigned char *image, int *minmaxArray, unsigned int *prob){
	__shared__ unsigned int PREC;
	int i = (blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	if(i == 0){
		PREC = 10000;
	}
	while (i < N){
		unsigned char color = (prob[image[i] + 256*(i%3)] * minmaxArray[2 + 3*(i%3)] + PREC * minmaxArray[0 + 3*(i%3)])/PREC;
		image[i] = color;
		i += stride;
	}
}

void CheckCudaError(char sms[], int line){
	cudaError_t error;

	error = cudaGetLastError();
	if(error){
		printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
		exit(EXIT_FAILURE);
	}
}

float GetTime(void){
	struct timeval tim;
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	tim = ru.ru_utime;
	return ((double) tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}

int minHisto(int i_histogram[], int l, int r){
	for(int i = l; i <= r; i++){
		if(i_histogram[i] != 0) return i;
	}
	return -1;
}

int maxHisto(int i_histogram[], int l, int r){
	for(int i = r; i >= l; i--){
		if(i_histogram[i] != 0) return i;
	}
	return -1;
}

int main(int argc, char** argv)
{
  unsigned int N;
  unsigned int numBytesImage, numBytesHisto, numBytesMinMax;
  unsigned int nBlocks, nThreads;
  int *d_histogram, *d_minmaxArray;
  unsigned char *d_image;
  unsigned int *d_prob;
  float t1, t2, tiempoKHistoK, tiempoKEqualize, tiempoProcessing, tiempoHtoD, tiempoDtoH, tiempoKernels;

  cudaEvent_t E1, E2, E3, E4, E5, E6;  

  // Ficheros de entrada y de salida 
  if (argc == 3) { fileIN = argv[1]; fileOUT = argv[2]; }
  else { printf("Usage: ./exe fileIN fileOUT (equalize)\n"); exit(0); }

  t1 = GetTime();
  printf("Reading image...\n");
  image = stbi_load(fileIN, &width, &height, &pixelWidth, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
     return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

  for (int i = 0; i < 768; i++) {
	  histogram[i] = 0;
  }
  printf("Filtrando\n");
  //Calculamos histograma:

  N = width * height * 3;
  nThreads = 256;
  nBlocks = (N + nThreads-1)/nThreads;

  numBytesImage = N * sizeof(unsigned char);
  numBytesHisto = 768 * sizeof(int);
  numBytesMinMax = 9 * sizeof(int);

  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);
  cudaEventCreate(&E4);
  cudaEventCreate(&E5);
  cudaEventCreate(&E6);

  cudaEventRecord(E5, 0);
  cudaEventSynchronize(E5);
  cudaMalloc((void**)&d_image, numBytesImage);
  cudaMalloc((void**)&d_histogram, numBytesHisto);
  cudaMalloc((void**)&d_minmaxArray, numBytesMinMax);
  cudaMalloc((void**)&d_prob, numBytesHisto);
  CheckCudaError((char*) "Error de Malloc ", __LINE__);
 
  cudaMemcpy(d_image, image, numBytesImage, cudaMemcpyHostToDevice);
  cudaMemcpy(d_histogram, histogram, numBytesHisto, cudaMemcpyHostToDevice);
  CheckCudaError((char*) "Error de Memcpy HtoD", __LINE__);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);
  HistoK<<<nBlocks, nThreads>>>(N, d_image, d_histogram);
  CheckCudaError((char*) "Error de HistoK", __LINE__);
  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);
  
  cudaMemcpy(histogram, d_histogram, numBytesHisto, cudaMemcpyDeviceToHost);
  CheckCudaError((char*) "Error de Memcpy DtoH", __LINE__);

  int minR = minHisto(histogram, 0, 255), minG = minHisto(histogram, 256, 511), minB = minHisto(histogram, 512, 767);
  int maxR = maxHisto(histogram, 0, 255), maxG = maxHisto(histogram, 256, 511), maxB = maxHisto(histogram, 512, 767);
  int maxminR = maxR - minR, maxminG = maxG - minG, maxminB = maxB - minB;
  unsigned int prob[768];
  int length = 256;
  prob[0] = PREC * histogram[0]/(width*height);
  prob[256] = PREC * histogram[256]/(width*height);
  prob[512] = PREC * histogram[512]/(width*height);
  for(int i = 1; i < length; ++i){
	prob[i] = prob[i-1] + PREC * histogram[i]/(width*height);
	prob[i + 256] = prob[i + 256 -1] + PREC * histogram[i + 256]/(width*height);
	prob[i + 512] = prob[i + 512 -1] + PREC * histogram[i + 512]/(width*height);
  }
  int auxMinmaxArray[9] = {minR, maxR, maxminR, minG, maxG, maxminG, minB, maxB, maxminB};

  cudaMemcpy(d_minmaxArray, auxMinmaxArray, numBytesMinMax, cudaMemcpyHostToDevice);
  cudaMemcpy(d_prob, prob, numBytesHisto, cudaMemcpyHostToDevice);
  CheckCudaError((char*) "Error de Memcpy HtoD", __LINE__);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);
  Equalize<<<nBlocks, nThreads>>>(N, d_image, d_minmaxArray, d_prob);
  CheckCudaError((char*) "Error de Equalize", __LINE__);
  cudaEventRecord(E4, 0);
  cudaEventSynchronize(E4);

  cudaMemcpy(image, d_image, numBytesImage, cudaMemcpyDeviceToHost);
  CheckCudaError((char*) "Error de Memcpy DtoH ", __LINE__);

  cudaFree(d_image);
  cudaFree(d_histogram);
  cudaFree(d_minmaxArray);
  cudaFree(d_prob);
  CheckCudaError((char*) "Error de Free ", __LINE__);
  cudaEventRecord(E6, 0);
  cudaEventSynchronize(E6);

  cudaEventElapsedTime(&tiempoKHistoK, E1, E2);
  cudaEventElapsedTime(&tiempoKEqualize, E3, E4);
  cudaEventElapsedTime(&tiempoProcessing, E5, E6);
  cudaEventElapsedTime(&tiempoHtoD, E5, E1);
  cudaEventElapsedTime(&tiempoDtoH, E4, E6);
  cudaEventElapsedTime(&tiempoKernels, E1, E4);

  cudaEventDestroy(E1);
  cudaEventDestroy(E2);
  cudaEventDestroy(E3);
  cudaEventDestroy(E4);
  cudaEventDestroy(E5);
  cudaEventDestroy(E6);

  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,image,0);
  t2 = GetTime();
  printf("---FILTRAR ColorSplitP---\n");
  printf("tiempo Global: %4.6f milseg\n", t2-t1);
  printf("tiempo HtoD: %4.6f milseg\n", tiempoHtoD);
  printf("tiempo DtoH: %4.6f milseg\n", tiempoDtoH);
  printf("tiempo Kernel HistoK: %4.6f milseg\n", tiempoKHistoK);
  printf("tiempo Kernel Equalize: %4.6f milseg\n", tiempoKEqualize);
  printf("tiempo Kernels juntos: %4.6f milseg\n", tiempoKernels);
  printf("tiempo Ecualización: %4.6f milseg\n", tiempoProcessing);
  printf("Ancho de banda: %4.2f MB/s\n", 0.000001*((width*height*3)/(tiempoProcessing*0.001)));
}

