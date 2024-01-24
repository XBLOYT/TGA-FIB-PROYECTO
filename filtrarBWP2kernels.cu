#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/resource.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

char *fileIN, *fileOUT;
unsigned char *image;
int width, height, pixelWidth; //meta info de la imagen
int histogramBW[256];
unsigned int PREC = 10000;

__global__ void HistoK(unsigned int N, unsigned char *image, int *h){
	__shared__ int h_private[256];
	int i = 3 * (blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	h_private[threadIdx.x] = 0;
	__syncthreads();
	while (i < N) {
		int colorR = image[i]*2126;
		int colorG = image[i+1]*7152;
		int colorB = image[i+2]*722;
		unsigned char color = (colorR+colorG+colorB)/10000;
		image[i]=color;
		image[i+1]=color;
		image[i+2]=color;	
		atomicAdd(&h_private[color], 1);
		i = i + stride;
	}
	__syncthreads();
	i = threadIdx.x;
	atomicAdd(&h[i], h_private[i]);
}

__global__ void Equalize(unsigned int N, unsigned char *image, int *minmaxArray, unsigned int *prob){
	__shared__ unsigned int PREC;
	int i = (blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	if(i == 0){
		PREC = 10000;
	}
	i *= 3;
	while (i < N){
		unsigned char color = (prob[image[i]] * minmaxArray[2] + minmaxArray[0] * PREC) / PREC;
		image[i] = color;
		image[i+1] = color;
		image[i+2] = color;
		i = i + stride;
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

int minHisto(int i_histogram[]){
	int length = 256;
	for(int i = 0; i < length; i++){
		if(i_histogram[i] != 0) return i;
	}
	return -1;
}

int maxHisto(int i_histogram[]){
	int length = 256;
	for(int i = length - 1; i >= 0; i--){
		if(i_histogram[i] != 0) return i;
	}
	return -1;
}

float GetTime(void){
	struct timeval tim;
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	tim = ru.ru_utime;
	return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
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
  else { printf("Usage: ./exe fileIN fileOUT (equalize) (mode (d|g))\n"); exit(0); }

  t1 = GetTime();
  printf("Reading image...\n");
  image = stbi_load(fileIN, &width, &height, &pixelWidth, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
     return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

  for (int i = 0; i < 256; i++) histogramBW[i] = 0;

  //Calculo tamanos threads / bloques
  N = width * height * 3;
  nThreads = 256;
  nBlocks = (N + nThreads-1)/nThreads;

  numBytesImage = N * sizeof(unsigned char);
  numBytesHisto = 256 * sizeof(int);
  numBytesMinMax = 3 * sizeof(int);

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
  cudaMemcpy(d_histogram, histogramBW, numBytesHisto, cudaMemcpyHostToDevice);
  CheckCudaError((char*) "Error de Memcpy HtoD", __LINE__);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);
  HistoK<<<nBlocks, nThreads>>>(N, d_image, d_histogram);
  CheckCudaError((char*) "Error de HistoK", __LINE__);
  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  cudaMemcpy(histogramBW, d_histogram, numBytesHisto, cudaMemcpyDeviceToHost);
  CheckCudaError((char*) "Error de Memcpy DtoH", __LINE__);

  int min = minHisto(histogramBW);
  int max = maxHisto(histogramBW);
  int maxmin = max - min;
  unsigned int prob[256];
  int length = 256;
  prob[0] = PREC * histogramBW[0] / (width*height);
  for(int i = 1; i < length; ++i){
  	prob[i] = prob[i-1] + PREC * histogramBW[i]/(width*height);
  }
  int auxMinmaxArray[3] = {min, max, maxmin}; 

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

  printf("---FILTRAR BWP---\n");
  printf("tiempo Global: %4.6f milseg\n", t2-t1);
  printf("tiempo HtoD: %4.6f milseg\n", tiempoHtoD);
  printf("tiempo DtoH: %4.6f milseg\n", tiempoDtoH);
  printf("tiempo Kernel HistoK: %4.6f milseg\n", tiempoKHistoK);
  printf("tiempo Kernel Equalize: %4.6f milseg\n", tiempoKEqualize);
  printf("tiempo Kernels juntos: %4.6f milseg\n", tiempoKernels);
  printf("tiempo Ecualización: %4.6f milseg\n", tiempoProcessing);
  printf("Ancho de banda: %4.2f MB/s\n", 0.000001*((width*height*3)/(tiempoProcessing*0.001)));

}

