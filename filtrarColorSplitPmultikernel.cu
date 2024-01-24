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
int histogramR[256], histogramB[256], histogramG[256];
unsigned int PREC = 10000;

__global__ void HistoR(unsigned int N, unsigned char *image, int *h){
	__shared__ int h_private[256];
	int i = 3*(blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	int j = threadIdx.x;
	h_private[j] = 0;
	__syncthreads();
	while (i < N) {
		unsigned char color = image[i];
		atomicAdd(&h_private[color], 1);
		i = i + stride;
	}
	__syncthreads();
	atomicAdd(&h[j], h_private[j]);
}

__global__ void HistoG(unsigned int N, unsigned char *image, int *h){
	__shared__ int h_private[256];
	int i = 3*(blockIdx.x*blockDim.x + threadIdx.x)+1;
	int stride = blockDim.x * gridDim.x;
	int j = threadIdx.x;
	h_private[j] = 0;
	__syncthreads();
	while (i < N) {
		unsigned char color = image[i];
		atomicAdd(&h_private[color], 1);
		i = i + stride;
	}
	__syncthreads();
	atomicAdd(&h[j], h_private[j]);
}

__global__ void HistoB(unsigned int N, unsigned char *image, int *h){
	__shared__ int h_private[256];
	int i = 3*(blockIdx.x*blockDim.x + threadIdx.x)+2;
	int stride = blockDim.x * gridDim.x;
	int j = threadIdx.x;
	h_private[j] = 0;
	__syncthreads();
	while (i < N) {
		unsigned char color = image[i];
		atomicAdd(&h_private[color], 1);
		i = i + stride;
	}
	__syncthreads();
	atomicAdd(&h[j], h_private[j]);
}

__global__ void EqualizeR(unsigned int N, unsigned char *image, int *minmaxArray, unsigned int *prob){
	__shared__ unsigned int PREC;
	int i = 3*(blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	PREC = 10000;
	while (i < N){
		unsigned char color = (prob[image[i]] * minmaxArray[2] + PREC * minmaxArray[0])/PREC;
		image[i] = color;
		i += stride;
	}
}

__global__ void EqualizeG(unsigned int N, unsigned char *image, int *minmaxArray, unsigned int *prob){
	__shared__ unsigned int PREC;
	int i = 3*(blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	PREC = 10000;
	while (i < N){
		unsigned char color = (prob[image[i+1]] * minmaxArray[5] + PREC * minmaxArray[3])/PREC;
		image[i+1] = color;
		i += stride;
	}
}

__global__ void EqualizeB(unsigned int N, unsigned char *image, int *minmaxArray, unsigned int *prob){
	__shared__ unsigned int PREC;
	int i = 3*(blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	PREC = 10000;
	while (i < N){
		unsigned char color = (prob[image[i+2]] * minmaxArray[8] + PREC * minmaxArray[6])/PREC;
		image[i+2] = color;
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
  int *d_histogramR, *d_histogramG, *d_histogramB, *d_minmaxArray;
  unsigned char *d_image;
  unsigned int *d_probR, *d_probG, *d_probB;
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

  for (int i = 0; i < 256; i++) {
	  histogramR[i] = 0; histogramG[i] = 0; histogramB[i] = 0;
  }
  printf("Filtrando\n");
  //Calculamos histograma:

  N = width * height * 3;
  nThreads = 256;
  nBlocks = (N + nThreads-1)/nThreads;

  numBytesImage = N * sizeof(unsigned char);
  numBytesHisto = 256 * sizeof(int);
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
  cudaMalloc((void**)&d_histogramR, numBytesHisto);
  cudaMalloc((void**)&d_histogramG, numBytesHisto);
  cudaMalloc((void**)&d_histogramB, numBytesHisto);
  cudaMalloc((void**)&d_minmaxArray, numBytesMinMax);
  cudaMalloc((void**)&d_probR, numBytesHisto);
  cudaMalloc((void**)&d_probG, numBytesHisto);
  cudaMalloc((void**)&d_probB, numBytesHisto);
  CheckCudaError((char*) "Error de Malloc ", __LINE__);
 
  cudaMemcpy(d_image, image, numBytesImage, cudaMemcpyHostToDevice);
  cudaMemcpy(d_histogramR, histogramR, numBytesHisto, cudaMemcpyHostToDevice);
  cudaMemcpy(d_histogramG, histogramG, numBytesHisto, cudaMemcpyHostToDevice);
  cudaMemcpy(d_histogramB, histogramB, numBytesHisto, cudaMemcpyHostToDevice);
  CheckCudaError((char*) "Error de Memcpy HtoD", __LINE__);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);
  HistoR<<<nBlocks, nThreads>>>(N, d_image, d_histogramR);
  HistoG<<<nBlocks, nThreads>>>(N, d_image, d_histogramG);
  HistoB<<<nBlocks, nThreads>>>(N, d_image, d_histogramB);
  CheckCudaError((char*) "Error de HistoK", __LINE__);
  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);
  
  cudaMemcpy(histogramR, d_histogramR, numBytesHisto, cudaMemcpyDeviceToHost);
  cudaMemcpy(histogramG, d_histogramG, numBytesHisto, cudaMemcpyDeviceToHost);
  cudaMemcpy(histogramB, d_histogramB, numBytesHisto, cudaMemcpyDeviceToHost);
  CheckCudaError((char*) "Error de Memcpy DtoH", __LINE__); 

  int minR = minHisto(histogramR, 0, 255), minG = minHisto(histogramG, 0, 255), minB = minHisto(histogramB, 0, 255);
  int maxR = maxHisto(histogramR, 0, 255), maxG = maxHisto(histogramG, 0, 255), maxB = maxHisto(histogramB, 0, 255);
  int maxminR = maxR - minR, maxminG = maxG - minG, maxminB = maxB - minB;
  unsigned int probR[256], probG[256], probB[256];
  int length = 256;
  probR[0] = PREC * histogramR[0]/(width*height);
  probG[0] = PREC * histogramG[0]/(width*height);
  probB[0] = PREC * histogramB[0]/(width*height);
  for(int i = 1; i < length; ++i){
	probR[i] = probR[i-1] + PREC * histogramR[i]/(width*height);
	probG[i] = probG[i-1] + PREC * histogramG[i]/(width*height);
	probB[i] = probB[i-1] + PREC * histogramB[i]/(width*height);
  }
  int auxMinmaxArray[9] = {minR, maxR, maxminR, minG, maxG, maxminG, minB, maxB, maxminB};

  cudaMemcpy(d_minmaxArray, auxMinmaxArray, numBytesMinMax, cudaMemcpyHostToDevice);
  cudaMemcpy(d_probR, probR, numBytesHisto, cudaMemcpyHostToDevice);
  cudaMemcpy(d_probG, probG, numBytesHisto, cudaMemcpyHostToDevice);
  cudaMemcpy(d_probB, probB, numBytesHisto, cudaMemcpyHostToDevice);
  CheckCudaError((char*) "Error de Memcpy HtoD", __LINE__);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);
  EqualizeR<<<nBlocks, nThreads>>>(N, d_image, d_minmaxArray, d_probR);
  EqualizeG<<<nBlocks, nThreads>>>(N, d_image, d_minmaxArray, d_probG);
  EqualizeB<<<nBlocks, nThreads>>>(N, d_image, d_minmaxArray, d_probB);
  CheckCudaError((char*) "Error de Equalize", __LINE__);
  cudaEventRecord(E4, 0);
  cudaEventSynchronize(E4);

  cudaMemcpy(image, d_image, numBytesImage, cudaMemcpyDeviceToHost);
  CheckCudaError((char*) "Error de Memcpy DtoH ", __LINE__);


  cudaFree(d_image);
  cudaFree(d_histogramR);
  cudaFree(d_histogramG);
  cudaFree(d_histogramB);
  cudaFree(d_minmaxArray);
  cudaFree(d_probR);
  cudaFree(d_probG);
  cudaFree(d_probB);
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

