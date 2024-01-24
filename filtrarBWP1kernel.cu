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

__global__ void reduceMin(int *g_iD, int *g_oD){
	unsigned int tid, s;
	__shared__ int sD[256];
	tid = threadIdx.x;
	sD[tid] = g_iD[tid];
	__syncthreads();
	for(s = 1; s < blockDim.x; s *= 2){
		if(tid % (2 * s) == 0){
			int other = sD[tid + s];
			sD[tid] = (sD[tid]<other) ? sD[tid] : other;
		}
		__syncthreads();
	}
	if (tid == 0) g_oD[0] = sD[0];
}

__global__ void reduceMax(int *g_iD, int *g_oD){
	unsigned int tid, s;
	__shared__ int sD[256];
	tid = threadIdx.x;
	sD[tid] = g_iD[tid];
	__syncthreads();
	for(s = 1; s < blockDim.x; s *= 2){
		if(tid % (2 * s) == 0){
			int other = sD[tid + s];
			sD[tid] = (sD[tid]>other) ? sD[tid] : other;
		}
		__syncthreads();
	}
	if (tid == 0) g_oD[0] = sD[0];
}

__global__ void scanCDF(unsigned int N, unsigned int *prob, int *h){
	__shared__ unsigned int prob_priv[256];
	unsigned int tmp;
	int i = threadIdx.x;
	prob_priv[i] = 100000 * h[i] / (N/3);
	__syncthreads();
	for(unsigned int s = 1; s < 256; s *= 2){
		if(i >= s){
			tmp = prob_priv[i] + prob_priv[i-s];
		}
		else{
			tmp = prob_priv[i];
		}
		__syncthreads();
		prob_priv[i] = tmp;
		__syncthreads();
	}
	prob[i] = prob_priv[i];
}

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

__global__ void Equalize(unsigned int N, unsigned char *image, int *min, int *max, unsigned int *prob){
	__shared__ unsigned int PREC;
	int i = (blockIdx.x*blockDim.x + threadIdx.x);
	int stride = blockDim.x * gridDim.x;
	PREC = 10000;
	i *= 3;
	__syncthreads();
	while (i < N){
		unsigned char color = (prob[image[i]] * (max[0] - min[0]) + min[0] * PREC) / PREC;
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
  unsigned int numBytesImage, numBytesHisto, numBytesMinMax, numBytesProb;
  unsigned int nBlocks, nThreads;  
  int *d_histogram, *d_minArray, *d_maxArray;
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
  numBytesProb = 256 * sizeof(int);
  numBytesMinMax = sizeof(int);

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
  cudaMalloc((void**)&d_minArray, numBytesMinMax);
  cudaMalloc((void**)&d_maxArray, numBytesMinMax);
  cudaMalloc((void**)&d_prob, numBytesProb);
  CheckCudaError((char*) "Error de Malloc ", __LINE__);
 
  cudaMemcpy(d_image, image, numBytesImage, cudaMemcpyHostToDevice);
  cudaMemcpy(d_histogram, histogramBW, numBytesHisto, cudaMemcpyHostToDevice);
  CheckCudaError((char*) "Error de Memcpy HtoD", __LINE__);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);
  HistoK<<<nBlocks, nThreads>>>(N, d_image, d_histogram);
  reduceMin<<<1, nThreads>>>(d_histogram, d_minArray);
  reduceMax<<<1, nThreads>>>(d_histogram, d_maxArray);
  scanCDF<<<1, nThreads>>>(N, d_prob, d_histogram);
  Equalize<<<nBlocks, nThreads>>>(N, d_image, d_minArray, d_maxArray, d_prob);
  CheckCudaError((char*) "Error de EqualizationFullK", __LINE__);
  cudaEventRecord(E4, 0);
  cudaEventSynchronize(E4);

  cudaMemcpy(image, d_image, numBytesImage, cudaMemcpyDeviceToHost);
  CheckCudaError((char*) "Error de Memcpy DtoH ", __LINE__);

  cudaFree(d_image);
  cudaFree(d_histogram);
  cudaFree(d_minArray);
  cudaFree(d_maxArray);
  cudaFree(d_prob);
  CheckCudaError((char*) "Error de Free ", __LINE__);
  cudaEventRecord(E6, 0);
  cudaEventSynchronize(E6);

  cudaEventElapsedTime(&tiempoKHistoK, E1, E2);
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
  printf("tiempo Kernels juntos: %4.6f milseg\n", tiempoKernels);
  printf("tiempo Ecualización: %4.6f milseg\n", tiempoProcessing);
  printf("Ancho de banda: %4.2f MB/s\n", 0.000001*((width*height*3)/(tiempoProcessing*0.001)));
}
