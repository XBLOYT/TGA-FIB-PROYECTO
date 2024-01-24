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
int histogramR[256];
int histogramG[256];
int histogramB[256];
unsigned int PREC = 10000;

float GetTime(void){
	struct timeval tim;
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	tim = ru.ru_utime;
	return ((double) tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
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

void equalizeGoodColor(){	
	int minR = minHisto(histogramR), minG = minHisto(histogramG), minB = minHisto(histogramB);
	int maxR = maxHisto(histogramR), maxG = maxHisto(histogramG), maxB = maxHisto(histogramB);
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
	for(int i = 0; i < width * height * 3; i=i+3){
		unsigned char colorR = (probR[image[i]] * maxminR + PREC * minR)/PREC;
		unsigned char colorG = (probG[image[i+1]] * maxminG + PREC * minG)/PREC;
		unsigned char colorB = (probB[image[i+2]] * maxminB + PREC * minB)/PREC;
		image[i] = colorR;
		image[i+1] = colorG;
		image[i+2] = colorB;
	}

}

int main(int argc, char** argv)
{
  bool eq = true;
  float t1, t2, t3, t4;
  // Ficheros de entrada y de salida 
  if (argc == 3) { fileIN = argv[1]; fileOUT = argv[2]; }
  else if (argc == 4) {fileIN = argv[1]; fileOUT = argv[2]; eq = argv[3];}
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
	  histogramR[i] = 0;
	  histogramG[i] = 0;
	  histogramB[i] = 0;
  }
  printf("Filtrando\n");
  //Calculamos histograma:
  t3 = GetTime();
  for(int i=0;i<width*height*3;i=i+3){
    histogramR[image[i]]++;
    histogramG[image[i+1]]++;
    histogramB[image[i+2]]++;
  }
  if (eq) equalizeGoodColor();
  t4 = GetTime();
  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,image,0);
  t2 = GetTime();
  printf("---FILTRAR COLORSPLIT---\n");
  printf("Tiempo Secuencial Global: %4.6f miliseg\n", t2-t1);
  printf("Tiempo Secuencial Ecualización: %4.6f miliseg\n", t4-t3);
  printf("Ancho de banda: %4.2f NB/s\n", 0.000001*((width*height*3)/((t4-t3)*0.001)));
}

