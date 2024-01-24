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
int histogramBW[256];
unsigned int PREC = 10000;

float GetTime(void){
	struct timeval tim;
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	tim = ru.ru_utime;
	return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
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

void equalizeGoodBW(){	
	int min = minHisto(histogramBW);
	int max = maxHisto(histogramBW);
	int maxmin = max - min;
	unsigned int prob[256];
	int length = 256;
	prob[0] = PREC * histogramBW[0] / (width*height);
	//Calculamos los valores de las probabilidades acumuladas
	for(int i = 1; i < length; ++i){
		prob[i] = prob[i-1] + PREC * histogramBW[i]/(width*height);
	}
	//Ecualizamos
	for(int i = 0; i < width * height * 3; i=i+3){
		unsigned char color = (prob[image[i]] * maxmin + min * PREC) / PREC;
		image[i] = color;
		image[i+1] = color;
		image[i+2] = color;
	}

}

int main(int argc, char** argv)
{
  bool eq = true;
  unsigned char mode = 'g';
  float t1, t2, t3, t4;
  // Ficheros de entrada y de salida 
  if (argc == 3) { fileIN = argv[1]; fileOUT = argv[2]; }
  else if (argc == 4) {fileIN = argv[1]; fileOUT = argv[2]; eq = argv[3];}
  else if (argc == 5) {
	fileIN = argv[1];
	fileOUT = argv[2];
	mode = argv[4][0];
  }
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
  printf("Filtrando\n");
  t3 = GetTime();
  //Calculamos histograma y pasamos imagen a blanco y negro:
  for(int i=0;i<width*height*3;i=i+3){
    int colorR = image[i]*2126;
    int colorG = image[i+1]*7152;
    int colorB = image[i+2]*722;
    unsigned char color = (colorR+colorG+colorB)/10000;
    image[i]=color;
    image[i+1]=color;
    image[i+2]=color;	
    histogramBW[color]++;
  }
  if (eq) equalizeGoodBW();
  t4 = GetTime();
  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,image,0);
  t2 = GetTime();
  printf("---FILTRAR BW---\n");
  printf("Análisis temporal de la ejecución:\n");
  printf("Tiempo secuencial Global: %4.6f milseg\n", t2-t1);
  printf("Tiempo secuencial Ecualización: %4.6f milseg\n", t4-t3);
  printf("Ancho de banda: %4.2f MB/s\n", 0.000001*((width*height*3)/((t4-t3)*0.001)));

}

