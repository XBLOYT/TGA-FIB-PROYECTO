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
int histogram[4096];
unsigned int PREC = 10000;

float GetTime(void){
	struct timeval tim;
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	tim = ru.ru_utime;
	return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}

int minHisto(int i_histogram[]){
	int length = 4096;
	for(int i = 0; i < length; i++){
		if(i_histogram[i] != 0) return i;
	}
	return -1;
}

int maxHisto(int i_histogram[]){
	int length = 4096;
	for(int i = length - 1; i >= 0; i--){
		if(i_histogram[i] != 0) return i;
	}
	return -1;
}

void equalizeGoodColor(){	
	int min = minHisto(histogram);
	int max = maxHisto(histogram);
	int maxmin = max - min;
	unsigned int prob[4096];
	int length = 4096;
	prob[0] = PREC * histogram[0]/(width*height);
	for(int i = 1; i < length; ++i){
		prob[i] = prob[i-1] + PREC * histogram[i]/(width*height);
	}
	for(int i = 0; i < width * height * 3; i=i+3){
		unsigned char colorR = image[i] >> 4;
		unsigned char colorG = image[i+1] >> 4;
		unsigned char colorB = image[i+2] >> 4;
		unsigned int index = (colorR << 8) | (colorG << 4) | (colorB); 
		unsigned int color = (prob[index] * maxmin + PREC * min)/PREC;

		unsigned char colorRH = color >> 8;
		unsigned char colorGH = (color >> 4) & 0b000000001111;
		unsigned char colorBH = (color) & 0b000000001111 ;
		
		unsigned char colorRFinal = (colorRH << 4) | (image[i] & 0b00001111);	
		unsigned char colorGFinal = (colorGH << 4) | (image[i+1] & 0b00001111);	
		unsigned char colorBFinal = (colorBH << 4) | (image[i+2] & 0b00001111);	

		image[i] = colorRFinal;
		image[i+1] = colorGFinal;
		image[i+2] = colorBFinal;
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

  for (int i = 0; i < 4096; i++) {
	  histogram[i] = 0;
  }
  printf("Filtrando\n");
  t3 = GetTime();
  //Calculamos la fusión de los colores:
  for(int i=0;i<width*height*3;i=i+3){
    unsigned char colorR = image[i];
    unsigned char colorG = image[i+1];
    unsigned char colorB = image[i+2];
    colorR = colorR >> 4;
    colorG = colorG >> 4;
    colorB = colorB >> 4;
    unsigned int index = (colorR << 8) | (colorG << 4) | (colorB); 
    histogram[index]++;
  }
  if (eq) equalizeGoodColor();
  t4 = GetTime();
  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,image,0);
  t2 = GetTime();
  printf("---FILTRAR BITS---\n");
  printf("Análisis temporal de la ejecución:\n");
  printf("Tiempo secuencial Global: %4.6f miliseg\n",t2-t1);
  printf("Tiempo secuencial Ecualización: %4.6f miliseg\n", t4-t3);
  printf("Ancho de banda: %4.2f MB/s\n", 0.000001*((width*height*3)/((t4-t3)*0.001)));
}

