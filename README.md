# README

En este proyecto hay diversos archivos. A continuación se explica cómo usarlos y para qué sirven.


# Archivos

## Informe.pdf y tablas.pdf

Informe del proyecto y resultados de los tests en tablas.

## filtrarBW.cu, filtrarBWP1kernel.cu, filtrarBWP2kernels.cu

Versiones secuencial (filtrarBW.cu) y paralelas (filtrarBWP1Kernel.cu, filtrarBWP2Kernels.cu) del método de ecualización en blanco y negro. Para compilarlos y ejecutarlos, las instrucciones a seguir son las siguientes:

 - `make filtrarBW`, `make filtrarBWP1kernel`, `make filtrarBWP2kernels`. Estas instrucciones compilarán las distintas versiones. 
 >IMPORTANTE: `make filtrarBW` da como resultado un filtrarBW.exe, pero, tanto `make filtrarBWP1kernel` como `make filtrarBWP2kernels` dan como resultado un ejecutable llamado filtrarBWP.exe, es decir, que no se puede tener las dos versiones paralelas compiladas a la vez.
 - `sbatch jobBW.sh` Esta directriz mandará los comandos del job.sh al boada para ejecutar los programas. Por defecto, el jobBW.sh tiene 5 instrucciones para ejecutar la versión secuencial y otras 5 para ejecutar filtrarBWP.exe (sin importar de qué archivo .cu provenga el ejecutable). Se pueden comentar líneas si se desea probar solo 1 variante.
 
 Las versiones paralelas hacen lo siguiente:
 
 - filtrarBWP1kernel.cu: Hace la ecualización en 1 solo Kernel, tanto la creación del histograma, como el cálculo de la distribución acumulada como la ecualización del histograma.
 -  filtrarBWP2kernels.cu: Hace la ecualización en 2 kernels, uno se dedica a la creación del histograma y el otro a la ecualización del histograma. El cálculo de la distribución de frecuencias acumuladas se realiza secuencialmente.

## filtrarColorSplit.cu, filtrarColorSplitPthreadbypixel.cu, filtrarColorSplitPthreadbychannel.cu, filtrarColorSplitPmultikernel.cu
 
Versiones secuencial (filtrarColorSplit.cu) y paralelas (filtrarColorSplitPthreadbypixel.cu, filtrarColorSplitPthreadbychannel.cu, filtrarColorSplitPmultithreads.cu) de la ecualización de los canales por separado. Para compilarlos y ejecutarlos, las instrucciones a seguir son las siguientes:
 - `make filtrarColorSplit`, `make filtrarColorSplitPthreadbypixel`, `make filtrarColorSplitPthreadbychannel` o `filtrarColorSplitPmultikernel`. Estas instrucciones compilarán las distintas versiones. I
 >MPORTANTE: `make filtrarColorSplit` da como resultado un filtrarColorSplit.exe, pero el resto dan como resultado un ejecutable llamado filtrarColorSplitP.exe, es decir, que no se puede tener las 3 versiones paralelas compiladas a la vez.
 - `sbatch jobColorSplit.sh` Esta directriz mandará los comandos del job.sh al boada para ejecutar los programas. Por defecto, el jobColorSplit.sh tiene 5 instrucciones para ejecutar la versión secuencial y otras 5 para ejecutar filtrarColorSplitP.exe (sin importar de qué archivo .cu provenga el ejecutable). Se pueden comentar líneas si se desea probar solo 1 variante.
 
 Las versiones paralelas hacen lo siguiente:
 
 - filtrarColorSplitPthreadbypixel.cu: En esta versión, cada thread se encarga de un píxel.
 -  filtrarColorSplitPthreadbychannel.cu: En esta otra, los threads se encargan de los canales individuales. Es decir, hay 3 threads por cada píxel, cada uno encargándose de cada canal.
 - filtrarColorSplitPmultikernel: Finalmente, en esta versión lanzamos 3 kernels. Cada kernel se encarga de ecualizar un canal.
 
## filtrarBITS.cu, filtrarBITSP.cu

Versiones secuencial (filtrarBITS.cu) y paralela (filtrarBITSP.cu) de la ecualización juntando los bits altos de los 3 canales en un valor y ecualizando en base a ese grupo de valores. Para compilarlos y ejecutarlos, las instrucciones a seguir son las siguientes:
 - `make filtrarBITS`, `make filtrarBITSP`. Estas instrucciones compilarán las distintas versiones.
 - `sbatch jobBITS.sh` Esta directriz mandará los comandos del job.sh al boada para ejecutar los programas. Por defecto, el jobBITS.sh tiene 5 instrucciones para ejecutar la versión secuencial y otras 5 para ejecutar la paralela. Se pueden comentar líneas si se desea probar solo 1 variante.

## Makefile

A parte de las directrices del Makefile anteriormente mencionadas, hay unas pocas más que vale la pena mencionar:

 - `make clean`: Borra todos los archivos .o, .exe, Out*.jpg y los submit.
 - `make`: Compila las versiones paralelas de los 3 métodos de ecualización (para ColorSplit elige la versión ColorSplitPthreadbypixel y para BW elige la versión de 2 kernels).
 - `make seq`: Compila las versiones secuenciales de los programas.
 - `make all`: Compila tanto las versiones secuenciales como las paralelas (eligiendo las anteriormente mencionadas) de los programas.

## jobBW.sh, jobBITS.sh, jobColorSplit.sh

Hemos dividido job.sh en estos 3 archivos para hacer más cómodo el testeo de los distintos programas. En los jobs se encuentran los comandos para ejecutar los .exe. Todos tienen esta forma:

    ./[programa].exe [NOMBRE_ARCHIVO_IN].jpg [NOMBRE_ARCHIVO_OUT].jpg


# About

Este proyecto pertenece a la asignatura de Targetas Gráficas y Aceleradores (TGA) y ha sido realizado por Albert Ruiz Vives y Xavier Bernat López.