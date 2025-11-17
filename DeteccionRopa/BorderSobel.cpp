//Se carga una imagen de disco y se muestra por pantalla


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // Necesario para applyColorMap
#include <opencv2/opencv.hpp>           // Necesario para calcHist
#include <iostream>

#include "BorderSobel.h"
#include <map>

using namespace std;
using namespace cv;

#define IMAGEN "C:\\AAAA\\OpenCV\\images\\"
//#define IMAGEN "C:\Users\Viktor\Dropbox\Universidad\Procesamiento de Imágenes por Computador\imagenes\mandril.jpg"

Mat imagen;
Mat imageGray;
Mat imageEdited;

string imageName = " ";
char* imageChar;

void ChargeImage()
{
    cout << "Enter image name (.jpg format)" << endl;
    cin >> imageName;
    // Nombre de la imagen que se va a cargar
    string path = IMAGEN + imageName + ".jpg";

    //{ para cargar la ruta de la imagen (necesitamos char* en vez de string)
    vector<char> imageChar(path.begin(), path.end());
    imageChar.push_back('\0');
    char* cstr = imageChar.data();
    //}

    // Se carga la imagen desde disco y se comprueba que lo ha hecho correctamente
    //imagen = imread(path);

    // Se carga la imagen desde disco en escala de grises
    imageGray = imread(path, IMREAD_GRAYSCALE);

    imagen = imread(path, IMREAD_COLOR);

    if (!imagen.data) {
        cout << "Error al cargar la imagen: " << path << endl;
        exit(1);
    }

    // namedWindow("Original", WINDOW_AUTOSIZE);
    // imshow("Original", imagen);
    // waitKey(0);


    imageEdited = imagen.clone(); // Copia de la imagen original para editarla
	BorderSobel();

    namedWindow("Resultado", WINDOW_AUTOSIZE);
    imshow("Resultado", imageEdited);
    waitKey(0);
}

void BorderSobel()
{
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    // Gradiente en la dirección X
    Sobel(imageGray, grad_x, CV_16S, 1, 0, 3);
    convertScaleAbs(grad_x, abs_grad_x);
    // Gradiente en la dirección Y
    Sobel(imageGray, grad_y, CV_16S, 0, 1, 3);
    convertScaleAbs(grad_y, abs_grad_y);
    // Combina ambos gradientes
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imageEdited);
}
