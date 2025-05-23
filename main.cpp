#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/red-eyes.h"
using namespace std;
using namespace cv;

int main() {

    //Mat pic1 = imread("C:\\CTI\\An 3 sem 2\\PI\\Proiect\\Project\\images\\pic1.png",
                      //  IMREAD_COLOR);
    Mat pic1 = imread("C:\\CTI\\An 3 sem 2\\PI\\Proiect\\Project\\tests\\teste\\pic3.jpg", IMREAD_COLOR);

    imshow("First Original Image", pic1);
    Mat pic2 = imread("C:\\CTI\\An 3 sem 2\\PI\\Proiect\\Project\\images\\pic2.png",
                        IMREAD_COLOR);
    //imshow("Second Original Image", pic2);

    RotatedRect face1 = showFace(pic1, 1);
    vector<Point> eyes1 = showEyes(pic1, face1,  1);
    showCorrected(pic1, face1, eyes1, 1);

    //RotatedRect face2 = showFace(pic2, 2);
    //vector<Point> eyes2 = showEyes(pic2, face2,  2);
    //showCorrected(pic2, face2, eyes2, 2);

    waitKey();

    return 0;
}