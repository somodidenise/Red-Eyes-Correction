#ifndef LAB6_H
#define LAB6_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const int n8_di[8] = {0,-1,-1, -1, 0, 1, 1, 1};
const int n8_dj[8] = {1, 1, 0, -1, -1,-1, 0, 1};

typedef struct{
    int size;
    vector<int> di;
    vector<int> dj;
} neighborhood_structure;

typedef struct{
    vector<Point> border;
    vector<int> dir_vector;
} contour;

static const neighborhood_structure ELLIPSE_5x5 = {
    13,
    { -2,  -1,  -1,  -1,   0,   0,   0,   0,   0,   1,   1,   1,   2 },
    {  0,  -1,   0,   1,  -2,  -1,   0,   1,   2,  -1,   0,   1,   0 }
};

static const neighborhood_structure ELLIPSE_3x3 = {
    5,
    { -1,  0,  0,  0,  1 },
    {  0, -1,  0,  1,  0 }
};

static const neighborhood_structure n8 = {
    8,
    {0,-1,-1,-1,0,1,1,1},
    {1,1,0,-1,-1,-1,0,1}
};

bool IsInside(int img_rows, int img_cols, int i, int j);
Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat opening(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat closing(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat convertRGBtoYCbCr(Mat src);
Mat convertRGBtoGrayscale(Mat source);
contour extract_contour(Mat src, Point P0);
Mat draw_contour(vector<Point> border, Mat source);
void removeRegion(Mat src, Point P0);
void findContours(Mat src, vector<vector<Point>>& contours);
int areaOfContour(vector<Point> border, int rows, int cols);
Mat thresholding(Mat src, uchar threshValue, bool invert);
Mat crop(Mat src, Rect rect);
Mat cropBGR(Mat src, Rect rect);
void splitBGR(Mat src, vector<Mat>& channels);
void maxMat( Mat A, Mat B, Mat& dst);
void subtractMat(Mat A, Mat B, Mat& dst);


bool detectFace(Mat src, RotatedRect& faceEllipse);
RotatedRect showFace(Mat src, int nr);
bool detectEyes(Mat src, RotatedRect faceEllipse, vector<Point>& eyeCenters);
vector<Point> showEyes(Mat src, RotatedRect faceEllipse, int nr);
bool correctRedEyes(Mat& src, RotatedRect faceEllipse, vector<Point> eyeCenters);
void showCorrected(Mat src, RotatedRect faceEllipse, vector<Point> eyeCenters, int nr);

#endif