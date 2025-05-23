
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "red-eyes.h"

#include <stack>

using namespace std;
using namespace cv;

bool IsInside(int img_rows, int img_cols, int i, int j){
    if (i < 0 || i >= img_rows || j < 0 || j >= img_cols) {
        return false;
    }
    return true;
}

//daca un pixel obiect (alb) este gasit, toti pixelii din structura de vecini vor lua culoarea obiectului
Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter){
    Mat dst, aux;
    int rows, cols;

    rows = source.rows;
    cols = source.cols;
    dst = source.clone();
    for (int nr = 0; nr < no_iter; nr++) {
        aux = dst.clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uchar val = aux.at<uchar>(i, j);
                if (val == 255) {
                    for (int k = 0; k < neighborhood.size; k++) {
                        int x = i + neighborhood.di[k];
                        int y = j + neighborhood.dj[k];
                        if (IsInside(rows, cols, x, y)) {
                            dst.at<uchar>(x, y) = val;
                        }
                    }
                }
            }
        }
    }

    return dst;

}

//daca un pixel obiect (alb) este gasit si daca unul dintre pixelii din structura de vecini nu este un pixel obiect atunci
//pixelul gasit initial va lua culoarea fundalului
Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter){
    Mat dst, aux;
    int rows, cols;

    rows = source.rows;
    cols = source.cols;
    dst = source.clone();
    for (int nr = 0; nr < no_iter; nr++) {
        aux = dst.clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uchar val = aux.at<uchar>(i, j);
                bool objectPoints = true;
                if (val == 255) {
                    for (int k = 0; k < neighborhood.size; k++) {
                        int x = i + neighborhood.di[k];
                        int y = j + neighborhood.dj[k];
                        if (!IsInside(rows, cols, x, y) || aux.at<uchar>(x, y) == 0) {
                            objectPoints = false;
                            break;
                        }
                    }
                    if (objectPoints == false) {
                        dst.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    }
    return dst;
}

//elimina elemente mai mici decat elementul structural si netezeste contururile
Mat opening(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat dst, aux;

    dst = source.clone();
    for (int nr = 0; nr < no_iter; nr++) {
        aux = dst.clone();
        dst = erosion(aux, neighborhood, 1);
        dst = dilation(dst, neighborhood, 1);
    }

    return dst;

}

//astupa golurile mici din interiorul obiectelor și face obiectul mai compact
Mat closing(Mat source, neighborhood_structure neighborhood, int no_iter) {

    Mat dst, aux;

    dst = source.clone();
    for (int nr = 0; nr < no_iter; nr++) {
        aux = dst.clone();
        dst = dilation(aux, neighborhood, 1);
        dst = erosion(dst, neighborhood, 1);
    }

    return dst;
}

Mat convertRGBtoYCbCr(Mat src) {
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec3b pixel = src.at<Vec3b>(i, j);
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];
            uchar Y = static_cast<uchar>( 0.299 * R + 0.587 * G + 0.114 * B );
            uchar Cb = static_cast<uchar>(-0.168736 * R - 0.331264 * G + 0.5 * B + 128);
            uchar Cr = static_cast<uchar>( 0.5 * R - 0.418688 * G - 0.081312 * B + 128);
            dst.at<Vec3b>(i, j) = Vec3b(Y, Cr, Cb);
        }
    }
    return dst;
}

Mat convertRGBtoGrayscale(Mat source){
    int rows, cols;
    Mat grayscale_image;

    rows = source.rows;
    cols = source.cols;
    grayscale_image = Mat(rows,cols,CV_8UC1);
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i,j);
            grayscale_image.at<uchar>(i,j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
        }
    }

    return grayscale_image;

}

//de la pixelul curent se verifica vecinii in sens orar pana se gaseste un pixel alb care va reprezenta urmatorul punct de pe contur
contour extract_contour(Mat src, Point P0) {
    int dir = 7;
    Point Pcurrent = P0;
    vector<Point> border;
    vector<int>  dir_vector;
    border.push_back(P0);

    int n = 0;
    do {
        if (dir % 2 == 0) {
            dir = (dir + 7) % 8;
        }
        else {
            dir = (dir + 6) % 8;
        }

        for (int i = 0; i < 8; ++i) { //clockwise order
            int y = Pcurrent.y + n8_di[dir];
            int x = Pcurrent.x + n8_dj[dir];
            if (IsInside(src.rows, src.cols, y, x) && src.at<uchar>(y, x) == 255) {
                break;
            }
            dir = (dir + 1) % 8;
        }

        Pcurrent = Point(Pcurrent.x + n8_dj[dir], Pcurrent.y + n8_di[dir]);
        dir_vector.push_back(dir);
        border.push_back(Pcurrent);
        ++n;
    } while (!((n >= 2) && (border[n] == border[1]) && (border[n-1] == border[0])));

    // remove the two closing points
    border.pop_back();
    dir_vector.pop_back();
    return { border, dir_vector };
}

Mat draw_contour(vector<Point> border, Mat source) {
    Mat dst;

    dst = Mat::zeros(source.rows, source.cols, CV_8UC1);
    for (int n = 0; n < border.size(); n++) {
        dst.at<uchar>(border[n].y, border[n].x) = 255;
    }

    return dst;
}

//Eliminarea componentei conectate care porneste de la P0 prin DFS pentru a evita contururi duplicate
void removeRegion(Mat src, Point P0) {
    stack<Point> stack;
    stack.push(P0);
    while (!stack.empty()) {
        Point p = stack.top();
        stack.pop();
        //daca e inafara imaginii trece la urmatorul
        if (!IsInside(src.rows, src.cols, p.y, p.x)) {
            continue;
        }
        //daca e pixel obiect (negru) trece mai departe
        if (src.at<uchar>(p.y, p.x) != 255) {
            continue;
        }
        //daca e pixel obiect si se afla in interiorul imaginii devine negru
        src.at<uchar>(p.y,p.x) = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (!(i == 0 && j == 0)) {
                    stack.push({p.x + j,p.y + i});  //se adauga toti cei 8 vecini pentru a fi verificati
                }
            }
        }
    }
}

void findContours(Mat src, vector<vector<Point>>& contours)
{
    while (true) {
        Point P0(-1,-1);
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                if (src.at<uchar>(i, j) == 255) {
                    P0 = {j, i};
                    break;
                }
            }
            if (P0.x >= 0) {
                break;
            }
        }

        if (P0.x < 0) {
            break;
        }

        contour cnt = extract_contour(src, P0);
        contours.push_back(cnt.border);

        //Remove region so it wont be refound
        removeRegion(src, P0);
    }
}

int areaOfContour(vector<Point> border, int rows, int cols)
{
    Rect rect = boundingRect(border);
    rect = rect & Rect(0, 0, cols, rows); //intersectie

    int total = 0;

    for (int j = rect.y; j < rect.y + rect.height; j++) //pentru fiecare linie orizontala din dreptunghi
    {
        vector<int> nodes;

        int n = (int)border.size();
        for (int i = 0; i < n; i++) //parcurgem nodurile din contur
        {
            Point p1 = border[i];
            Point p2 = border[(i + 1) % n];

            bool cross = (p1.y <= j && p2.y > j) || (p2.y <= j && p1.y > j); //daca un punct se afla deasupra scan-lineului iar celelalt sub
            if (cross)
            {
                int x = p1.x + (double)(j - p1.y) * (p2.x - p1.x) / (p2.y - p1.y); //x-ul intersectiei
                nodes.push_back(x);
            }
        }

        if (nodes.empty())
            continue;

        sort(nodes.begin(), nodes.end());

        for (int k = 0; k + 1 < (int)nodes.size(); k += 2)
        {
            int x0 = max(nodes[k], rect.x);
            int x1 = min(nodes[k + 1], rect.x + rect.width - 1);
            if (x1 >= x0)
                total += (x1 - x0 + 1);
        }
    }

    return total;
}

bool detectFace(Mat src, RotatedRect& faceEllipse) {
    Mat ycrcb = convertRGBtoYCbCr(src);
    Mat skinMask;
    inRange(ycrcb, Scalar(0, 140, 100), Scalar(255, 165, 135), skinMask);

    Mat closed = closing(skinMask,  ELLIPSE_5x5, 1);
    Mat opened = opening(closed, ELLIPSE_5x5, 1);
    skinMask = opened.clone();
    //imshow("skinMask", skinMask);

    vector<vector<Point>> contours;
    findContours(skinMask, contours);
    //cout<<"Contours: "<<contours.size()<<endl;
    if (contours.empty())
        return false;

    int maxArea = 0;
    int maxIdx = -1;
    for (int i = 0; i < contours.size(); i++) {
        int area = areaOfContour(contours[i], skinMask.rows, skinMask.cols);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    if (maxIdx >= 0 && contours[maxIdx].size() >= 5 && maxArea > 1000) {
        faceEllipse = fitEllipse(contours[maxIdx]);
        return true;
    }
    return false;
}

RotatedRect showFace(const Mat src, int nr) {
    RotatedRect faceEllipse;
    if (detectFace(src, faceEllipse)) {
        Mat dst = src.clone();
        ellipse(dst, faceEllipse, Scalar(0, 0, 255), 2);
        const String s = "Detected Face " + to_string(nr);
        imshow(s, dst);
    }else {
        cout << "Face not detected" << endl;
    }
    return faceEllipse;
}

vector<Point> showEyes(const Mat src, const RotatedRect faceEllipse, int nr)
{
    Mat dst = src.clone();
    vector<Point> eyes;
    if (detectEyes(src, faceEllipse, eyes)) {
        float boxW = faceEllipse.size.width  * 0.20f;
        float boxH = faceEllipse.size.height * 0.10f;
        for (int i = 0; i < eyes.size(); i++) {
            Point eye = eyes[i];
            Point topLeft( cvRound(eye.x - boxW/2), cvRound(eye.y - boxH/2) );
            Point bottomRight( cvRound(eye.x + boxW/2), cvRound(eye.y + boxH/2) );
            rectangle(dst, topLeft, bottomRight, Scalar(0,255,0), 2);
        }
    } else {
        cout << "Eyes not detected in image " << nr << endl;
    }
    imshow("Detected Eyes " + to_string(nr), dst);
    return eyes;
}

void showCorrected(Mat src, const RotatedRect faceEllipse, vector<Point> eyeCenters, int nr) {
    if (correctRedEyes(src, faceEllipse, eyeCenters)) {
        imshow("Corrected Image " + to_string(nr), src);
    } else {
        cout << "Image not corrected" << endl;
    }
}

Mat thresholding(const Mat src, uchar threshValue, bool invert) {
    Mat dst = src.clone();
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (invert == false) {
                if (dst.at<uchar>(i, j) < threshValue) {
                    dst.at<uchar>(i, j) = 255;
                } else {
                    dst.at<uchar>(i, j) = 0;
                }
            } else {
                if (dst.at<uchar>(i, j) < threshValue) {
                    dst.at<uchar>(i, j) = 0;
                } else {
                    dst.at<uchar>(i, j) = 255;
                }
            }
        }
    }
    return dst;
}

Mat crop(Mat src, Rect rect) {
    rect = rect & Rect(0, 0, src.cols, src.rows);

    Mat dst(rect.height, rect.width, src.type());

    for (int i = 0; i < rect.height; ++i) {
        for (int j = 0; j < rect.width; ++j) {
            dst.at<uchar>(i, j) = src.at<uchar>(rect.y + i, rect.x + j);
        }
    }
    return dst;
}

bool detectEyes(const Mat src, const RotatedRect faceEllipse, vector<Point>& eyeCenters) {
    Mat gs = convertRGBtoGrayscale(src);
    Rect faceRect = faceEllipse.boundingRect();
    Mat faceCrop = crop(gs, faceRect);

    int half = faceCrop.rows / 2;
    Mat eyes = crop(faceCrop, Rect(0, 0, faceCrop.cols, half));
    //imshow("Eyes", eyes);

    Mat aux;
    double otsuVal = threshold(eyes, aux, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    Mat thr = thresholding(eyes, static_cast<int>(otsuVal), false);
    cout << "Thresh Val: " << otsuVal << endl;
    //imshow("Thresholding", thr);

    thr = opening(thr, ELLIPSE_3x3, 1);

    vector<vector<Point>> contours;
    findContours(thr, contours);

    vector<Point> candidates;
    for (int i = 0; i < contours.size(); i++) {
        int area = areaOfContour(contours[i], thr.rows, thr.cols);
        if (area < 10 || area > 500)
            continue;

        int minX=INT_MAX, maxX=0, minY=INT_MAX, maxY=0;  //box-ul care cuprinde ochiul
        for (int j = 0; j < contours[i].size(); j++) {
            Point p = contours[i][j];
            minX = min(minX, p.x);
            maxX = max(maxX, p.x);
            minY = min(minY, p.y);
            maxY = max(maxY, p.y);
        }

        int w = maxX - minX + 1;
        int h = maxY - minY + 1;
        float aspect = float(w) / h; //forma conturului
        if (aspect < 0.3f || aspect > 3.0f) //daca e prea turtita sau prea alungita nu e un candidat bun
            continue;

        Point center(minX + w/2, minY + h/2);
        candidates.push_back(center);
    }

    if (candidates.size() < 2)
        return false;

    float bestDist = 0;
    Point eye1, eye2;
    for (int i = 0; i < candidates.size(); i++) {
        for (int j = i+1; j < candidates.size(); j++) {
            float dx = candidates[i].x - candidates[j].x; //cat de departe sunt pe orizontala
            float dy = abs(candidates[i].y - candidates[j].y); //cat de aliniati sunt pe verticala
            float dist  = sqrt(dx * dx + dy * dy); //dist dintre puncte
            if (dy < 15 && dist > bestDist) {      //dy trebuie sa fie cat mai mic iar dist cat mai mare pentru o pereche buna
                bestDist = dist;
                eye1 = candidates[i];
                eye2 = candidates[j];
            }
        }
    }
    if (bestDist == 0)
        return false;

    eye1 += Point(faceRect.x, faceRect.y);
    eye2 += Point(faceRect.x, faceRect.y);
    eyeCenters.push_back(eye1);
    eyeCenters.push_back(eye2);
    return true;
}

Mat cropBGR(const Mat src, Rect rect) {
    rect &= Rect(0,0,src.cols,src.rows);
    Mat dst(rect.height, rect.width, src.type());
    for(int i = 0; i < rect.height; i++) {
        for(int j = 0; j < rect.width; j++) {
            Vec3b vec = src.at<Vec3b>(rect.y + i, rect.x + j);
            dst.at<Vec3b>(i, j) = vec;
        }
    }
    return dst;
}

void splitBGR(const Mat src, vector<Mat>& channels) {
    int rows = src.rows, cols = src.cols;
    channels[0] = Mat(rows, cols, CV_8UC1);
    channels[1] = Mat(rows, cols, CV_8UC1);
    channels[2] = Mat(rows, cols, CV_8UC1);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            Vec3b vec = src.at<Vec3b>(i, j);
            channels[0].at<uchar>(i, j) = vec[0];
            channels[1].at<uchar>(i, j) = vec[1];
            channels[2].at<uchar>(i, j) = vec[2];
        }
    }
}

void maxMat(const Mat A, const Mat B, Mat& dst) {
    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < A.cols; j++) {
            if (A.at<uchar>(i, j) > B.at<uchar>(i, j)) {
                dst.at<uchar>(i, j) = A.at<uchar>(i, j);
            }
             else {
                 dst.at<uchar>(i, j) = B.at<uchar>(i, j);
             }
        }
    }
}

void subtractMat(const Mat A, const Mat B, Mat& dst) {
    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < A.cols; j++) {
            int diff = A.at<uchar>(i, j) - B.at<uchar>(i, j);
            if (diff > 0) {
                dst.at<uchar>(i, j) = diff;
            } else {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
}


bool correctRedEyes(Mat& src, const RotatedRect faceEllipse, const vector<Point> eyeCenters) {
    if (eyeCenters.size() < 2)
        return false;

    float boxW = faceEllipse.size.width  * 0.20f;
    float boxH = faceEllipse.size.height * 0.10f;

    for (int i = 0; i < eyeCenters.size(); i++) {
        Point eye = eyeCenters[i];
        Rect eyeRect(int(eye.x - boxW/2), int(eye.y - boxH/2), int(boxW), int(boxH));
        eyeRect &= Rect(0, 0, src.cols, src.rows);
        if (eyeRect.width < 2 || eyeRect.height < 2)
            continue;

        Mat eyeBGR = cropBGR(src, eyeRect);
        Mat eyeLab;
        cvtColor(eyeBGR, eyeLab, COLOR_BGR2Lab); //mai usor de sters rosul fara a strica luminozitatea


        vector<Mat> bgr(3);
        splitBGR(eyeBGR, bgr);
        Mat maxGB = eyeBGR.clone();
        maxMat(bgr[1], bgr[0], maxGB);
        Mat diff = eyeBGR.clone();
        subtractMat(bgr[2], maxGB, diff);
        //in diff raman pixelii care au valoarea R care depaseste cu mult valoarea G, B
        Mat redMask = thresholding(diff, 50, true);
        redMask = closing(redMask, ELLIPSE_3x3, 1);

        Mat grayEye = convertRGBtoGrayscale(eyeBGR);
        Mat pupilMask = thresholding(grayEye, 20, false);
        pupilMask = opening(pupilMask, ELLIPSE_3x3, 2);

        Mat scleraMask;
        inRange(eyeBGR, Scalar(200,200,200), Scalar(255,255,255), scleraMask);

        Mat irisMask = 255 - (scleraMask | pupilMask);

        Scalar meanLab = mean(eyeLab, irisMask);  //valoarea medie a culorii irisului
        uchar meanA = static_cast<uchar>(meanLab[1]); //a - opozitia verde/rosu
        uchar meanB = static_cast<uchar>(meanLab[2]); //b - opositia albastru/galben

        for (int i = 0; i < eyeLab.rows; i++) {
            for (int j = 0; j < eyeLab.cols; j++) {
                if (redMask.at<uchar>(i, j)) {
                    Vec3b& LAb = eyeLab.at<Vec3b>(i, j);
                    LAb[1] = meanA;
                    LAb[2] = meanB;
                }
            }
        }

        for (int i = 0; i < eyeLab.rows; i++) {
            for (int j = 0; j < eyeLab.cols; j++) {
                if (pupilMask.at<uchar>(i, j)) {
                    eyeLab.at<Vec3b>(i, j) = Vec3b(0,128,128);
                }
            }
        }

        Mat correctedBGR;
        cvtColor(eyeLab, correctedBGR, COLOR_Lab2BGR);
        correctedBGR.copyTo(src(eyeRect));
    }

    return true;
}