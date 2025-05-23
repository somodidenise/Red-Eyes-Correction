// test.cpp
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "red-eyes.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

//Function for second Red Correction method
static void fillHoles(Mat &mask) {
    Mat tmp = mask.clone();
    floodFill(tmp, Point(0,0), Scalar(255));
    Mat tmp2;
    bitwise_not(tmp, tmp2);
    mask = tmp2 | mask;
}

int main(int argc, char** argv) {
    string testDir;
    if (argc > 1) {
        testDir = argv[1];
    } else {
        testDir = "C:\\CTI\\An 3 sem 2\\PI\\Proiect\\Project\\tests";
    }

    //Subfolders names
    vector<string> categories = { "lighting", "noise", "pose", "skin_tones", "red_eyes" };

    CascadeClassifier faceCascade, eyeCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml") ||
        !eyeCascade.load  ("haarcascade_eye.xml")) {
        cerr << "Error loading Haar cascades\n";
        return 1;
    }

    for (int i = 0; i < categories.size(); i++) {
        string subfolderName = categories[i];
        string subfolderPath = testDir + "/" + subfolderName;

        //Take all .jpg/.png files from subfolder
        vector<string> files;
        for (auto& entry : fs::directory_iterator(subfolderPath)) {
            if (!entry.is_regular_file())
                continue;
            string path = entry.path().string();
            string ext  = entry.path().extension().string();
            for (int j = 0; j < ext.size(); j++) {
                ext[j] = char(tolower(ext[j]));
            }
            if (ext == ".jpg" || ext == ".png") {
                files.push_back(path);
            }
        }

        ofstream logFile(subfolderName + "_results.csv");
        logFile << "file,method,face_detected,eyes_detected,correction_done,runtime_ms\n";

        // 5) Process each image in that list
        for (int im = 0; im < files.size(); im++) {
            string path = files[im];
            Mat img = imread(path);
            if (img.empty()) {
                cerr << "Failed to load " << path << "\n";
                continue;
            }

            //Personal implementation testing
            auto t0 = chrono::high_resolution_clock::now();
            RotatedRect face;
            bool faceCheck = detectFace(img, face);

            vector<Point> eyes;
            bool eyesCheck = false, correction = false;
            if (faceCheck) {
                eyesCheck = detectEyes(img, face, eyes);
                if (eyesCheck) {
                    Mat tmp = img.clone();
                    correction = correctRedEyes(tmp, face, eyes);
                }
            }
            auto t1 = chrono::high_resolution_clock::now();
            double time = chrono::duration<double, milli>(t1 - t0).count();

            logFile
              << path << ",custom,"
              << faceCheck << "," << eyesCheck << "," << correction << "," << time << "\n";

            //Haar Cascade
            t0 = chrono::high_resolution_clock::now();
            vector<Rect> faces;
            faceCascade.detectMultiScale(img, faces);
            bool faceCheckH = !faces.empty();

            RotatedRect faceH;
            vector<Point> eyesH;
            bool eyesCheckH = false, correctionH = false;
            if (faceCheckH) {
                Rect r = faces[0];
                faceH = RotatedRect(Point2f(r.x + r.width/2.f, r.y + r.height/2.f), Size2f (r.width, r.height), 0);

                //detect eyes
                vector<Rect> eRects;
                eyeCascade.detectMultiScale( img(r), eRects, 1.3, 4, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));
                if (eRects.size() >= 2) {
                    eyesCheckH = true;
                    for (int j = 0; j < eRects.size(); j++) {
                        Mat eye = img(r)(eRects[j]);
                        //Mat &eyeOutROI = imgOut(r)(eRects[j]);

                        vector<Mat>bgr(3);
                        split(eye,bgr);
                        Mat mask = (bgr[2] > 150) & (bgr[2] > ( bgr[1] + bgr[0] ));

                        fillHoles(mask);
                        dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);

                        if (countNonZero(mask) > 0) {
                            correctionH = true;
                        }

                        Mat mean = (bgr[0]+bgr[1])/2;
                        mean.copyTo(bgr[0], mask);
                        mean.copyTo(bgr[1], mask);
                        mean.copyTo(bgr[2], mask);

                        Mat eyeOut;
                        merge(bgr,eyeOut);
                        //eyeOut.copyTo(eyeOutROI);
                    }
                }
            }
            t1 = chrono::high_resolution_clock::now();
            double time2 = chrono::duration<double, milli>(t1 - t0).count();

            logFile
              << path << ",haar,"
              << faceCheckH << "," << eyesCheckH << "," << correctionH << "," << time2 << "\n";
        }

        logFile.close();
        cout << "Generated " << subfolderName << "_results.csv for " << files.size() << " images\n";
    }

    return 0;
}
