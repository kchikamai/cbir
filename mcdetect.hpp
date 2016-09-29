/*
 * MPI functs. Caution: No validity checks done on input structures..
*/
#ifndef MCDETECT_HPP
#define MCDETECT_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <regex>
#include <iomanip>
#include <math.h> // for NaN

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/gpu/gpu.hpp>

#include "regionprops.h"
#include "region.h"
#include "lib.hpp"

#define MASTER 0

// Other constants defined in lib.hpp
const int GAU_SIGMA = 2.9; // Gaussian Sigma_x (=Sigma_y)

using namespace std;
using namespace cv;

class cMcDetect
{
	public:
		Mat exec(Mat &in);

	private:
		Mat gausFilt(Mat &in);
		Mat medFilt(Mat &in);
		Mat borderProcess(Mat &in);
		void threshold(Mat &in, Mat &out);
		Mat denoise(const cv::Mat &BI);
		Mat shapeStats(const cv::Mat &BI);
		Mat getFeats(const cv::Mat &BI, cv::Mat &GI, bool o_getclus);
		int * distMatrix(vector<Point> centroids,vector<float> *dm=NULL);
		void cvh(const cv::Mat &BI);
		void anotherfunc(cv::Mat &img);
		void getHaralick(Mat &GI);
		Mat removeSmallArea(const Mat &BI);
};

#endif
