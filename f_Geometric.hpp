/*
 * MPI functs. Caution: No validity checks done on input structures..
*/
#ifndef GEOMETRIC_HPP
#define GEOMETRIC_HPP

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

#include "mydatatype.hpp"
#include "lib.hpp"
#include "regionprops.h"
#include "region.h"

// Define constants


using namespace std;
using namespace cv;

class cGeometric
{
	public:
		vector<vector<vector<double>>> exec(Mat &BI, Mat &GI); // binary and grey level image

	private:
		void shapeStats(const cv::Mat &greyImage, const cv::Mat &binary_image, vector<vector<double>> &output_container);
		vector<vector<vector<double>>> getFeats(const cv::Mat &binary_image, cv::Mat &grey_image, bool o_get_cluster);
		int * distMatrix(vector<Point> &centroids,vector<float> *distance_matrix=NULL);
		void cvh(const cv::Mat &binary_image);
};

#endif
