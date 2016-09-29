/*
 * MPI functs. Caution: No validity checks done on input structures..
*/
#ifndef TRAIN_HPP
#define TRAIN_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <regex>
#include <iomanip>
#include <math.h> // for NaN


#include "mydatatype.hpp"
#include "lib.hpp"

// Define constants

using namespace std;
using namespace cv;

class cTrain
{
	public:
		vector<vector<vector<double>>> exec(Mat &BI, Mat &GI); // binary and grey level image

	private:
		//void shapeStats(const cv::Mat &greyImage, const cv::Mat &binary_image, vector<vector<double>> &output_container);
};

#endif
