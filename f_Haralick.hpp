/*
 * MPI functs. Caution: No validity checks done on input structures..
*/
#ifndef HARALICK_HPP
#define HARALICK_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <regex>
#include <iomanip>
#include <math.h> // for NaN, log()

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/gpu/gpu.hpp>

#include "lib.hpp"

using namespace std;
using namespace cv;

class cHaralick
{
	public:
		cHaralick();
		~cHaralick();
		double *exec(Mat &in);

	private:
		void calculateGLCM(Mat &GI);
		void harFeats(); // Calculate Haralick features from a_GLCM (run calculateGLCM before this)

		//double a_GLCM[n_GLCMS][GLCM_MAX][GLCM_MAX];
		double ***a_GLCM;
		double h_Feats[n_GLCMS*n_h_Feats]; // #haralick features: Contrast, correlation, energy, homogeneity, Entropy, Max probability
		int max_G, min_G; // maximum and minimum grey levels in image
};

#endif
