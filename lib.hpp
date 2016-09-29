/*
 * MPI functs. Caution: No validity checks done on input structures..
*/
#ifndef LIB_HPP
#define LIB_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <regex>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/gpu/gpu.hpp>

#include "mydatatype.hpp"

#define MASTER 0

// Relevant to f_Haralick class
const static int GLCM_MAX=255; // exclusive GLCM level. glcm[0..GLCM_MAX-1]
const static int n_GLCMS=20; // number of glcms
const static int n_h_Feats=6; // #haralick features: Contrast, correlation, energy, homogeneity, Entropy, Max probability

// relevant to geometric analysis (mcdetect.cpp, f_Haralick.cpp)
const static float t_CLUS_DIST=50; //maximum inter-distance required for mcs to form cluster
const static int t_CLUS_SZ=2;
const static int t_MIN_AREA=2, t_MAX_AREA=50;
const static double t_Eccentricity=0.90;
const static int N_FEATS=6; 	// Area, Area (blob pixel count), Compactness, Orientation, Eccentricity, Solidity

const static int DIM_CLUS_FEAT=144; // cluster feature vector length
const static int IND_CLUS_FEAT=5+(n_GLCMS*n_h_Feats)+4; // individual feature vector length: 6 shape + 120 haralick + 4 bounding box dimensions
const static int PCA_VEC_SZ=35;

const string DATA_DIR="/media/femkha/HDB_FemkhaAcer/images/full/";

const static string trainPosDir="/media/femkha/HDB_FemkhaAcer/images/full/calc/";//"/media/femkha/HDB_FemkhaAcer/images2/calc_gt/train/"; // Positive training samples
const static string trainNegDir="/media/femkha/HDB_FemkhaAcer/images/full/norm/";//"/media/femkha/HDB_FemkhaAcer/images2/norm_gt/train/"; // Negative training samples
const static string testPosDir ="/media/femkha/HDB_FemkhaAcer/images2/calc_gt/test/"; // Positive testing samples
const static string testNegDir ="/media/femkha/HDB_FemkhaAcer/images2/norm_gt/test/"; // Negative testing samples

const static string EXTS = "png|jpg|pgm";

using namespace std;
using namespace cv;

class cLib
{
	public:
		void rescaleIntensity();
		int readImage(string path, Mat &mat);
		void converter(cMyDataType &im, Mat &mat, int opt); /// opt == {1:cMyDataType => Mat, 2:Mat => cMyDataType}
		void output(const cMyDataType &im);
		template <class T> void output(const vector<vector<T>> &v); // print vector
		template <class T> void output(const vector<T> &v); // print vector
		void getMeanStdev(Mat &Mat, vector<double> &mn, vector<double> &stdev, bool dim1=true, bool o_removeNanInf=true); // get the mean and standard deviation of the 2-d vector
		void getMeanStdev(vector<vector<double>> &v, vector<double> &mn, vector<double> &stdev, bool dim1=true, bool o_removeNanInf=true); // get the mean and standard deviation of the 2-d matrix
		void Mat2Vec(vector<vector<double>> &v, Mat &m, bool mat2vec=true);
		template <class T> Mat removeNanInf(Mat M, bool t_Nan=true, bool t_Inf=true, bool t_byRows=true);
		void serializeMat(string impath, Mat &mat, bool o_READ=true);


	private:
		int myRank;
};

// templated functions must be seen by compiler to implement all possible variations
template <class T> void cLib::output(const vector<vector<T>> &v)
{
	for (size_t i = 0; i < v.size(); i++) {
		this->output<T>(v[i]);
	} printf("\n");
}
template <class T> void cLib::output(const vector<T> &v)
{
	for (size_t i = 0; i < v.size(); i++) {
		//printf("%-7.7g ",v[i]);
		printf("%g ",v[i]);
	} printf("\n");
}
template <class T> Mat cLib::removeNanInf(Mat M, bool t_Nan, bool t_Inf, bool t_byRows)
{
	// Remove Rows (t_byRows==true) and/or columns (t_byRows==false)  from Mat structure (type=T) M containing
	//  NaN values (t_Nan == true), and/or Infinity values (t_Inf==true)
	Mat out;
	std::vector<size_t> x_rows(M.rows,0), x_cols(M.cols,0);
	for (size_t i = 0; i < M.rows; i++) {
		for (size_t j = 0; j < M.cols; j++) {
			if(t_Nan && cvIsNaN(M.at<T>(i,j))) {
				x_rows[i]=1; x_cols[j]=1;
			}
			if(t_Inf && cvIsInf(M.at<T>(i,j))) {
				x_rows[i]=1; x_cols[j]=1;
			}
		}
	}

	if (t_byRows) {
		for (size_t i = 0; i < x_rows.size(); i++) {
			if (!x_rows[i]) {
				if(out.data) vconcat(out, M.row(i), out);
				else out = M.row(i);
			}
		}
	} else {
		for (size_t i = 0; i < x_cols.size(); i++) {
			if (!x_cols[i]) {
				if(out.data) hconcat(out, M.col(i), out);
				else out = M.col(i);
			}
		}
	}
	return out;
	// End of function
}

#endif /*lib*/
