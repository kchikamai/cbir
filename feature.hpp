/*
 * MPI functs. Caution: No validity checks done on input structures..
*/
#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <regex>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ml/ml.hpp>

#include "mydatatype.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

typedef double DBType;

class cFeature
{
	public:
		cFeature(void);
		void exec();
		void normalizeDB(Mat &fv, char c_opt, bool normalizeVec);

		// sent following to private section
		Mat doPCA(Mat &vec, char opt); // conduct PCA on vector
		void doSVM(Mat &data, Mat &lab, Mat &res, char opt); // conduct SVM on vector
		void addFV(size_t fileidx, string impath, float label);
		void generateDB();
		Mat doLDA(Mat &vec, char opt);

	private:
		void selectFV(Mat &fv, char c_opt);
		void getSelData();
		void getDB();
		void saveClassData(bool o_READ);


		Mat i_select, c_select, p_select; // individual, cluster and pca selected features
		Mat i_DB, c_DB; // individual, cluster database




		// Next: Database metadata. Assert that ( |i/c_file_idx| = |i/c_DB| ) and ( |i/c_labels| = |file_name| ).
		struct {
			std::vector<int> i_file_idx, c_file_idx; // file indices
			vector<float> i_labels, c_labels; // +ve or -ve labels, relevant for training
			vector<string> i_file_name, c_file_name;
		} db_data;

		struct {
			Ptr<ml::SVM> i_svm, c_svm;
			Mat s_iDB, s_cDB;
		} svm_data;

		struct {
			Mat i_mean, c_mean;
			bool lda_trained;
			LDA iLda, cLda; // Local discriminant analysis
		} qda_data;

		struct {
			vector<double> i_mean, c_mean, i_stdev, c_stdev; // mean and standard deviation
		} norm_data; // Normalization data

		struct {
			bool i_NORMALIZED, c_NORMALIZED, // nornmalization data (norm_data) set up?
					 SELECTED; // feature vector db selected?
		} db_status;

		struct {
			PCA metadata;
			bool PCA_INIT; // is pca structure initialized?
		} pca_data;

		// End of class
};

#endif
