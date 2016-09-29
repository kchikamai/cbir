/*
 * MPI functs. Caution: No validity checks done on input structures..
*/
#ifndef FEATUREP_HPP
#define FEATUREP_HPP

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
#include <mpi.h>

#include "mydatatype.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

typedef double DBType;

// MPI-specific  definitions
#define MASTER 0
#define LOCAL_MASTER 0
#define LOCAL_GROUP_SZ 8
#define TAG_HEADER 0
#define TAG_DATA 1
#define TAG_RESULT 2
#define TAG_TERMINATE 3
#define SZ_HEADER 10 // instr code, nRows, nCols, #Image, #worker, LOCAL_MASTER, slave_wait_latency, slave_bandwidth, slave_process_time, file count
#define WORKERS 999999 // Excluding the master node.. set at higher value to technically render this setting pointless

class cFeatureP
{
	public:
		cFeatureP(void);
		void exec();
		void normalizeDB(Mat &fv, char c_opt, bool normalizeVec);

		// sent following to private section
		Mat doPCA(Mat &vec, char opt); // conduct PCA on vector
		void doSVM(Mat &data, Mat &lab, Mat &res, char opt); // conduct SVM on vector
		void addFV(size_t fileidx, string impath, Mat GI, float label);
		void generateDB();
		Mat doLDA(Mat &vec, char opt);

	private:
		void selectFV(Mat &fv, char c_opt);
		void getSelData();
		void getDB();
		void saveClassData(bool o_READ);
		void init_stuff();
		void updateK(int N, double Y, double to, int pf);
		void slave();

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

		// Next -> MPI specific variables
		int myRank, glob_myRank,
			loc_master,					// local master group
			*local_group,			// local master assignments (index = rank no)
			num_loc_groups,			// number of local groups
			*master_table,
			node_sz, glob_node_sz,	// number of all threads / cores
			color, key,				// to establish group and ordering in local communication groups
			MAX_FILE_COUNT, 		// Maximum files to read
			HEADER_SZ; 				//size of header data for preliminaries
		double *in_buf, *out_buf, header[SZ_HEADER];
		MPI_Request *Req; 				// MPI Request object
		MPI_Status *Stat;
		MPI_Comm MY_COMM_GRP;
		int Errno;
		double *send_time_sum; int *send_time_N;
		struct {
			int N, k, p, proc_files;
			double sY, B, sto; // Gamma(Y) and Beta (B)
		} opt_k;
		// End of class
};

#endif
