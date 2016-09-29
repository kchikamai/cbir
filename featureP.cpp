#include "featureP.hpp"
#include "mcdetect.hpp"
#include "f_Haralick.hpp"
#include "f_Geometric.hpp"
#include "lib.hpp"
#include <fstream>



#define READ_DB 0
#define READ_PCA 0
#define READ_SVM 0
#define READ_LDA 0

cFeatureP::cFeatureP(void)
{
	MAX_FILE_COUNT = 2;

	db_status.i_NORMALIZED = db_status.c_NORMALIZED; db_status.SELECTED;
	qda_data.lda_trained=false;

	// Next: set SVM structure. Use param to simply this
	svm_data.i_svm = ml::SVM::create();
	svm_data.i_svm->setType(ml::SVM::C_SVC);	svm_data.i_svm->setKernel(ml::SVM::POLY);
	svm_data.i_svm->setGamma(2.7803);	svm_data.i_svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
	svm_data.i_svm->setC(1000); svm_data.i_svm->setDegree(3);

	svm_data.c_svm = ml::SVM::create();
	svm_data.c_svm->setType(ml::SVM::C_SVC);	svm_data.c_svm->setKernel(ml::SVM::POLY);
	svm_data.c_svm->setGamma(2.7803);	svm_data.c_svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
	svm_data.c_svm->setC(1000); svm_data.c_svm->setDegree(3);

	this->getSelData();
	//cout<<this->i_select<<"\n-------\n"<<this->p_select<<"\n-------\n";
	MPI_Init(0,0);
	init_stuff();
}
void cFeatureP::exec()
{
	double tmp_time=MPI_Wtime();
	generateDB();
	tmp_time = MPI_Wtime()-tmp_time;
	std::cout << "time--: " << tmp_time << std::endl;
}
void cFeatureP::init_stuff()
{
	this->glob_node_sz = MPI::COMM_WORLD.Get_size();
	this->glob_myRank = MPI::COMM_WORLD.Get_rank();

	this->key=this->glob_myRank; // set to default for now
	if(glob_myRank>=MAX_FILE_COUNT) color=MPI_UNDEFINED;

	// replace following with better strategy
	//MPI_Comm_dup(MPI::COMM_WORLD, &MY_COMM_GRP); //MY_COMM_GRP=MPI::COMM_WORLD;
	MPI_Comm_split(MPI_COMM_WORLD, this->color, this->key, &MY_COMM_GRP);

	// Get global and local rank/size
	if (MPI_COMM_NULL != MY_COMM_GRP) {
		MPI_Comm_size(MY_COMM_GRP, &this->node_sz);
		MPI_Comm_rank(MY_COMM_GRP, &this->myRank);
	}

	// other init stuff
	this->Req=new MPI_Request;
	this->Stat=new MPI_Status;
	this->loc_master = MASTER;
	this->master_table = 0;
	this->local_group = 0;

	opt_k.N=MAX_FILE_COUNT, opt_k.k=1, opt_k.p=MPI::COMM_WORLD.Get_size();
	//printf("Local rank:%d of:%d. \t Global rank:%d of:%d, color:%d\n",myRank,node_sz,glob_myRank,glob_node_sz,color);
}
void cFeatureP::addFV(size_t fileidx, string impath, Mat GI, float label)
{
	// Add individual+cluster features for image 'fileidx' to main DB. Selection is done implicitly
	// raw_feats is the original individual+cluster (dim = 129+144) vector
	cLib L; cMcDetect M; cGeometric G; Mat raw_mat, BI;

	BI = M.exec(GI);// Get binary image
	//namedWindow("Display Image", WINDOW_NORMAL); imshow("Display Image", GI); waitKey(0);
	std::cout << "-- extract feats --" << std::endl;
	vector<vector<vector<double>>> raw_feats = G.exec(BI,GI);
	std::cout << "-- -- done extract feats --" << std::endl;
	if (raw_feats.size() && raw_feats[0].size() && raw_feats[0][0].size()) {
		int i_feats = raw_feats[0].size();
		for (size_t i = 0; i < raw_feats[0].size(); i++) { // Insert the file index for each individual object feature
			db_data.i_file_idx.push_back(fileidx);
			db_data.i_labels.push_back(label);
			db_data.i_file_name.push_back(impath);
		}
		// Start with individual features.
		L.Mat2Vec(raw_feats[0], raw_mat, false);
		this->selectFV(raw_mat,'i'); // remove irrelevant features
		if (this->i_DB.cols) {
			vconcat(raw_mat, this->i_DB, this->i_DB); // add to main feature database
		} else {
			this->i_DB = raw_mat;
		}

		if(raw_feats[1].size() && raw_feats[1][0].size()){ // add cluster features
			for (size_t i = 0; i < raw_feats[1].size(); i++) { // Insert the file index for each cluster object feature
				db_data.c_file_idx.push_back(fileidx);
				db_data.c_labels.push_back(label);
				db_data.c_file_name.push_back(impath);
			}
			L.Mat2Vec(raw_feats[1], raw_mat, false);
			this->selectFV(raw_mat,'c'); // remove irrelevant features
			if (this->c_DB.cols) {
				vconcat(raw_mat, this->c_DB, this->c_DB); // add to main feature database
			} else {
				this->c_DB = raw_mat;
			}
		}
	}
	// End of function
}
void cFeatureP::selectFV(Mat &fv, char c_opt)
{
	// Select features from fv that passed relevance criteria. getSelData() should be called first
	// to set up the columns to be selected, before this method is called
	// If fv is not provided then selection is done on stored database. c_opt determines class of
	// features (i=individual, c=cluster, p=pca)
	if (fv.cols) {
		Mat t_DB;
		if('i'==c_opt && this->i_select.cols) { // select 35 individual features from original 129
			t_DB = fv.col(this->i_select.at<uchar>(0));
			for (size_t i = 1; i < this->i_select.cols; i++) {
				hconcat(t_DB, fv.col(this->i_select.at<uchar>(i)), t_DB);
			}
		} else if('c'==c_opt && this->c_select.cols) { // select 99 cluster features from original 144
			t_DB = fv.col(this->c_select.at<uchar>(0));
			for (size_t i = 1; i < this->c_select.cols; i++) {
				hconcat(t_DB, fv.col(this->c_select.at<uchar>(i)), t_DB);
			}
		} else if('p'==c_opt) {
			t_DB = fv.col(this->c_select.at<uchar>(0));
			for (size_t i = 1; i < this->p_select.cols; i++) {
				hconcat(t_DB, fv.col(this->p_select.at<uchar>(i)), t_DB);
			}
		}
		fv = t_DB;
	}
	// End of function
}
void cFeatureP::normalizeDB(Mat &fv, char c_opt, bool normalizeVec)
{
	// Prepare the normalization data as well as normalize the database (normalizeVec), OR,
	// normalize feature vector fv column-wise (normalizeVec=true).
	cLib L;
	if (normalizeVec) {
		if ('i'==c_opt && db_status.i_NORMALIZED) {
			for (size_t i = 0; i < fv.cols; i++) {
				fv.col(i) = (fv.col(i)-this->norm_data.i_mean[i])/this->norm_data.i_stdev[i];
			}
		} else if ('c'==c_opt && db_status.c_NORMALIZED) {
			for (size_t i = 0; i < fv.cols; i++) {
				fv.col(i) = (fv.col(i)-this->norm_data.c_mean[i])/this->norm_data.c_stdev[i];
			}
		} else fprintf(stderr, "Normalization variables (mu and Sigma) not initialized\n");
	} else {
		if (this->i_DB.data && 'i'==c_opt){
			L.getMeanStdev(this->i_DB, this->norm_data.i_mean, this->norm_data.i_stdev, false);
			db_status.i_NORMALIZED=true;
			this->normalizeDB(this->i_DB,'i',1);
		}	else if (this->c_DB.data && 'c'==c_opt){
			L.getMeanStdev(this->c_DB, this->norm_data.c_mean, this->norm_data.c_stdev, false);
			db_status.c_NORMALIZED=true;
			this->normalizeDB(this->c_DB,'c',1);
		}
	}
}
void cFeatureP::getSelData()
{
	// Get preselected features from Matlab data file
	ifstream idxfile("/media/femkha/HDB_FemkhaAcer/images/full//fromM2C");
	std::string str; std::vector<int> inputs[3]; size_t idx=0;
	while (std::getline(idxfile, str))
	{
		if (str.size()) {
			std::istringstream in; in.str(str);
			std::copy( std::istream_iterator<int>( in ), std::istream_iterator<int>(),  std::back_inserter( inputs[idx++] ) );
		}
	}

	this->i_select = Mat::zeros(1,inputs[0].size(), CV_8UC1); // first, initialize all to zero, de-selecting all features
	this->c_select = Mat::zeros(1,inputs[1].size(), CV_8UC1);
	this->p_select = Mat::zeros(1,inputs[2].size(), CV_8UC1);
	//i_select.colRange(i_select.cols-4,i_select.cols) *= 0; // strip off coordinates
	// Next: use features preselected using MATLAB version (read from text file)
	for (size_t i = 0; i < inputs[0].size(); i++) { // individual features
		this->i_select.at<uchar>(i) = inputs[0][i];
	}
	for (size_t i = 0; i < inputs[1].size(); i++) { // cluster features
		this->c_select.at<uchar>(i) = inputs[1][i];
	}
	idx=0;
	for (size_t i = 0; i < inputs[2].size(); i++) { // cluster features
		if(inputs[2][i]) // this vector is in zeros and ones
			this->p_select.at<uchar>(idx++) = i;
	}
	this->p_select = this->p_select.colRange(0,idx);
	// End of function
}
Mat cFeatureP::doPCA(Mat &vec, char opt)
{
	// Transform cluster features vec to PCA features. During generation, the generating vector is first cleaned
	// of all rows containing NaN/Inf values.	Selection not done here, do it after calling this function
	// issue: find a way of handling where there's no data for pca analysis after cleaning
	cLib L;
	string file_name = DATA_DIR + "pca_data.xml";
	if ('c'==opt) { // if no mat provided. set/train/prepare pca structure using pcadata
		pca_data.PCA_INIT = false;
		if (READ_PCA) { // Read PCA structure from file

			FileStorage fs(file_name,FileStorage::READ);
						if (fs.isOpened()) { // Read
							try {
								pca_data.metadata.read(fs.root());
								pca_data.PCA_INIT = true;
								std::cout << "PCA retrieved from File" << std::endl;
							} catch(...) { std::cout << "Error processing PCA file" << std::endl;;}
						} else std::cerr << "Cannot open file. Failed to initialize PCA structure" << std::endl;
		} else { // Generate PCA structure and save
			const int MAX_COMPONENTS = 0; // set to default (retain all components)
			try {
				Mat cleanVec = L.removeNanInf<DBType>(this->c_DB,1,1,1);
				if (cleanVec.data) {
					FileStorage fs(file_name,FileStorage::WRITE);
					//PCA pca(this->c_DB,Mat(),PCA::DATA_AS_ROW,MAX_COMPONENTS);
					PCA pca(cleanVec,Mat(),PCA::DATA_AS_ROW,MAX_COMPONENTS);
					pca.write(fs);
					fs.release();
					pca_data.metadata=pca;
					pca_data.PCA_INIT = true;
					std::cout << "PCA generated" << std::endl;
				} else std::cout << "No data for PCA analysis after cleaning" << std::endl;
			} catch(...){
				std::cout << "Failed to initialize PCA structure" << std::endl;; // all failed
			}
		}
	} else {
		if(true==pca_data.PCA_INIT){
			if('p' == opt){ // project sample given by 'vec'
				//CV_Assert( vec.cols == pcaset.cols );
				Mat coeff(vec.size(),vec.type());
				pca_data.metadata.project(vec, coeff);
				return coeff;
			} else if('r'==opt){// reconstruct
				Mat reconstructed(vec.size(),vec.type());
				pca_data.metadata.backProject(vec, reconstructed);
				return reconstructed;
			}
		} else std::cout << "PCA structure not initialized" << std::endl;
	}
	return Mat();
	// End of function
}
void cFeatureP::doSVM(Mat &data, Mat &labels, Mat &res, char opt)
{
	cLib L;
	String svmpath = DATA_DIR + "svm_data.svm.xml";
	// Train the svm_data.svm structure using data
	Mat_<float> t_data; Mat_<int> t_labels;
	if (!READ_SVM && 't'==opt && i_DB.data) { // train
		// train individual feature machine
		t_data = i_DB.clone(); t_labels = Mat(db_data.i_labels);
		t_data = L.removeNanInf<double>(t_data,1,1,0); // trims down features. hope this remains steady
		svm_data.i_svm->train(t_data,ml::ROW_SAMPLE,t_labels);
		//svm_data.i_svm->save(svmpath);
		// cluster feature machine next
		t_data = c_DB.clone(); t_labels = Mat(db_data.c_labels);
		t_data = L.removeNanInf<double>(t_data,1,1,0); // trims down features. hope this remains steady
		svm_data.c_svm->train(t_data,ml::ROW_SAMPLE,t_labels);
		//svm_data.c_svm->save(svmpath);

		std::cout << svm_data.i_svm->getSupportVectors().size() << " -- " << svm_data.c_svm->getSupportVectors().size() << std::endl;
		std::cout << "SVM models trained and saved" << std::endl;
	} else if(READ_SVM && 't'==opt){ // data not provided, so load from file
		//cv::ml::SVM::load(svmpath); //svm_data.svm = ml::SVM::load<ml::SVM>load(svmpath);
		//svm_data.svm = Algorithm::load<ml::SVM>(svmpath);
		std::cout << "Loaded SVM model from " << svmpath<< std::endl;
	}	else if ('i'==opt && data.data) { // classify/predict
		svm_data.i_svm->predict(data, res);
	} else if ('c'==opt && data.data) { // classify/predict
		svm_data.c_svm->predict(data, res);
	} else std::cerr << "SVM error: illegal option or invalid parameter value(s)" << std::endl;
}
Mat cFeatureP::doLDA(Mat &vec, char opt)
{
	// Train (opt=='t') using saved database (i/c_DB) or project 'vec' based on
	// trained individual (opt=='i') or cluster (opt=='c') data.
	Mat out; cLib L;
	if ('t'==opt) { // train both individual and cluster lda. assumption on existence of both databases
		qda_data.lda_trained = false;
		if (READ_LDA) {
			try {
				qda_data.iLda.load(DATA_DIR+"i_lda.xml");
				qda_data.cLda.load(DATA_DIR+"c_lda.xml");
				std::cout << "LDA structure retrieved from file: "<< DATA_DIR+"lda.xml" << std::endl;
				qda_data.lda_trained = true;
			} catch(...){}
		} else {
			//reduce(i_DB, qda_data.i_mean, 0, CV_REDUCE_AVG);	qda_data.i_mean.convertTo(qda_data.i_mean, CV_64F);
			try {
				vector<double> i_mn, c_mn, tmp_mat;
				L.getMeanStdev(i_DB, i_mn, tmp_mat, 0, 1); L.getMeanStdev(c_DB, c_mn, tmp_mat, 0, 1);
				qda_data.i_mean = Mat(i_mn).reshape(1,1); qda_data.c_mean = Mat(c_mn).reshape(1,1);
				//std::cout << qda_data.c_mean << std::endl<< std::endl;
				qda_data.iLda.compute(i_DB, Mat(db_data.i_labels));
				qda_data.cLda.compute(c_DB, Mat(db_data.c_labels));
				qda_data.lda_trained = true;
				qda_data.iLda.save(DATA_DIR+"i_lda.xml");
				qda_data.cLda.save(DATA_DIR+"c_lda.xml");
				std::cout << "LDA structure generated/trained and saved in files: " << DATA_DIR+"?_lda.xml" << std::endl;
				std::cout << " vectors: " << qda_data.iLda.eigenvectors().size() << std::endl;
			} catch(...){}
		}
	} else { // project
		if ('i'==opt && qda_data.lda_trained)
			out = qda_data.iLda.subspaceProject(qda_data.iLda.eigenvectors(), qda_data.i_mean, vec);
		else if ('c'==opt && qda_data.lda_trained)
			out = qda_data.cLda.subspaceProject(qda_data.cLda.eigenvectors(), qda_data.c_mean, vec);
		else std::cerr << "Cannot project vector. Either LDA structure not trained or insufficient data" << std::endl;
	}
	return out;
	// some of ideas borrowed from: http://answers.opencv.org/question/64165/how-to-perform-linear-discriminant-analysis-with-opencv/
	// and http://stackoverflow.com/questions/32001390/what-is-correct-implementation-of-lda-linear-discriminant-analysis
}
void cFeatureP::generateDB()
{
	// Set up training data
	getDB();
	// Do post-processing on raw vectors
	if(this->i_DB.data && this->c_DB.data){
		//this->saveClassData(false);
		if(this->i_DB.data)	{
			normalizeDB((Mat &)noArray(),'i',0); // Normalize the database
		}
		if(this->c_DB.data)	{
			normalizeDB((Mat &)noArray(),'c',0);
			std::cout << "Doing PCA analysis on data" << std::endl;
			doPCA((Mat &)noArray(),'c'); selectFV(this->c_DB,'p'); // Convert cluster DB to PCA domain space and select relevant components
		}
	}
	doLDA((Mat &)noArray(),'t'); doLDA(i_DB,'i');	doLDA(c_DB,'c');

	// End of function
}
void cFeatureP::getDB()
{
	// Acquire (either by reading in saved file or extracting from scratch) the database into class variables i/c_DB
	// set MAX_FILE_COUNT = -1 to loop over all images, or to any value to restrict to it
	std::vector<bool> t_opts(4,false);  // +ve train, -ve train, +ve test, -ve test;
	t_opts[0] = 1; t_opts[1] = 1; cLib L;
	int LABEL; string dir;//, s_iDb1 = DATA_DIR+"iDb1.xml", s_cDb1 = DATA_DIR+"cDb1.xml";// = "/home/femkha/db_par/1.pgm";

	// Get raw feature vectors
	if (READ_DB) { // Read in saved databases
		this->saveClassData(true);
		if(i_DB.data)	std::cout << "Individual calcification features read from file "<< std::endl;
		else std::cerr << "Cannot read Individual calcification features read from file "<< std::endl;
		if(c_DB.data)	std::cout << "Cluster calcification features read from file "<< std::endl;
		else std::cerr << "Cannot read Cluster calcification features read from file "<< std::endl;
	} else { // Otherwise, generate and save databases
		for (size_t i = 0; i < t_opts.size(); i++) {
			if (t_opts[i] && i==0) { // Get positive training cases
				LABEL = 1; dir = trainPosDir;
			} else if (t_opts[i] && i==1) { // Get negative training cases
				LABEL = 0; dir = trainNegDir;
			} else if (t_opts[i] && i==2) { // Get positive test samples
				LABEL = 1; dir = testPosDir;
			} else if (t_opts[i] && i==3) {  // Get negative test samples
				LABEL = 0; dir = testNegDir;
			} else continue;
			//dir="/media/femkha/HDB_FemkhaAcer/images/mias-database/";
			//F.addFV(1,string("/media/femkha/HDB_FemkhaAcer/images/full/calc/mdb212.pgm"),0);

			DIR *dp; struct dirent *entry; Mat GI;
		    if( ( dp=opendir(dir.c_str()) ) != NULL ){
					short file_cnt = 0; bool maxx = (MAX_FILE_COUNT>=0)?file_cnt<MAX_FILE_COUNT:true;
					while((entry=readdir(dp)) && maxx ){ //
						// Note: d_type might be incompatible with windows..! use (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) instead
						if(entry->d_type==DT_REG && regex_match (entry->d_name,regex(string(".*\\.(")+EXTS+")",regex_constants::icase))){
							string impath = dir+entry->d_name;
							cout<<impath<<endl;
							if(!L.readImage(impath,GI)){ // Get grey level image
								addFV(file_cnt,impath,GI,LABEL);
								++file_cnt;
								maxx = (MAX_FILE_COUNT>=0)?file_cnt<MAX_FILE_COUNT:true;
							}
						}
					}
					(void) closedir(dp);
				} else {
					perror("Couldn't open the directory.");
		    }
		} // end for
		// Next: store raw, unnormalized, selected individual + cluster features
		//L.serializeMat(s_iDb1,this->i_DB,false); L.serializeMat(s_cDb1,this->c_DB,false);
	}

	// End of function
}
void cFeatureP::saveClassData(bool o_READ)
{
	// Save (o_READ==false) or Restore (o_READ == true) class attributes
	// issue: fix the serialization of filenames (this->db_data.c_file_name)
	Mat out; string impath = DATA_DIR+"db_data.xml";
	if (o_READ) {
		FileStorage fs(impath.c_str(), FileStorage::READ);
		if (fs.isOpened()){
			// retrieve individual database
			fs["i_DB"]>>this->i_DB;
			fs["db_data__i_file_idx"]>>this->db_data.i_file_idx;
			fs["db_data__i_labels"]>>this->db_data.i_labels;
			//fs["db_data__i_file_name"]>>this->db_data.i_file_name;
			// retrieve cluster database
			fs["c_DB"]>>this->c_DB;
			fs["db_data__c_file_idx"]>>this->db_data.c_file_idx;
			fs["db_data__c_labels"]>>this->db_data.c_labels;
			//fs["db_data__c_file_name"]>>this->db_data.c_file_name;
			// retrieve other attributes
			fs["db_status__i_NORMALIZED"]>>this->db_status.i_NORMALIZED;
			fs["db_status__c_NORMALIZED"]>>this->db_status.c_NORMALIZED;
			fs["db_status__SELECTED"]>>this->db_status.SELECTED;
			fs["i_select"]>>this->i_select;
			fs["c_select"]>>this->c_select;
			fs["p_select"]>>this->p_select;
			fs["pca_data__PCA_INIT"]>>this->pca_data.PCA_INIT;
		} else cerr << "Serialization error: failed to open " << impath << endl;
		fs.release();
	} else {
		cv::FileStorage fs(impath.c_str(), cv::FileStorage::WRITE);
		fs << "i_DB" << this->i_DB;
		fs << "db_data__i_file_idx" << this->db_data.i_file_idx;
		fs << "db_data__i_labels" << this->db_data.i_labels;
		fs << "db_data__i_file_name" << this->db_data.i_file_name;
		// store cluster features
		fs << "c_DB" << this->c_DB;
		fs << "db_data__c_file_idx" << this->db_data.c_file_idx;
		fs << "db_data__c_labels" << this->db_data.c_labels;
		fs << "db_data__c_file_name" << this->db_data.c_file_name;
		// store other attributes
		fs << "db_status__i_NORMALIZED" << this->db_status.i_NORMALIZED;
		fs << "db_status__c_NORMALIZED" << this->db_status.c_NORMALIZED;
		fs << "db_status__SELECTED" << this->db_status.SELECTED;
		fs << "i_select" << this->i_select;
		fs << "c_select" << this->c_select;
		fs << "p_select" << this->p_select;
		fs << "pca_data__PCA_INIT" << this->pca_data.PCA_INIT;
		fs.release();
	}
	// End of function
}
void cFeatureP::slave()
{
	// receive job and do computation
	// tstats: 6: slave's master wait lat, 7: bandwidth 8: ldp processing time
	double *Data; vector<Mat> dataQ;
	//vector<hist_type> resQ;
	vector<Mat> resQ;
	char out_opt='b'; // irrelevant
	//myDataType *out_im = new myDataType;
	//hist_type *out_hist;
	//CLdp ldp;
	double time_arr[3]={}; //1:wait latency, 2:average bandwidth unit 4 task set, 3: average process time per image
	double header[SZ_HEADER];

	std::fill(header, header+SZ_HEADER, 0);
	header[4]=myRank;

	while(1){
		int file_cnt=0; double tmp_t; //local variables
		time_arr[0]=time_arr[2]=0; time_arr[1]=0; // record fresh stats for every task set
		tmp_t = MPI_Wtime();
		MPI_Send(header,SZ_HEADER,MPI_DOUBLE,MASTER,TAG_HEADER,MY_COMM_GRP);
		time_arr[0]=MPI_Wtime()-tmp_t; // Time to get masters attention (wait latency)
		MPI_Recv(header,SZ_HEADER,MPI_DOUBLE,MASTER,MPI_ANY_TAG,MY_COMM_GRP,this->Stat);
		if(this->Stat->MPI_TAG == TAG_TERMINATE) break;
		//receive data
		while(true){
			tmp_t = MPI_Wtime();
			Data = new double[(int)(header[1]*header[2])];
			MPI_Recv(Data,(int)(header[1]*header[2]),MPI_DOUBLE,MASTER,TAG_DATA,MY_COMM_GRP,this->Stat);
			Mat_<DBType> d((int)header[1],(int)header[2],Data); // remember to deallocate Data
			//myDataType d; d.data=Data; d.nRows=(int)header[1]; d.nCols=(int)header[2];
			dataQ.push_back(d);
			file_cnt++;
			time_arr[1]+= (MPI_Wtime()-tmp_t); // Bandwidth
			if ((int)header[9]) {
				tmp_t = MPI_Wtime();
				MPI_Recv(header,SZ_HEADER,MPI_DOUBLE,MASTER,MPI_ANY_TAG,MY_COMM_GRP,this->Stat);
				time_arr[0]=MPI_Wtime()-tmp_t; // Time to get masters attention (wait latency)
			}	else break;
		}

		tmp_t = MPI_Wtime();
		while (dataQ.size()) { // process data
			/*out_hist = new hist_type();
			myDataType d = dataQ.back(); dataQ.pop_back();
			//ldp.process(d.data, d.nRows,d.nCols,out_opt,out_im, out_hist);
			resQ.push_back(*out_hist); out_hist=0;
			delete[] d.data; delete[] out_im->data;*/
		}
		time_arr[2] = (MPI_Wtime()-tmp_t); // Total processing time for this

		//time_arr[1] /= file_cnt; time_arr[2] /= file_cnt;
		header[6]=time_arr[0]; header[7]=time_arr[1]; header[8]=time_arr[2];
		header[4]=myRank; header[9]=resQ.size();
		while (!resQ.empty()) { // tuma results
			/*hist_type h = resQ.back(); resQ.pop_back(); header[9]=(int)header[9]-1;
			header[1]=h.h_nHists; header[2]=h.h_binSz;
			MPI_Send(header,SZ_HEADER,MPI_DOUBLE,MASTER,TAG_RESULT,MY_COMM_GRP);
			MPI_Send(h.hist_data,h.h_nHists*h.h_binSz,MPI_DOUBLE,MASTER,TAG_RESULT,MY_COMM_GRP);
			delete[] h.hist_data;*/
			;
		}
	}
	//MPI_Barrier(MY_COMM_GRP);
}
void cFeatureP::updateK(int N, double Y, double to, int pf)
{
	// database size, model latency, bandwidth per original block, files already processed
	opt_k.proc_files=pf; opt_k.sY = Y; opt_k.sto = to;
	//printf("k val: = %d, N: %d, Y: %f, to: %f, SY: %f, Sto: %f, fp: %d\n", opt_k.k, N, Y, to, opt_k.sY, opt_k.sto, opt_k.proc_files);

	if (opt_k.proc_files>=opt_k.p && N>0 && Y>=0 && to>=0) { // Set N as threshold beyond which the optimal k calculation is allowed to take place
		opt_k.N=N;
		double Y=opt_k.sY/opt_k.proc_files, to=opt_k.sto/opt_k.proc_files; // use average values for Y and to
		// if p>=N/K, use method 1
		//printf("opt_k.N: %d, opt_k.Y: %f, opt_k.to: %f, n*y %f\n", opt_k.N,Y,to,opt_k.N*Y);
		if(opt_k.p>=opt_k.N/opt_k.k && to>0){
			opt_k.k = (int)sqrt(opt_k.N*Y/to);
		}else{
			opt_k.k = sqrt(opt_k.N*Y);
		}
		//spamandla nd700147
		// ensure that k is within bounds 0 < k <= N
		if (opt_k.k<=0)
			opt_k.k=1;
		else if (opt_k.k>N)
			opt_k.k=N;

		//opt_k.p = (int)opt_k.N/opt_k.k;
		//printf("updated k: = %d\n", opt_k.k);
	}
}
