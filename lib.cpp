#include "lib.hpp"

void cLib::rescaleIntensity()
{
}
int cLib::readImage(string path, Mat &mat)
{
	// Read image and convert it to our custom type (cMyDataType)
	mat = imread(path.c_str(),0);//CV_LOAD_IMAGE_GRAYSCALE
	if( !mat.data ){
		printf( "No image data \n" );
		return 1;
	}
	return 0;
	//namedWindow("Display Image", WINDOW_AUTOSIZE); imshow("Display Image", m); waitKey(0);
}
void cLib::converter(cMyDataType &im, Mat &mat, int opt)
{
	switch(opt){
		case 1: //cMyDataType => Mat
			mat.create(im.nRows,im.nCols, CV_64FC1);
			for(int i=0;i<im.nRows*im.nCols;++i){
				mat.data[i] = im.data[i];
			}
			break;
		case 2: //Mat => cMyDataType
			im.nRows = mat.rows, im.nCols = mat.cols, im.data = new double[im.nRows*im.nCols];
			for(int i=0;i<mat.rows*mat.cols;++i){
				im.data[i] = mat.data[i];
			}
			break;
		default: ;
	}
}
void cLib::output(const cMyDataType &im)
{
	cout<<"Rows: "<<im.nRows<<", Columns: "<<im.nCols<<endl;
	if(im.data){
		for(int i=0;i<im.nRows;++i){
			for(int j=0;j<im.nCols;++j){
				printf("%10.6g ",im.data[i*im.nCols+j]);
			} printf("\n");
		}
	}
}
void cLib::getMeanStdev(Mat &m, vector<double> &mn, vector<double> &stdev, bool dim1, bool o_removeNanInf)
{
	// Get mean and standard deviation of 2-d matrix m. dim1=true means calculate statistics along rows,
	// dim1=false means calculate along columns. No need to preallocate mn and stdev
	if (dim1) {
		mn.resize(m.rows); stdev.resize(m.rows);
		for (size_t i = 0; i < m.rows; i++) {
			Scalar_<double> imn, istdev;
			Mat cleanRow = m.row(i).reshape(1);
			if(o_removeNanInf)
				cleanRow = removeNanInf<double>(cleanRow,1,1,1);
			meanStdDev(cleanRow,imn,istdev);
			mn[i]=imn[0]; stdev[i]=istdev[0];
		}
	} else {
		mn.resize(m.cols); stdev.resize(m.cols);
		for (size_t i = 0; i < m.cols; i++) {
			Scalar_<double> imn, istdev;
			Mat cleanCol = m.col(i).reshape(1);
			if(o_removeNanInf)
				cleanCol = removeNanInf<double>(cleanCol,1,1,1);
			meanStdDev(cleanCol,imn,istdev);
			mn[i]=imn[0]; stdev[i]=istdev[0];
		}
	}
}
void cLib::getMeanStdev(vector<vector<double>> &v, vector<double> &mn, vector<double> &stdev, bool dim1, bool o_removeNanInf)
{
	// Get mean and standard deviation of 2-d matrix m. dim1=true means calculate statistics along rows,
	// dim1=false means calculate along columns. Overloaded version
	Mat t_Mat; // temporary matrix
	this->Mat2Vec(v,t_Mat,false);
	this->getMeanStdev(t_Mat,mn,stdev,dim1,o_removeNanInf);
}
void cLib::Mat2Vec(vector<vector<double>> &v, Mat &m, bool mat2vec)
{
	// Convert 2-d vector v to 2-d Mat m (vec2mat==true) or otherwise. Don't initialize both containers!
	// v's columns are assumed to be equal
	if (!mat2vec) {
		m = Mat(v.size(),v[0].size(),CV_64FC1);
		for (size_t i = 0; i < v.size(); i++) {
			for (size_t j = 0; j < v[0].size(); j++) {
				m.at<double>(i,j)=v[i][j];
			}
		}
	} else { // convert from m to v
		v.resize(m.rows);
		for (size_t i = 0; i < m.rows; i++) {
			v[i].resize(m.cols);
			for (size_t j = 0; j < m.cols; j++) {
				v[i][j] = ((Mat_<double>) m).at<double>(i,j);
			}
		}
	}
	// end of function
}
void cLib::serializeMat(string impath, Mat &mat, bool o_READ)
{
	// use for serialization to store and retrieve Mat objects. impath should specify full
	// path and filename
	Mat out;
	if (o_READ) {
		FileStorage fs(impath.c_str(), FileStorage::READ);
		if (fs.isOpened()){
			fs["Mat"]>>mat;//fs.root();
		} else cerr << "Serialization error: failed to open " << impath << endl;
		fs.release();
	} else {
		cv::FileStorage fs(impath.c_str(), cv::FileStorage::WRITE);
		fs << "Mat" << mat;
		fs.release();
	}
}
