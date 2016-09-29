#include "mydatatype.hpp"
#include "mcdetect.hpp"
#include "feature.hpp"
#include "featureP.hpp"
#include "lib.hpp"
#include "f_Haralick.hpp"
#include <fstream>

using namespace std;
using namespace cv;


cFeature F;
void tmp();

// todo: remove the contribution of rows containing NaNs from the calculation of mean/stdev values
// check that all hconcat functions append correctly and do not inverse matrices

int main( int argc, char** argv )
{
	//cMyDataType im;
	Mat src, ee;//, dst, dst2, m;
	cLib l;	cFeature f; cFeatureP fp;
	string impath = "/home/femkha/tmp/1.pgm.yml";
	//string impath = "/home/femkha/18b.jpg";
	//l.readImage(impath,src);
	//cHaralick H; H.exec(src);
	Mat C = (Mat_<double>(4,3) << 1, 0, 3, 2, 1, 5, 6, 2, numeric_limits<double>::infinity(), 4,nan(""), 9), D;
	//Mat_<int> Q = (Mat_<int>)C;
	//Mat_<double> Q = (Mat_<double>(4,3) << 1, 0, 3, 2, 1, 5, 6, 2, 1, 4, 5, 9);
	//f.doPCA(C,'c'); Mat coef = f.doPCA(C,'p'); Mat recon = f.doPCA(coef,'r');
	//cout<<C<<" \n\n"<<D<<" \n\n"<<coef<<" \n\n"<<recon<<"\n";

	//f.exec();
	fp.exec();
	//tmp();

	return 0;
}
void tmp()
{
	cLib L;
	//Mat fv = (Mat_<double>(4,3) << 1, 0, 3, 2, 1, 5, 6, 2, numeric_limits<double>::infinity(), 4,nan(""), 9);
	Mat fv = (Mat_<double>(6,4) << 1,2, 2,3, 3,3, 4,5, 5,5, 1,0, 2,1, 3,1, 3,2, 5,3, 6,5, 7,6);
	int labels[4] = {0, 1, 1, 1};
	//float trainingData[4][2] = {{501, 10}, {255, 10}, {501, 255}, {10, 501}};
	//Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
	Mat trainingDataMat = fv;
	//Mat labelsMat = (Mat_<int>(6, 1) << 0, 0, 1, 1, 2, 2);//, 1, 1, 1, 1, 1, 1);
	Mat labelsMat = (Mat_<int>(6, 1) << 0, 0, 0, 2, 2, 2);//, 1, 1, 1, 1, 1, 1);


	//trainingDataMat = repeat(trainingDataMat,2,2);
	//labelsMat = repeat(labelsMat,2,1);

	Mat mean;
	reduce(trainingDataMat, mean, 0, CV_REDUCE_AVG);
	mean.convertTo(mean, CV_64F);

	LDA lda(2);
	lda.compute(trainingDataMat,labelsMat);
	Mat projection = lda.subspaceProject(lda.eigenvectors(), mean, trainingDataMat);

	//std::cout << "projection: "<< lda.eigenvectors()<< std::endl;
	std::cout << trainingDataMat.size()<< " labesl"<<labelsMat.size() << std::endl;

	/*Mat ress;
	reduce(fv,ress,1,CV_REDUCE_MIN); // CV_REDUCE_SUM  CV_REDUCE_AVG  CV_REDUCE_MAX  CV_REDUCE_MIN*/
	//std::cout << " fv = " << trainingDataMat << std::endl;
}

/*

reduce(fv,ress,1,CV_REDUCE_MIN); // CV_REDUCE_SUM  CV_REDUCE_AVG  CV_REDUCE_MAX  CV_REDUCE_MIN

 A Mapping of Type to Numbers in OpenCV

				C1	C2	C3	C4
CV_8U		0		8		16	24
CV_8S		1		9		17	25
CV_16U	2		10	18	26
CV_16S	3		11	19	27
CV_32S	4		12	20	28
CV_32F	5		13	21	29
CV_64F	6		14	22	30
* Unsigned 8bits uchar 0~255
IplImage: IPL_DEPTH_8U
Mat: CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4

Signed 8bits char -128~127
IplImage: IPL_DEPTH_8S
Mat: CV_8SC1，CV_8SC2，CV_8SC3，CV_8SC4

Unsigned 16bits ushort 0~65535
IplImage: IPL_DEPTH_16U
Mat: CV_16UC1，CV_16UC2，CV_16UC3，CV_16UC4

Signed 16bits short -32768~32767
IplImage: IPL_DEPTH_16S
Mat: CV_16SC1，CV_16SC2，CV_16SC3，CV_16SC4

Signed 32bits int -2147483648~2147483647
IplImage: IPL_DEPTH_32S
Mat: CV_32SC1，CV_32SC2，CV_32SC3，CV_32SC4

Float 32bits float -1.18*10-38~3.40*10-38
IplImage: IPL_DEPTH_32F
Mat: CV_32FC1，CV_32FC2，CV_32FC3，CV_32FC4

Double 64bits double
Mat: CV_64FC1，CV_64FC2，CV_64FC3，CV_64FC4

Unsigned 1bit bool
IplImage: IPL_DEPTH_1U
*
* Mat M(2,2, CV_32FC1, Scalar(0)), N, O,P;
* Mat E = Mat::eye(4, 4, CV_64F);
* Mat O = Mat::ones(2, 2, CV_32F);
* Mat Z = Mat::zeros(3,3, CV_8UC1);
*  Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0); // with initialization
*
* OUTPUT
* default => cout << "R (default) = " << endl <<        R           << endl << endl;
* python => cout << "R (python)  = " << endl << format(R,"python") << endl << endl;
* comma separated values => cout << "R (csv)     = " << endl << format(R,"csv"   ) << endl << endl;
* numpy => cout << "R (numpy)   = " << endl << format(R,"numpy" ) << endl << endl;
* C =>  cout << "R (c)       = " << endl << format(R,"C"     ) << endl << endl;
* can only output points, vectors directly
*
* WINDOW_NORMAL WINDOW_AUTOSIZE WINDOW_OPENGL
*
* void normalize(InputArray src, OutputArray dst, double alpha=1, double beta=0, int norm_type=NORM_L2, int dtype=-1, InputArray mask=noArray() )
*
* vector<int> x_ind; for (std::vector<int>::iterator it = x_ind.begin(); it != x_ind.end(); ++it) std::cout << ' ' << *it;
* */
