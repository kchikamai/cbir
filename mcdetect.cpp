#include "mcdetect.hpp"

Mat cMcDetect::exec(Mat &in)
{
	Mat M, G, B, mask; double min, max; // median, gaussian, binary, mask
	mask = this->borderProcess(in);
	M = this->medFilt(in);
	G = this->gausFilt(in);
	B = M.mul(G).mul(mask);
	minMaxLoc(in, &min, &max); normalize(B, B, min,max,NORM_MINMAX); // rescale to original intensity range
	this->threshold(B,B);

	B=(B>0);
	B=this->denoise(B);
	//namedWindow("Display Image", WINDOW_NORMAL); imshow("Display Image", B); waitKey(0);
	//namedWindow("Display Image", WINDOW_NORMAL); imshow("Display Image", B); waitKey(0);
	return B;

	// minMaxLoc(in, &min, &max); cB<<"min: "<<min<<" max: "<<max<<endl;
}
Mat cMcDetect::gausFilt(Mat &in)
{
	//void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT )
	Mat out, outArray[3];
	double min, max;  minMaxLoc(in, &min, &max); // get the minimum and maximum intensities in original image for rescaling
	for(int i=5,j=0;i<10;i+=2){ //5,7,9 -- original discards the 3x3
		GaussianBlur(in, outArray[j++], Size(i,i),GAU_SIGMA);
	}
	out = in - (outArray[0]*2/3+outArray[1]/6+outArray[2]/6);

	//void normalize(InputArray src, OutputArray dst, double alpha=1, double beta=0, int norm_type=NORM_L2, int dtype=-1, InputArray mask=noArray() )
	normalize(out, out, min,max,NORM_MINMAX); // rescale to original intensity range
	return out;
}
Mat cMcDetect::medFilt(Mat &in)
{
	//void medianBlur(InputArray src, OutputArray dst, int ksize);
	Mat out, outArray[3];
	double min, max;  minMaxLoc(in, &min, &max); // get the minimum and maximum intensities in original image for rescaling
	for(int i=5,j=0;i<10;i+=2){ //square kernel sizes of 5,7,9
		medianBlur(in, outArray[j++], i);
	}
	out = in - (outArray[0]*2/3+outArray[1]/6+outArray[2]/6);
	normalize(out, out, min,max,NORM_MINMAX); // rescale to original intensity range
	return out;
}
Mat cMcDetect::borderProcess(Mat &in)
{
	/*** in => original image (not filtered)
	 * out =>
	*/
	const short ROI_BORDER_PAD = 8, IM_BORDER_PAD = 10;
	Mat mask; // maskput image
	double t = (cv::mean(in)[0])*3, min, max;

	mask = (in>t); minMaxLoc(in, &min, &max);
	// next: mark border pixels
	cv::Mat se = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(ROI_BORDER_PAD,ROI_BORDER_PAD));
	erode(mask, mask, se);
	mask(Range(0,IM_BORDER_PAD),Range::all())=0; mask(Range::all(),Range(0,IM_BORDER_PAD))=0;
	mask(Range(mask.rows-IM_BORDER_PAD,mask.rows),Range::all())=0; mask(Range::all(),Range(mask.cols-IM_BORDER_PAD,mask.cols))=0;
	mask = (mask/max);

	//namedWindow("out Image", WINDOW_NORMAL); imshow("out Image", out); waitKey(0);
	return mask;
}
void cMcDetect::threshold(Mat &in, Mat &out)
{
	/*** Threshold filtered grey level image 'in' to leave MC like
	objects. 'in' is converted to range [0..255]. Output = grey
	level threshold t in [0..255] that divides 'in' into two classes
	optimally
	* */
	int h_sz=256, hist[h_sz], len_q=10, max_T; // histogram
	double q[len_q]={0.1,0.3,0.5,0.7,0.9,2,3,4.1,4.3,4.4}, max_TE;

	for(int i=0;i<h_sz;++i) hist[i]=0; // initialize histogram
	for(int i=0;i<in.rows*in.cols;++i) {
		hist[(int)(in.data[i])]++;
	}

	bool f_pass = true; // housekeeping variable
	len_q=1;
	for(int i=0;i<len_q;++i){
		for(int t=0;t<h_sz-1;++t){
			Mat pA, pB, npA, npB; int szA=0, szB=0; double sqA, sqB, TE;
			pA = Mat_<double>(1,t+1); pB = Mat_<double>(1,h_sz-(t+1));
			for(int k=0;k<h_sz;k++){
				if(k<t+1) pA.at<double>(szA++)=hist[k];
				else pB.at<double>(szB++)=hist[k];
			}
			npA = sum(pA)[0]; npB = sum(pB)[0];
			pA=pA/npA; pB=pB/npB;
			for(int k=0;k<t+1;k++)
				pA.at<double>(k)=pow(pA.at<double>(k),q[i]);
			for(int k=0;k<h_sz-(t+1);k++)
				pB.at<double>(k)=pow(pB.at<double>(k),q[i]);
			sqA = (1.0/q[i])*(1.0-sum(pA)[0]);
			sqB = (1.0/q[i])*(1.0-sum(pB)[0]);
			TE = sqA + sqB + ( (1-q[i])*sqA*sqB );
			if(f_pass){
				max_T = t; max_TE = TE; f_pass = false;
			}else{
				if(TE>=max_TE){
					max_T=t; max_TE = TE;
				}
			}
		}
	}
	//t_img = wthresh(out,'h',max_T); // threshold by max_T
	Mat mask; double min,max;
	mask = (in>max_T); minMaxLoc(mask, &min, &max); mask /= max; out=mask.mul(in);

	//namedWindow("Display Image", WINDOW_NORMAL); imshow("Display Image", out); waitKey(0);
}
Mat cMcDetect::denoise(const cv::Mat &BI)
{
	// Remove objects that don't meet area+linearity (eccentricity) criteria from
	// BI (BI is assumed to be a binary image)

	cv::Mat out, centroids; Mat_<int> stats, labels;

	int cc_num = connectedComponentsWithStats(BI,labels,stats,centroids);
  vector<Point> cc[cc_num];

	out = Mat::zeros(BI.size(),CV_8UC1);

	for(int i=0;i<labels.rows;++i){
		for(int j=0;j<labels.cols;++j){
			cc[labels(i,j)].push_back(Point(j,i)); // remember point(col, row)!
		}
	}

	//cout<<"all objects: "<<cc_num<<endl;
	Region r[cc_num];
	int *x_ind=new int[cc_num]; x_ind[0]=0; // set background to zero
	for(int i=1;i<cc_num;++i){ // ensures background is not checked
		// denoise by area if specified by o_prep
		x_ind[i]=1;
		if(stats(i,CC_STAT_AREA)<t_MIN_AREA || stats(i,CC_STAT_AREA)>t_MAX_AREA){
			 x_ind[i]=0; continue; // continue to prevent expensive calculation of unqualified object
		}
		RegionProps regionProps(cc[i], BI); r[i]=regionProps.getRegion();
		// denoise by linearity
		if (r[i].Eccentricity()>=t_Eccentricity)
			x_ind[i]=0;
	}
	for(int i=0;i<labels.rows;++i){ // nullify small objects from labels
		for(int j=0;j<labels.cols;++j){
			if(int(x_ind[labels(i,j)])!=0){
				out.at<uchar>(i,j)=255;
			}
		}
	}
	//	namedWindow("Contours",CV_WINDOW_AUTOSIZE); imshow("Contours", out ); cv::waitKey(); // throws a seg fault
	// CC_STAT_LEFT CC_STAT_TOP CC_STAT_WIDTH CC_STAT_HEIGHT CC_STAT_AREA CC_STAT_MAX
	return out;
}

void cMcDetect::anotherfunc(cv::Mat &img)
{
	std::vector<std::vector<cv::Point> > contours;

	//cv::Mat gray;
  //cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::Mat gray = Mat::zeros(img.size(), CV_8UC3), gray2;

	for (size_t i = 150; i < 450; i++) {
		for (size_t j = 150; j < 450; j++) {
			img.at<uchar>(i,j)=255;
		}
	}

  cv::findContours(img, contours, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	printf("contour: %ld\n", contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::drawContours(gray, contours, i, cv::Scalar(0, 255, 0), 0, 8);
	}

	//gray2 = gray-img;
	//RegionProps regionProps(cents, img); Region r = regionProps.getRegion();
	//printf("Area: %f = %f\n", r.Area(),r.Eccentricity());
	namedWindow("out Image", WINDOW_NORMAL); imshow("out Image", gray); waitKey(0);
}
	/*

	uint8_t* pixelPtr = (uint8_t*)foo.data;
	bgrPixel.val[2] = pixelPtr[i*foo.cols*cn + j*cn + 2]; // R

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	    // down-scale and upscale the image to filter out the noise
	    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
	    pyrUp(pyr, timg, image.size());
	    vector<vector<Point> > contours;
			... from http://answers.opencv.org/question/58466/regionprops-vs-findcontours/
	*/
	// -----------------------------------------------------------------------------
