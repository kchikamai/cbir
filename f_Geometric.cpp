#include "f_Geometric.hpp"
#include "f_Haralick.hpp"

vector<vector<vector<double>>> cGeometric::exec(Mat &BI, Mat &GI)
{
   return this->getFeats(BI,GI,true);
}
vector<vector<vector<double>>> cGeometric::getFeats(const cv::Mat &BI, cv::Mat &GI, bool o_getclus)
{
	// calculate cluster and individual mc shape features from BI (BI is assumed to be a binary image)
	// and store in feats. First are individual features (index=0), then cluster features (index1..n) where
	// n is the number of true clusters. The following features are stored
	// Area, Area (blob pixel count), Compactness, Orientation, Eccentricity, Solidity
	// Also calculates Haralick features from (*GI)
  vector<vector<vector<double>>> feats; // N x 144
  feats.resize(2); // 0: individual features, 1: cluster features
	cv::Mat BI_centroids; Mat_<int> stats, labels;
	int cc_num = connectedComponentsWithStats(BI,labels,stats,BI_centroids);
  cLib L;
	// cc_num includes background i.e. cc_num = <<1(background) + num of objects>>, so calculations will omit this
	--cc_num; // omit background. Remember to point indices (to ((*BI),labels,stats,BI_centroids)) from 1 onwards!
	if (cc_num) { // at least one object
		vector<Point> cc[cc_num]; // array of mcs. cc[i] refers to i'th mc (len(mc[i])=no. of pixels of mc)
		// Next: extract connected components into cc
		for(int i=0;i<labels.rows;++i){
			for(int j=0;j<labels.cols;++j){
				if(labels(i,j)) // don't include background (labels(i,j)==0)
					cc[labels(i,j)-1].push_back(Point(j,i)); // decrement index bcoz of background effect! Also remember point(col, row)!
			}
		}

    // Next: calculate geometric features
    this->shapeStats(GI,BI,feats[0]); // store individual results
		if(o_getclus && cc_num>1){
			vector<Point> cents; // BI_centroids for all mc objects. Size=1..cc_num
			for(int i=0;i<cc_num;++i) // Retrieve mc BI_centroids. remember point(col, row)!
				cents.push_back(Point(BI_centroids.at<double>(i+1,0),BI_centroids.at<double>(i+1,1)));

			int *clus=distMatrix(cents); std::vector<size_t> t_clus; // cluster indices for mcs, true clusters
      int dist_crit[cents.size()]; std::fill(dist_crit,dist_crit+cents.size(),0);
			// Add only true clusters (i.e. #objects >= t_CLUS_SZ)
			for (size_t i = 0; i < cents.size(); i++){ // get cluster sizes
				if(clus[i]){ // check if mc is potential cluster
					if(t_CLUS_SZ == ++dist_crit[clus[i]]) // enough mcs to form cluster?
						t_clus.push_back(clus[i]); // true cluster - add to list
				}
			}
			//namedWindow("og",CV_WINDOW_AUTOSIZE); imshow("og", (*BI) ); cv::waitKey();

      feats[1].resize(t_clus.size()); // set rows to number of clusters.
      vector<vector<double>> t_vec; vector<double> mn,stdev; // temporary variables
      for (size_t i = 0; i < t_clus.size(); i++) {
        feats[1][i].resize(DIM_CLUS_FEAT); // no of features
        size_t c_idx = t_clus[i], c_sz=dist_crit[clus[i]]; // get index and size of true cluster
				// Next: create binary image 't_BI' containing all mcs in cluster 'c_idx'
				cv::Mat t_BI = Mat::zeros(BI.size(),CV_8UC1); // Binary image for current cluster and its objects
				std::vector<cv::Point> t_clus_mc, t_mc_cent; // cluster's mcs and cluster centroid
				for (size_t j = 0; j < cents.size(); j++) { // Get (into t_clus_mc) mcs belonging to current cluster
					if (c_idx==clus[j]) {
						for (size_t k = 0; k < cc[j].size(); k++) {
							t_BI.at<uchar>(cc[j][k])=255;
							t_clus_mc.push_back(cc[j][k]);
						}
						t_mc_cent.push_back(Point(BI_centroids.at<double>(j,0),BI_centroids.at<double>(j,1))); // mc centroid
					}
				}
				//namedWindow("cluster",CV_WINDOW_AUTOSIZE); imshow("cluster", t_BI ); cv::waitKey();
        this->shapeStats(GI,t_BI, t_vec); // Shape feats for mcs in cluster
        //std::cout << "++++ Cluster features ++++++" << std::endl;
				// NExt: Add 10 features (0..9) - Mean + standard deviation (Area, Compactness, Orientation, Eccentricity, Solidity)
        L.getMeanStdev(t_vec, mn, stdev, false);
        for (size_t j = 0; j < N_FEATS-1; j++) {
          feats[1][i][j*2] = mn[j]; feats[1][i][j*2+1] = stdev[j]; // j+1 => leave out area(0)
				}

        //cin.ignore();
				// Get convex hull representing the cluster region
        vector<std::vector<cv::Point>> hull(1); Mat ch_t_BI=cv::Mat::zeros(t_BI.size(), CV_8UC1);
        cv::convexHull(cv::Mat(t_clus_mc).reshape(2), hull[0]);
			 	drawContours( ch_t_BI, hull, -1, Scalar(255), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
				// Get cluster mcs' shape features
				Mat tBI_centroids; Mat_<int> tBI_stats, tBI_labels;
				connectedComponentsWithStats(ch_t_BI,tBI_labels,tBI_stats,tBI_centroids); // there should just be one object here..
        t_vec.clear();
        this->shapeStats(GI,ch_t_BI,t_vec); // Get Cluster only shape features
        // Add to main feature vector: [A C O E S Ds mean(D) std(D) mean(Dm) std(Dm) length(idx)]; % 11 features
				for (size_t j = 0; j < N_FEATS-1; j++) // Add clusters A C O E S to main feature vector from feature 11..15
					feats[1][i][10+j]=t_vec[0][j];
        feats[1][i][16] = t_mc_cent.size()/t_vec[0][1]; // Add density to vector. Density = (mcs in cluster)/cluster Area
        // Next: Get cluster mcs to cluster-centroid distances into cents2, and stats of the same into mn and stdev (overwrites the two!)
        Mat cents2(1,t_mc_cent.size(),CV_64FC1);
				for (size_t j = 0; j < t_mc_cent.size(); j++) {
					// remember tBI_centroids[0] is the background! t_mc_cent has been stripped of background already..
					cents2.at<double>(j) = sqrt(pow(t_mc_cent[j].x-tBI_centroids.at<double>(1,0),2)+pow(t_mc_cent[j].y-tBI_centroids.at<double>(1,1),2));
				}
        mn.clear(); stdev.clear();
        meanStdDev(cents2,mn,stdev); // mean/std dev for mc-to-cluster_centroid distance
				feats[1][i][17] = mn[0]; feats[1][i][18] = stdev[0]; // Add stats on mc-to-centroid distance
				vector<float> *dm = new vector<float>; distMatrix(t_mc_cent,dm); //t_mc_cent
        mn.clear(); stdev.clear();
        meanStdDev(Mat(*dm).reshape(1),mn,stdev); // mean/std dev for mc-to-mc distances
        feats[1][i][18] = mn[0]; feats[1][i][19] = stdev[0]; // Add stats on mc-to-mc distance
				feats[1][i][20] = t_mc_cent.size(); // number of mcs in cluster
        dm=0;
        // Next: Get Haralick features
        cHaralick H;
        // Next: Replicate GI using binary image BI as mask
        Mat nGI = Mat::zeros(GI.size(), CV_8UC1);
        for (size_t k = 0; k < GI.rows; k++) { // Convert t_BI to (*GI)'s grey level using t_BI as mask
          for (size_t j = 0; j < GI.cols; j++) {
            if(ch_t_BI.at<uchar>(k,j)>0)
              nGI.at<uchar>(k,j)=GI.at<uchar>(k,j);
          }
        }
        double * h_feats = H.exec(nGI); // Calculate Haralick features
        for (size_t j = 0; j < n_GLCMS*n_h_Feats; j++) { //
          feats[1][i][j+21] = h_feats[j];
        }
        h_feats = 0;
        //namedWindow( "Source", WINDOW_NORMAL ); 	imshow( "Source", t_BI ); cv::waitKey();

        // Next: add cluster centroids and diameter. % these just serve to identify cluster n are not real features
				feats[1][i][141] = tBI_centroids.at<double>(1,0); feats[1][i][142] = tBI_centroids.at<double>(1,1); // move to end of feature vector
				feats[1][i][143] = max(tBI_stats(1,CC_STAT_WIDTH),tBI_stats(1,CC_STAT_HEIGHT));
				//for (size_t j = 0; j <DIM_CLUS_FEAT; j++) cout << feats[1][i][j] << " ";	cout<<"\n"<<endl;/**/

				//std::cout << "+++++++++++++++++++++++++" << std::endl;
			 	//namedWindow( "Source", WINDOW_NORMAL ); 	imshow( "Source", t_BI ); cv::waitKey();
			}
      //delete[] clus;
		} // #endif(o_getclus && cc_num>1)
	} // #endif (cc_num)
	// CC_STAT_LEFT CC_STAT_TOP CC_STAT_WIDTH CC_STAT_HEIGHT CC_STAT_AREA CC_STAT_MAX

	return feats;
std::cout << "c" << std::endl;
}
void cGeometric::cvh(const cv::Mat &src_gray)
{
	cv::Mat_<int> points; Mat_<uchar> out_im= cv::Mat::zeros(src_gray.size(), CV_8UC1);
	vector<std::vector<cv::Point>> hull(1);
	for (int x = 0; x < src_gray.cols; x++)
    for (int y = 0; y < src_gray.rows; y++)
			if(src_gray.at<uchar>(y,x)){
        points.push_back(cv::Point(x, y));
			}
  cv::convexHull(cv::Mat(points).reshape(2), hull[0]);
	drawContours( out_im, hull, -1, Scalar(255), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
	namedWindow( "Source", WINDOW_NORMAL ); 	imshow( "Source", out_im );
	/*
	// Show in a window
	namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );	imshow( "Hull demo", drawing );
	*/
	waitKey(0);
}
void cGeometric::shapeStats(const cv::Mat &GI, const cv::Mat &BI, vector<vector<double>> &vec)
{
	// calculate shape features from BI (BI is assumed to be a binary image)
	// Area, Area (blob pixel count), Compactness, Orientation, Eccentricity, Solidity
  const int padVal = ceil(5.f/2); // pad microcalcification region by this value on each side
  size_t vec_idx=vec.size();
	cv::Mat_<double> centroids; Mat_<int> stats, labels;
	int cc_num = connectedComponentsWithStats(BI,labels,stats,centroids);
	// cc_num includes background i.e. cc_num = <<1(background) + num of objects>>, so calculations will omit this
	--cc_num; // omit background. Remember to point indices (to (BI,labels,stats,centroids)) from 1 onwards!
	if (cc_num) { // at least one object
    vec.resize(vec_idx+cc_num); // set number of rows = number of calcifications
    vector<Point> cc[cc_num];
    // Next: extract connected components into cc
		for(int i=0;i<labels.rows;++i){
			for(int j=0;j<labels.cols;++j){
				if(labels(i,j)) // don't include background (labels(i,j)==0)
					cc[labels(i,j)-1].push_back(Point(j,i)); // decrement index bcoz of background effect! Also remember point(col, row)!
			}
		}

    // Next: Replicate GI using binary image BI as mask
    Mat nGI = Mat::zeros(GI.size(), CV_8UC1);
    for (size_t k = 0; k < GI.rows; k++) { // Convert t_BI to (*GI)'s grey level using t_BI as mask
      for (size_t j = 0; j < GI.cols; j++) {
        if(BI.at<uchar>(k,j)>0)
          nGI.at<uchar>(k,j)=GI.at<uchar>(k,j);
      }
    }
    cHaralick H;
		// Next: calculate geometric features
		for(int j=0, i=vec_idx;i<vec_idx+cc_num;++i,++j){
			RegionProps regionProps(cc[j], BI);
      Region r=regionProps.getRegion();

      vec[i].resize(IND_CLUS_FEAT); // 6 features

			//vec[i][0] = fabs(r->Area()); // ignore opencv's version of area (is in decimals)
			vec[i][0] = stats(i+1,CC_STAT_AREA);
			vec[i][1] = fabs(r.Area())/(r.BoundingBox().width*r.BoundingBox().height);
			vec[i][2] = r.Orientation();
			vec[i][3] = r.Eccentricity();
			vec[i][4] = r.Solidity();
			//for(int j=0;j<N_FEATS;++j)	printf("%f ", vec[i][j]);	printf("\n");
      Mat ROI = nGI(r.BoundingBox()); ROI.adjustROI(padVal,padVal,padVal,padVal);
      double *h_feats = H.exec(ROI);
      for (size_t j = 0; j < n_GLCMS*n_h_Feats; j++) {
        vec[i][j+5] = h_feats[j];
      }
      vec[i][125]= r.BoundingBox().x; // padding value not considered!
      vec[i][126]= r.BoundingBox().y;
      vec[i][127]= r.BoundingBox().width;
      vec[i][128]= r.BoundingBox().height;
      h_feats=0;
		}
	} // #endif (cc_num)
	// CC_STAT_LEFT CC_STAT_TOP CC_STAT_WIDTH CC_STAT_HEIGHT CC_STAT_AREA CC_STAT_MAX
}
int * cGeometric::distMatrix(vector<Point> &centroids,vector<float> *dm)
{
	// Label mcs by their cluster indices. 0 means mc doesn't belong to any cluster
	// Output container clus_idx[i] contains the cluster index for an mc whose centroid is indexed by i
	// initialize dm if it's desired to store the distance values

	vector<vector<int> > clusters;
	int N = centroids.size(), idx=0; int *clus_idx=new int[N];
	std::fill(clus_idx,clus_idx+N,0);
	//cout<<"size:"<<centroids.size()<<"\n";	//cout<<centroids[62]<<endl;
	for(int i=0;i<N-1;++i){
		Point L = centroids[i];
		for(int j=i+1;j<N;++j){
			// find distance between element i and j
			Point R = centroids[j];
			float dist = sqrt(pow(L.x-R.x,2)+pow(L.y-R.y,2));
			if(dm) dm->push_back(dist); // keep value if desired by caller
			// if distance within threshold, flag element i and j
			if (dist<=t_CLUS_DIST) {
				if (clus_idx[i]) {
					clus_idx[j]=clus_idx[i];
				} else {
					clus_idx[j]=clus_idx[i]=++idx;
				}
			}
		}
	}

	return clus_idx;

	//namedWindow("out Image", WINDOW_NORMAL); imshow("out Image", drawing); waitKey(0);
}
