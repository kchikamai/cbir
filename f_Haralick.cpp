#include "f_Haralick.hpp"

cHaralick::cHaralick()
{
  // initialize values
  this->max_G = 255; this->min_G = 0;
  //[n_GLCMS][GLCM_MAX][GLCM_MAX];
  this->a_GLCM = new double**[n_GLCMS];
  for (size_t i = 0; i < n_GLCMS; i++) {
    this->a_GLCM[i] = new double*[GLCM_MAX];
    for (size_t j = 0; j < GLCM_MAX; j++) {
      this->a_GLCM[i][j] = new double[GLCM_MAX];
      for (size_t k = 0; k < GLCM_MAX; k++) {
        this->a_GLCM[i][j][k]=0;
      }
    }
  }
}
cHaralick::~cHaralick()
{
  // initialize
  //[n_GLCMS][GLCM_MAX][GLCM_MAX];
  for (size_t i = 0; i < n_GLCMS; i++) {
    for (size_t j = 0; j < GLCM_MAX; j++) {
      delete[] this->a_GLCM[i][j];
    }
    delete[] this->a_GLCM[i];
  }
  delete[] this->a_GLCM;
}
double *cHaralick::exec(Mat &GI)
{
  // #haralick features: Contrast, correlation, energy, homogeneity, Entropy, Max probability x #no of glcms (20 i suppose!)
  this->calculateGLCM(GI);
  this->harFeats();
  return this->h_Feats;
}
void cHaralick::calculateGLCM(Mat &GI)
{
	// Calculate 120 Haralick features for Grey level input image GI
	// Next: Do the necessary padding...

  bool o_Normalize=true, o_Pad=true;

	int g_OFFSETS[n_GLCMS][2] = { // (x,y) x=+D(right)|-D(left), y=+D(bottom)|-D(up)
		{0, 1}, {0, 3}, {0, 5}, {0, 7}, {0, 9},
		{-1, 1}, {-3, 3}, {-5, 5}, {-7, 7}, {-9, 9},
		{-1, 0}, {-3, 0}, {-5, 0}, {-7, 0}, {-9, 0},
		{-1, -1}, {-3, -3}, {-5, -5}, {-7, -7}, {-9, -9}
	};
	int top=0, bottom=0, left=0, right=0;
	top=bottom=round((10-GI.size().height)/2);
	left=right=round((10-GI.size().width)/2);
	// Do padding
  if(o_Pad){
    if(top>0) // covers bottom as well
  		copyMakeBorder( GI, GI, top, bottom, 0, 0, BORDER_CONSTANT, 0 ); // copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
  	if(left>0) // covers right as well
  		copyMakeBorder( GI, GI, 0, 0, left, right, BORDER_CONSTANT, 0 );
  }

	// generate glcm
  //std::fill( &this->a_GLCM[0][0][0], &this->a_GLCM[0][0][0] + sizeof(this->a_GLCM), 0 );
  //memset(this->a_GLCM,0,sizeof(this->a_GLCM)); // initialize a_GLCM to zero
	// Generate GLCMS
	int c_px, n_px;  size_t n_x, n_y; // temp loop vars: current pix, neighbor pix val, x/y coordinates
  this->min_G = this->max_G = (int)GI.at<uchar>(0,0);
	for (size_t i = 0; i < GI.rows; i++) {
		for (size_t j = 0; j < GI.cols; j++) {
      // Get max + min grey levels
      if((int)GI.at<uchar>(i,j)>this->max_G)
        this->max_G = (int)GI.at<uchar>(i,j);
      if((int)GI.at<uchar>(i,j)<this->min_G)
        this->min_G = (int)GI.at<uchar>(i,j);
      // Get GLCM for k'th parameters/offsets
			for(size_t k=0;k<n_GLCMS;++k){
				n_y = i+g_OFFSETS[k][0]; n_x = j+g_OFFSETS[k][1];
        if (n_y>=0 && n_y<GI.rows && n_x>=0 && n_x<GI.cols) {
          c_px=(int)GI.at<uchar>(i,j); n_px=(int)GI.at<uchar>(n_y,n_x);
          ++this->a_GLCM[k][c_px][n_px]; ++this->a_GLCM[k][n_px][c_px]; // symmetric
        }
			}
		}
	}

  // Next: normalize GLCM
  if(o_Normalize){
    for (size_t k = 0; k < n_GLCMS; k++){
      double k_sum=0;
      // First get the total
      for (size_t i = 0; i < GLCM_MAX; i++) {
        for (size_t j = 0; j < GLCM_MAX; j++)
          k_sum += this->a_GLCM[k][i][j];
      }
      // Then convert to probability value
      for (size_t i = 0; i < GLCM_MAX; i++) {
        for (size_t j = 0; j < GLCM_MAX; j++)
          this->a_GLCM[k][i][j]/=k_sum;
      }
    }
  }
  /*
  std::cout << "Image" << std::endl;
  std::cout << GI << std::endl;
  std::cout << "GLCM" << std::endl;
  for (size_t i = 0; i < this->max_G+1; i++) {
    for (size_t j = 0; j < this->max_G+1; j++) {
      std::cout << this->a_GLCM[18][i][j] <<" \t ";
    } std::cout << std::endl;
  }
  std::cout << "Max G: " << this->max_G<< std::endl;*/
	// ************* End of calculateGLCM() **************** //
}
void cHaralick::harFeats()
{
  // Calculate Haralick features for all n_GLCMS glcms, storing them at this->h_Feats
  // Contrast, correlation, energy, homogeneity, Entropy, Max probability
  double contrast[n_GLCMS], correlation[n_GLCMS]={}, energy[n_GLCMS]={}, homogeneity[n_GLCMS]={},
    entropy[n_GLCMS]={}, maxp[n_GLCMS]; // initialize features
  int max_GLCM = this->max_G+1; // can assign this to GLCM_MAX or go with max grey level
  for (int k=0; k<n_GLCMS; k++) { // For every glcm
    double mu_i=0.f, mu_j=0.f, var_i=0.f, var_j=0.f; // intermediate results
    for (int i = 0; i < max_GLCM; i++) { // Get GLCM means
      for (int j = 0; j < max_GLCM; j++) {
        double pij=this->a_GLCM[k][i][j]; // probability at this position
        mu_i += i*pij; mu_j += j*pij; // http://www.fp.ucalgary.ca/mhallbey/glcm_mean.htm
      }
    }
    for (int i = 0; i < max_GLCM; i++) { // Get GLCM variances
      for (int j = 0; j < max_GLCM; j++) {
        double pij=this->a_GLCM[k][i][j]; // probability at this position
        var_i += pij*((i-mu_i)*(i-mu_i)); var_j += pij*((j-mu_j)*(j-mu_j)); // http://www.fp.ucalgary.ca/mhallbey/glcm_variance.htm
      }
    }
    var_i = sqrt(var_i); var_j = sqrt(var_j);
    if(var_i==0 || var_j==0) // assign correlation of 1 since image is uniform.. check link on correlation below!
      correlation[k]=1;

    maxp[k]=this->a_GLCM[k][0][0];
    for (int i = 0; i < max_GLCM; i++) {
      for (int j = 0; j < max_GLCM; j++) {
        double pij=this->a_GLCM[k][i][j]; // probability at this position
        contrast[k] += pij*(i-j)*(i-j); // http://www.fp.ucalgary.ca/mhallbey/contrast.htm
        if(correlation[k]!=1)
          correlation[k] += pij*( ((i-mu_i)*(j-mu_j))/(sqrt(var_i*var_j)) ); // http://www.fp.ucalgary.ca/mhallbey/correlation.htm
        energy[k] += pij*pij; // http://www.fp.ucalgary.ca/mhallbey/asm.htm
        homogeneity[k] += pij / (1+((i-j)*(i-j))); // also "inverse difference moment //http://www.fp.ucalgary.ca/mhallbey/homogeneity.htm"
        if(pij!=0) // http://www.fp.ucalgary.ca/mhallbey/entropy.htm
          entropy[k] += pij*(-1*(log(pij)));
        if(maxp[k]<pij) // http://www.fp.ucalgary.ca/mhallbey/max_probability.htm
          maxp[k]=pij;
      }
    }
    energy[k] = sqrt(energy[k]);
    //std::cout << maxp[k] << " ";
  } // end for all glcms

  for (size_t i=0; i<n_GLCMS; i++) { // Contrast, correlation, energy, homogeneity, Entropy, Max probability
    this->h_Feats[i]=contrast[i];
    this->h_Feats[i+n_GLCMS]=correlation[i];
    this->h_Feats[i+n_GLCMS*2]=energy[i];
    this->h_Feats[i+n_GLCMS*3]=homogeneity[i];
    this->h_Feats[i+n_GLCMS*4]=entropy[i];
    this->h_Feats[i+n_GLCMS*5]=maxp[i];
  }
  /*for (size_t i = 0; i < n_GLCMS*n_h_Feats; i++) {
    std::cout << h_Feats[i]<<" ";
  } std::cout << "" << std::endl;*/
  // ************* End of function **************** //
}
