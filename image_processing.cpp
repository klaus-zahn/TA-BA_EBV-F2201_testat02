//ZaK :-)

#include "image_processing.h"




CImageProcessor::CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
}

CImageProcessor::~CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		delete m_proc_image[i];
	}
}

cv::Mat* CImageProcessor::GetProcImage(uint32 i) {
	if(2 < i) {
		i = 2;
	}
	return m_proc_image[i];
}



int CImageProcessor::DoProcess(cv::Mat* image) {
	int64 startTic = cv::getTickCount();
	
	if(!image) return(EINVALID_PARAMETER);

	if(image->channels() >1){ //enable use of color and Gray Images
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		colorImage = *image;
	} else{
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

double threshold1 = 50;
double threshold2 = 200;

std::srand(time(0)); //seed for random function

cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);
cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);
cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

std::vector<std::pair<int, int> > vecRowCol;
std::vector<std::pair<int, int> > vecRhoAlpha;

double deltaAlpha = M_PI/180*1.5;
double deltaRho = 1.5;
double alphaBins = 1 + floor(1+2*M_PI/deltaAlpha);
double imageDiameter = sqrt(grayImage.rows* grayImage.rows + grayImage.cols* grayImage.cols);
double rhoBins = floor(1+2*imageDiameter/deltaRho);


cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);

for(int rows = 0; rows < imgCanny.rows; rows++){ //moves trough rows 
	for(int cols = 0; cols < imgCanny.cols; cols++){ //moves trough each value in a row
		if(imgCanny.at<uint8_t>(rows, cols) != 0 ){ //is only used if value is greater than 0
			double dx = imgDx.at<int16_t>(rows, cols); //get coresponding dx value
			double dy = imgDy.at<int16_t>(rows, cols); //get coresponding dy value

			double dr = sqrt(dx*dx + dy*dy); //calculate normal vector dr

			double alpha = atan2(dy, dx) + M_PI;

			double rho = cols*dx/dr + rows*dy/dr +imageDiameter;

			double indexRho = (int)rho/deltaRho;
			double indexAlpha = (int)alpha/deltaAlpha;

			accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;

			vecRowCol.push_back(std::vector<std::pair<int, int>>::value_type(rows, cols));
			vecRhoAlpha.push_back(std::vector<std::pair<double, double>>::value_type(indexRho, indexAlpha));

		}
	}
}

cv::Size filterBlur(3,3);
cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);
double threshold = 5;
cv::threshold(accImg, binaryImage, threshold, 255, cv::THRESH_BINARY);
binaryImage.convertTo(binaryImage, CV_8U);

cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

cv::Mat stats, centeroids, labelImage;
connectedComponentsWithStats(binaryImage, labelImage, stats, centeroids);

double nHoodAlpha = 5 * deltaAlpha;
double nHoodRoh = 5 * deltaRho;


for(int i1 = 1; i1 < stats.rows; i1++){
	double cx = centeroids.at<double>(i1, 0);
	double cy = centeroids.at<double>(i1, 1);

	cv::Scalar color{ //generate random colors 
	(double)std::rand() / RAND_MAX * 255,
	(double)std::rand() / RAND_MAX * 255,
	(double)std::rand() / RAND_MAX * 255
	};


	for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++){
		if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRoh && fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha){
			cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first);

			cv::line(colorImage, poi, poi, color, 3);
		}
	}
}


	*m_proc_image[0] = imgDx;
	*m_proc_image[1] = imgCanny;
	*m_proc_image[2] = colorImage;

	int64 endTic = cv::getTickCount();

	double deltaTime = (double) (endTic - startTic) / cv::getTickFrequency();

	std::cout << "time: " << (int) (1000*deltaTime) << " ms" << std::endl;

	return(SUCCESS);
}









