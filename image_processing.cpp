//ZaK :-)

#include "image_processing.h"
#include <cmath>
#include <list>


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
	startTic = cv::getTickCount();
	if(!image) return(EINVALID_PARAMETER);	

	cv::Mat grayImage;
	cv::Mat colorImage;

	if(image->channels() > 1) {		// Convert image into gray image
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		colorImage = *image;
	}
	else {
		grayImage = *image;			// Convert gray image into three channel image (color image)
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	if(grayImage.size() != cv::Size()) {
		cv::Mat imgDx;
		cv::Mat imgDy;
		cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);	// 1. order partial derivative in x direction
		cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);	// 1. order partial derivative in y direction

		double threshold1 = 50;		// first threshold of the canny hysteresis
		double threshold2 = 200;	// second threshold
		cv::Mat imgCanny;

		cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);	// Apply canny filter for edge detection
		
		int alphaBins = 1 + floor(1 + 2*M_PI/deltaAlpha);					// Number of alpha bins (accu dimension)
		int imageDiameter = sqrt(imgCanny.rows*imgCanny.rows + imgCanny.cols*imgCanny.cols);		// Calculate diameter of image
		int rhoBins = floor(1 + 2 * imageDiameter / deltaRho);				// Number of rho bins (accu dimension)

		// Create accumulator image
		cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);		// Define empty accumulator image y=rhoBins, x=alphaBins

		// Create vectors to hold row/col and rho/alpha values
		std::vector<std::pair<int, int> > vecRowCol;
		std::vector<std::pair<int, int> > vecRhoAlpha;

		startTicAcc = cv::getTickCount();
		// Fill accumulator
		for(int rows = 0; rows < imgCanny.rows; rows++) {
			for(int cols = 0; cols < imgCanny.cols; cols++) {
				if(imgCanny.at<uint8_t>(rows, cols) != 0) {
					double dx = imgDx.at<int16_t>(rows, cols);
					double dy = imgDy.at<int16_t>(rows, cols);
					double dr = sqrt(dx*dx + dy*dy);							// Calculate norm of the derivative

					double alpha = atan2(dy, dx);								// +- pi
					double rho = cols * dx/dr + rows * dy/dr;					// +- imageDiameter

					double slopeAlpha = 1.0 * (alphaBins - 1) / (M_PI + M_PI);					// Calculate slope to map input to output values
					int indexAlpha = round(slopeAlpha * (alpha + M_PI));						// Apply slope to Alpha
					double slopeRho = 1.0 * (rhoBins - 1) / (imageDiameter + imageDiameter);	// Calculate slope
					int indexRho = round(slopeRho * (rho + imageDiameter));						// Apply slope to Rho

					vecRowCol.push_back(std::vector<std::pair<int, int> >::value_type(rows, cols));
					vecRhoAlpha.push_back(std::vector<std::pair<int, int> >::value_type(indexRho, indexAlpha));

					accImg.at<uint16_t>(indexRho, indexAlpha) += 1;
				}
			}
		}
		endTicAcc = cv::getTickCount();

		// Blur the accumulator image
		cv::Size filterBlur(3, 3);
		cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);

		double threshold = 5;
		cv::Mat binaryImage;
		cv::threshold(accImg, binaryImage, threshold, 255, cv::THRESH_BINARY); // Create binary image
		binaryImage.convertTo(binaryImage, CV_8U);	// Convert CV_16U into CV_8U

		// Morphology
		cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
		cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);	// Apply closing morphology
		
		// Extract regions
		cv::Mat stats, centroids, labelImage;
		connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

		// Color array
		cv::Scalar colors[] = {	cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
								cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)};

		startTicLine = cv::getTickCount();
		for(int i1 = 1; i1 < stats.rows; i1++) {
			double cx = centroids.at<double>(i1, 0);
			double cy = centroids.at<double>(i1, 1);

			int nHoodRho = 15;
			int nHoodAlpha = 15;

			// Generate random color
			// uint8_t red = (rand() % (255 - 0 + 1));
			// uint8_t green = (rand() % (255 - 0 + 1));
			// uint8_t blue = (rand() % (255 - 0 + 1));
			cv::Scalar lineColor = colors[i1 % (sizeof(colors)/sizeof(colors[1]))];	// Take one out of the colors from colors array
			

			for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++) {
				if(fabs(vecRhoAlpha[i0].first - cy) < nHoodRho &&
					fabs(vecRhoAlpha[i0].second - cx) < nHoodAlpha) {
						cv::Point poi = cv::Point(vecRowCol[i0].second,
													vecRowCol[i0].first);

						cv::line(colorImage, poi, poi, lineColor, 3);
					}
			}
		}
		endTicLine = cv::getTickCount();

		// Output
		*m_proc_image[0] = imgDx;
		*m_proc_image[1] = imgCanny;
		*m_proc_image[2] = colorImage;	// save in results
		
	}
	endTic = cv::getTickCount();
	double deltaTimeAcc = (double) (endTicAcc - startTicAcc) / cv::getTickFrequency();
	double deltaTimeLine = (double) (endTicLine - startTicLine) / cv::getTickFrequency();
	double deltaTime = (double) (endTic - startTic) / cv::getTickFrequency();
	
	std::cout << "Acc time: " << (int) (1000*deltaTimeAcc) << " ms" << std::endl;
	std::cout << "Line time: " << (int) (1000*deltaTimeLine) << " ms" << std::endl;
	std::cout << "complete doProcess time: " << (int) (1000*deltaTime) << " ms" << std::endl; 
	std::cout << "\n" << std::endl;
	return(SUCCESS);
}









