//ZaK :-)

#include "image_processing.h"
#include <math.h>

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
	
	if(!image) return(EINVALID_PARAMETER);	

	int64_t endTick = 0, startTick = cv::getTickCount();
	double deltaT = 0.0;

	// Prepare grayscale image for processing
	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		colorImage = *image;
	} else {
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	endTick = cv::getTickCount();
	deltaT = (double) (endTick - startTick) / cv::getTickFrequency();
	std::cout << "convert image: " << (int)(1000*deltaT) << " ms\n";
	startTick = cv::getTickCount();

	// a) Partial derivatives
	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0);
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1);
	*m_proc_image[0] = imgDx;

	endTick = cv::getTickCount();
	deltaT = (double) (endTick - startTick) / cv::getTickFrequency();
	std::cout << "sobel: " << (int)(1000*deltaT) << " ms\n";
	startTick = cv::getTickCount();

	// b) Detect edges using canny edge detection
	double threshold1 = 50;
	double threshold2 = 200;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);
	*m_proc_image[1] = imgCanny;

	endTick = cv::getTickCount();
	deltaT = (double) (endTick - startTick) / cv::getTickFrequency();
	std::cout << "canny edge detection: " << (int)(1000*deltaT) << " ms\n";
	startTick = cv::getTickCount();

	// c) Calculate alpha and rho for each pixel in an edge
	// Bin size and count (source: example solution from testat 1)
	double deltaAlpha = M_PI/180.0*1.5;
	double deltaRho = 1.5;
	double imageDiameter = sqrt(grayImage.cols * grayImage.cols + grayImage.rows * grayImage.rows);
	size_t alphaBins = 1+floor(1+2*M_PI/deltaAlpha);
	size_t rhoBins = floor(1+2*imageDiameter/deltaRho);
	// Generate accumulator image
	accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);
    std::vector<std::pair<int, int>> vecRowCol;
    std::vector<std::pair<int, int>> vecRhoAlpha;
    for(int rows = 0; rows < imgCanny.rows; rows++) {
		for(int cols = 0; cols < imgCanny.cols; cols++) {
			if(imgCanny.at<uint8_t>(rows, cols) == 0) continue;
			
			double dx = imgDx.at<int16_t>(rows, cols);
			double dy = imgDy.at<int16_t>(rows, cols);
			double dr = sqrt(dx * dx + dy * dy);

			double alpha = atan2(dy, dx);
			double rho = cols * dx / dr + rows * dy / dr;
			size_t indexRho = floor(0.5 + (imageDiameter + rho) / deltaRho);
			size_t indexAlpha = floor(0.5 + (M_PI+alpha)/deltaAlpha);

			accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;
			vecRowCol.push_back(std::vector<std::pair<int, int>>::value_type(rows, cols));
			vecRhoAlpha.push_back(std::vector<std::pair<double, double>>::value_type(indexRho, indexAlpha));
		}
	}

	endTick = cv::getTickCount();
	deltaT = (double) (endTick - startTick) / cv::getTickFrequency();
	std::cout << "hough accumulator: " << (int)(1000*deltaT) << " ms\n";
	startTick = cv::getTickCount();

	// d) e) Process accumulator and draw pixels that correspond to peaks
	cv::Size filterBlur(3,3);
	cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);
	double threshold = 3;
	cv::threshold(accImg, binaryImage, threshold, 255, cv::THRESH_BINARY);
	binaryImage.convertTo(binaryImage, CV_8U);
	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

	endTick = cv::getTickCount();
	deltaT = (double) (endTick - startTick) / cv::getTickFrequency();
	std::cout << "accumulator processing: " << (int)(1000*deltaT) << " ms\n";
	startTick = cv::getTickCount();

	const cv::Scalar colors[] = {
		cv::Scalar(255,	0,	 0),
		cv::Scalar(0,	255, 0),
		cv::Scalar(0,	0,	 255),
		cv::Scalar(255,	255, 0),
		cv::Scalar(0,	255, 255),
		cv::Scalar(255,	0,	 255),
		cv::Scalar(255, 255, 255)
	};
	size_t colorCount = sizeof(colors) / sizeof(colors[0]);

	cv::Mat stats, centroids, labelImage;
	cv::connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
	int lineCount = (stats.rows > (int)colorCount ? colorCount : stats.rows);
	for (int i1 = 1; i1 <= lineCount; i1++) {
		double alpha = centroids.at<double>(i1, 0);
		double rho = centroids.at<double>(i1, 1);

		for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++) {
			size_t nHoodRho = 5, nHoodAlpha = 5;
			if(fabs(vecRhoAlpha[i0].first-rho) < nHoodRho && fabs(vecRhoAlpha[i0].second-alpha) < nHoodAlpha) {
				cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first);
				cv::line(colorImage, poi, poi, colors[i1-1], 3);
			}
		}
	}
	endTick = cv::getTickCount();
	deltaT = (double) (endTick - startTick) / cv::getTickFrequency();
	std::cout << "draw pixels: " << (int)(1000*deltaT) << " ms\n" << std::endl;
	startTick = cv::getTickCount();
	*m_proc_image[2] = colorImage;

	return(SUCCESS);
}
