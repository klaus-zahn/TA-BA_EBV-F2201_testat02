//ZaK :-)

// EBV Testat 02 - Joel Koch
// 26.05.2022
// Hough Transformation

#include "image_processing.h"
#include <chrono>
#include <iostream>
#include <unistd.h>

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
        
	//Variables and names
	cv::Mat grayImage;
	cv::Mat colorImage;
	cv::Mat binaryImage;
	cv::Mat imgDx;
	cv::Mat imgDy;
	cv::Mat imgCanny;
	double threshold_low = 20;
	double threshold_high= 200;
	double rhoBins = 0.1;
	double alphaBins = 1;
	double threshold = 5;
	double indexAlpha;
	double indexRho;

	std::vector<std::pair<int, int> > vecRowCol;
	std::vector<std::pair<int, int> > vecRhoAlpha;

	//Color definition
	cv::Scalar Colos[8]={
		cv::Scalar(255,0,0),
		cv::Scalar(0,255,0),
		cv::Scalar(0,0,255),
		cv::Scalar(255,255,0),
		cv::Scalar(0,255,255),
		cv::Scalar(126,142,47),
		cv::Scalar(125,125,255),
		cv::Scalar(64,0,10)
	};

	if (image->channels() > 1){
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		colorImage = *image;
	}
	else {
		grayImage = *image;
		cv::cvtColor (*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	if(mPrevImage.size() != cv::Size()){
		double imagediam = sqrt((grayImage.cols*grayImage.cols)+(grayImage.rows*grayImage.rows)); //get image diagonal
		auto start = std::chrono::steady_clock::now();

		//--1a.)
		cv::Sobel(grayImage, imgDx, CV_16S, 1, 0);		
		cv::Sobel(grayImage, imgDy, CV_16S, 0, 1);	

		//--1g.)
		auto sobel = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed = sobel-start;
		std::cout << "Time for Sobel: " << (elapsed.count()*1000) << "ms\n";

		start = std::chrono::steady_clock::now();

		//--1b.)
		cv::Canny(grayImage, imgCanny, threshold_low, threshold_high, 3, true);	

		//--1g.)
		auto edgecanny = std::chrono::steady_clock::now();
		elapsed = edgecanny-start;
		std::cout << "Time for Edge Canny: " << (elapsed.count()*1000) << "ms\n";

		//--1c.)
		alphaBins = 1+int((1+2*CV_PI/(CV_PI/180*1.5)));
		rhoBins = int(1+2*imagediam/1.5);

		cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);		

		start = std::chrono::steady_clock::now();
		for(int rows = 0; rows < imgCanny.rows; rows++){					
			for(int cols = 0; cols < imgCanny.cols; cols++){
				if(imgCanny.at<uint8_t>(rows,cols)!= 0){
					double dx = imgDx.at<int16_t>(rows, cols);
					double dy = imgDy.at<int16_t>(rows, cols);

					double alpha = atan2(dy,dx)*180/CV_PI;
					double rho = cols*dx/(sqrt(dx*dx+dy*dy))+rows*dy/(sqrt(dx*dx+dy*dy));

					indexAlpha = (180+alpha)/1.5;
					indexRho = (imagediam+rho)/1.5;

					accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;

					vecRowCol.push_back(std::vector<std::pair<int,int>>::value_type(rows, cols));
					vecRhoAlpha.push_back(std::vector<std::pair<double, double> >::value_type(indexRho, indexAlpha));
				}
			}
		}

		//--1g.)
		auto AccumulatorImage = std::chrono::steady_clock::now();
		elapsed = AccumulatorImage-start;
		std::cout << "Time for AccumulatorImage: " << (elapsed.count()*1000) << "ms\n";

		cv::Size filterBlur(3,3);
		cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);		

		cv::threshold(accImg, binaryImage, threshold,255,cv::THRESH_BINARY);	

		binaryImage.convertTo(binaryImage, CV_8U);

		cv::Mat kernel = cv::Mat::ones(3,3,CV_8UC1);
		cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

		cv::Mat stats, centriods, labelImage;
		connectedComponentsWithStats(binaryImage, labelImage, stats, centriods);

		double nHoodRho = 	4;
		double nHoodAlpha = 4;
		start = std::chrono::steady_clock::now();
		for( int i1 = 1; i1 < stats.rows; i1++){
			
			double cx = centriods.at<double>(i1, 0);
			double cy = centriods.at<double>(i1,1);

			for(uint i0 = 0; i0 < vecRhoAlpha.size();i0++){
				if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho && fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha){
					cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first);
					cv::line(colorImage, poi, poi, Colos[i1%8],3);
				}
			}
		}
		
		//--1g.)
		auto LineDrawing = std::chrono::steady_clock::now();
		elapsed = LineDrawing-start;
		std::cout << "Time for Line Drawing: " << (elapsed.count()*1000) << "ms\n";

		//--1f.)
		*m_proc_image[0] = imgDx;
		*m_proc_image[1] = imgCanny;
		*m_proc_image[2] = colorImage;
		}
	mPrevImage = grayImage.clone();
	
	return(SUCCESS);
}









