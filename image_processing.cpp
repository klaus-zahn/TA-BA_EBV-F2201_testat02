

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
	int64 startTic_DoProcess = cv::getTickCount();


	
	if(!image) return(EINVALID_PARAMETER);	

	double pi = 3.1415;
	double threshold1 = 50;
	double threshold2 = 200;
        
	cv::Mat grayImage;
	cv::Mat colorImage;
	cv::Mat imgDx;
	cv::Mat imgDx_skal02;
	cv::Mat imgDy;
	cv::Mat imgCanny;

	//std::cout << "number channels input: "<<image->channels() << std::endl;

	int64 startTic_A1 = cv::getTickCount();
	//Aufgabe 1
    if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_BGR2GRAY ); 
		colorImage = *image;
	} else {
		grayImage = *image; 
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	} 
	int64 endTic_A1 = cv::getTickCount();     

	//std::cout << "number channels processing:: "<<grayImage.channels() << std::endl;


	int64 startTic_A2 = cv::getTickCount();
	/*
	//Aufgabe 2 
	cv::Sobel(grayImage, imgDx, -1, 1, 0, 3, 1, 128);	//satturiert da 8 Bit unsignet 
	//cv::Sobel(grayImage, imgDx_skal02, -1, 1, 0, 3, 0.2, 128);
	cv::Sobel(grayImage, imgDy, -1, 0, 1, 3, 1, 128);
	*/
	int64 endTic_A2 = cv::getTickCount();


	int64 startTic_A3 = cv::getTickCount();
	// Aufgabe 3
	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);
	//cv::Sobel(grayImage, imgDx_skal02, CV_16S, 1, 0, 3, 0.2, 0); // jetzt gleiche Ausgabe
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);
	int64 endTic_A3 = cv::getTickCount();


	int64 startTic_A4 = cv::getTickCount();
	// Aufgabe 4
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);
	int64 endTic_A4 = cv::getTickCount();


	int64 startTic_A5 = cv::getTickCount();
	// Aufgabe 5
	int width = image->size().width;
	int height = image->size().height;
	double deltaAlpha = pi/180*1.5;
	double deltaRho = 1.5;
	double imageDiameter = sqrt(width*width + height*height);
	double nHoodSize = 5;
	

	double rhoBins = floor(1+2*imageDiameter/deltaRho);
	double alphaBins = 1+floor(1+2*pi/deltaAlpha );

	cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);


	std::vector<std::pair<int, int> > vecRowCol;
	std::vector<std::pair<int, int> > vecRhoAlpha;


	for(int rows = 0; rows < imgCanny.rows; rows++) {
		for(int cols = 0; cols < imgCanny.cols; cols++) {
			if(imgCanny.at<uint8_t>(rows, cols) != 0) {
				double dx = imgDx.at<int16_t>(rows, cols);
				double dy = imgDy.at<int16_t>(rows, cols);
				double dr = sqrt(dx*dx+dy*dy);
				
				//std::cout << "da: "<<image->channels() << std::endl;
				double alpha = atan2(dy, dx);
				double rho = cols*dx/dr + rows*dy/dr;
				//... Bestimmung von indexRho, indexAlpha
				double indexAlpha = round((pi+alpha)/deltaAlpha);
				double indexRho = round((imageDiameter+rho)/deltaRho);

				accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;

				vecRowCol.push_back(std::vector<std::pair<int, int> >:: value_type(rows, cols));
				vecRhoAlpha.push_back(std::vector<std::pair <double, double> >::value_type(indexRho, indexAlpha));

			}
		}
	}
	int64 endTic_A5 = cv::getTickCount();


	int64 startTic_A6 = cv::getTickCount();
	// Aufgabe 6
	cv::Size filterBlur(3,3);
	cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);
	cv::Mat binaryImage = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);

	double threshold = 4;
	cv::threshold(accImg, binaryImage, threshold, 255, cv::THRESH_BINARY);

	binaryImage.convertTo(binaryImage, CV_8U);
	int64 endTic_A6 = cv::getTickCount();


	int64 startTic_A7 = cv::getTickCount();
	// Aufgabe 7 
	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

	cv::Mat stats, centroids, labelImage;
	connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);



	for (int i1 = 1; i1 < stats.rows; i1++) {
		double cx = centroids.at<double>(i1, 0);
		double cy = centroids.at<double>(i1, 1);
		
		// Aufagbe 8 
		double nHoodAlpha = nHoodSize;//ZaK pixel basis
		double nHoodRho = nHoodSize;

		int R = floor(0+255/i1);
		int G = floor(0+255/(stats.rows-i1));
		int B = floor(100+155/i1);

		for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++) {
			if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho && fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha) {
				cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first);
				cv::line(colorImage, poi, poi,cv::Scalar(R,G,B), 3);
			}
		}
	}
	int64 endTic_A7 = cv::getTickCount();


	int64 startTic_Out = cv::getTickCount();
	*m_proc_image[0] = imgCanny;
	*m_proc_image[1] = binaryImage;
	*m_proc_image[2] = colorImage;
	int64 endTic_Out = cv::getTickCount();

	int64 endTic_DoProcess = cv::getTickCount();

	double deltaTime_DoProcess = (double) (endTic_DoProcess - startTic_DoProcess)/cv::getTickFrequency();
	double deltaTime_A1 = (double) (endTic_A1 - startTic_A1)/cv::getTickFrequency();
	double deltaTime_A2 = (double) (endTic_A2 - startTic_A2)/cv::getTickFrequency();
	double deltaTime_A3 = (double) (endTic_A3 - startTic_A3)/cv::getTickFrequency();
	double deltaTime_A4 = (double) (endTic_A4 - startTic_A4)/cv::getTickFrequency();
	double deltaTime_A5 = (double) (endTic_A5 - startTic_A5)/cv::getTickFrequency();
	double deltaTime_A6 = (double) (endTic_A6 - startTic_A6)/cv::getTickFrequency();
	double deltaTime_A7 = (double) (endTic_A7 - startTic_A7)/cv::getTickFrequency();
	double deltaTime_Out = (double) (endTic_Out - startTic_Out)/cv::getTickFrequency();

	std::cout << "Process Time: "<<(int) (1000*deltaTime_DoProcess) << " ms" << std::endl;
	std::cout << "Color Shift A1: "<<(int) (1000*deltaTime_A1) << " ms" << std::endl;
	std::cout << "Sobel Filter A2: "<<(int) (1000*deltaTime_A2) << " ms" << std::endl;
	std::cout << "Sobel Filter A3: "<<(int) (1000*deltaTime_A3) << " ms" << std::endl;
	std::cout << "Canny Filter A4: "<<(int) (1000*deltaTime_A4) << " ms" << std::endl;
	std::cout << "Build and store Accumulator A5: "<<(int) (1000*deltaTime_A5) << " ms" << std::endl;
	std::cout << "Process Accumulator A6: "<<(int) (1000*deltaTime_A6) << " ms" << std::endl;
	std::cout << "Extract Acc and drwaw Points: "<<(int) (1000*deltaTime_A7) << " ms" << std::endl;
	std::cout << "Do Outputs: "<<(int) (1000*deltaTime_Out) << " ms" << std::endl;
	
	return(SUCCESS);
}









