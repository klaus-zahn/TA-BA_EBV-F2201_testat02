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
	
	if(!image) return(EINVALID_PARAMETER);	
        
        
      //  cv::imwrite("dx.png", *m_proc_image[0]);
      //  cv::imwrite("dy.png", *m_proc_image[1]);

	//Teil 1
	int64 startTic = cv::getTickCount();

	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY ); 
		colorImage = *image;
	} else {
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	//Teil 2
	cv::Mat imgDx;
	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);
	cv::Mat imgDy;
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);

	int64 endTic_a = cv::getTickCount();
	double deltaTime = (double) (endTic_a - startTic)/cv::getTickFrequency();
	std::cout << "time for step a:" << (int) (1000*deltaTime) << " ms" << std::endl;


	//Teil4
	double threshold1 = 50;
	double threshold2 = 200;
	cv::Mat imgCanny;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);


	int64 endTic_b = cv::getTickCount();
	deltaTime = (double) (endTic_b - endTic_a)/cv::getTickFrequency();
	std::cout << "time for step b:" << (int) (1000*deltaTime) << " ms" << std::endl;


	//Teil5
	double deltaAlpha = M_PI/180*1.5;
	double deltaRho = 1.5;
	double alphaBins = 1+floor(1+2*M_PI/deltaAlpha);
	double imageDiameter = sqrt(grayImage.rows* grayImage.rows + grayImage.cols*grayImage.cols);
	double rhoBins = floor(1+2*imageDiameter/deltaRho);
	cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);

	std::vector<std::pair<int, int> > vecRowCol;
	std::vector<std::pair<int, int> > vecRhoAlpha;

	for(int rows = 0; rows < imgCanny.rows; rows++) {
		 for(int cols = 0; cols < imgCanny.cols; cols++) {
			if(imgCanny.at<uint8_t>(rows, cols) != 0) { 
				double dx = imgDx.at<int16_t>(rows, cols);
				double dy = imgDy.at<int16_t>(rows, cols);
				double dr = sqrt(dx*dx + dy*dy);
				double alpha = atan2(dy, dx) + M_PI;
				double rho = cols*dx/dr + rows*dy/dr + imageDiameter;
				int indexAlpha = (int)alpha/deltaAlpha;
				int indexRho = (int)rho/deltaRho;
				if (0<= indexRho && indexRho < rhoBins && 0<= indexAlpha && indexAlpha < alphaBins)
				{
					accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;
					vecRowCol.push_back(std::vector<std::pair<int, int> >:: value_type(rows, cols));
					vecRhoAlpha.push_back(std::vector<std::pair<double, double> >::value_type(indexRho, indexAlpha));
				}else	{
					int i = 0;
				}	
			} 
		}
	}


	int64 endTic_c = cv::getTickCount();
	deltaTime = (double) (endTic_c - endTic_b)/cv::getTickFrequency();
	std::cout << "time for step c:" << (int) (1000*deltaTime) << " ms" << std::endl;

	//Teil6
	cv::Size filterBlur(3,3);
	cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);
	double threshold = 5;
	cv::Mat binaryImage;
	cv::threshold(accImg, binaryImage, threshold, 255,cv::THRESH_BINARY);
	binaryImage.convertTo(binaryImage, CV_8U);


	//Teil 7
	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

	cv::Mat stats, centroids, labelImage; 
	connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);


	int64 endTic_d = cv::getTickCount();
	deltaTime = (double) (endTic_d - endTic_c)/cv::getTickFrequency();
	std::cout << "time for step d:" << (int) (1000*deltaTime) << " ms" << std::endl;


	double nHoodRho = 5 * deltaRho;
	double nHoodAlpha = 5 * deltaAlpha;

	cv::Scalar colors[] = {cv::Scalar(255,0,0), cv::Scalar(0,255,0), cv::Scalar(0,0,255),cv::Scalar(0,255,255),cv::Scalar(255,0,255),
		cv::Scalar(255,255,0),cv::Scalar(255,128,128), cv::Scalar(128,255,128), cv::Scalar(128,128,255)};
	int colors_length = sizeof(colors)/sizeof(colors[1]);

	for (int i1 = 1; i1 < stats.rows; i1++) {
		double cx = centroids.at<double>(i1, 0); 	//zu winkel alpha
		double cy = centroids.at<double>(i1, 1);	//zu distanz p
		//.do sth with values..
		
		for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++) { 
			if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho && fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha) {
				cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first);
				cv::line(colorImage, poi, poi, colors[i1 % colors_length], 3);
			}
 		}
	}

	int64 endTic_e = cv::getTickCount();
	deltaTime = (double) (endTic_e - endTic_d)/cv::getTickFrequency();
	std::cout << "time for step e:" << (int) (1000*deltaTime) << " ms" << std::endl;



	*m_proc_image[0] = imgDx;
	*m_proc_image[1] = imgCanny;
	*m_proc_image[2] = colorImage;

	//*m_proc_image[2] = resultImage;

	int64 endTic = cv::getTickCount();
	deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
	std::cout << "total processing time:" << (int) (1000*deltaTime) << " ms" << std::endl;



	return(SUCCESS);
}









