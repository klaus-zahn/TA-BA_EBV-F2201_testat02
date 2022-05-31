

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
        
	//cv::subtract(cv::Scalar::all(255), *image,*m_proc_image[0]);
	
	bool changeDetectioIsOn = false;
	double threshold1 = 50;
	double threshold2 = 200;
	
	cv::Mat grayImage;
	cv::Mat colorImage;
	if(image->channels()>1){
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		colorImage = *image;
	}else{
		grayImage = *image;
		cv::cvtColor( *image, colorImage, cv::COLOR_GRAY2RGB );
	}

	cv::Mat imgDx;
	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);//ZaK do not add 128

	cv::Mat imgDy;
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);

	cv::Mat imgCanny;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

	double deltaAlpha = M_PI/180*1.5; //angle bin size in radiant
	double deltaRho = 1.5; //rho bin size in pixel

	double imageDiameter = sqrt((imgCanny.rows)*(imgCanny.rows) + (imgCanny.cols)*(imgCanny.cols));

	double alphaBins = 1+floor(1+2*M_PI/deltaAlpha);
	double rhoBins = floor(1+2*imageDiameter/deltaRho);
	float alphaMax = M_PI;
	float rhoMax = imageDiameter;
	cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);

	std::vector<std::pair<int, int> > vecRowCol;
	std::vector<std::pair<int, int> > vecRhoAlpha;
	int64 startTic = cv::getTickCount();
	
	for(int rows = 0; rows < imgCanny.rows; rows++) {
		for(int cols = 0; cols < imgCanny.cols; cols++) {
			if(imgCanny.at<uint8_t>(rows, cols) != 0) {
				double dx = imgDx.at<int16_t>(rows, cols);
				double dy = imgDy.at<int16_t>(rows, cols);

				double alpha = atan2(dy, dx);
				double rho = cols*cos(alpha) + rows*sin(alpha);//ZaK wrong formula

				uint16_t indexRho=floor((rho/rhoMax)*0.5*rhoBins)+ceil(rhoBins/2);
				uint16_t indexAlpha=floor((alpha/alphaMax)*0.5*alphaBins)+ceil(alphaBins/2);

				vecRowCol.push_back(std::vector<std::pair<int, int> >::value_type(rows, cols));
				vecRhoAlpha.push_back(std::vector<std::pair<double, double> >::value_type(indexRho, indexAlpha));
				
				accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;
			}
		}
	}
	//ZaK time measure not for full processing
	int64 endTic = cv::getTickCount(); // see how long it takes to make Accu Picture
	double deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
	std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;

	cv::Size filterBlur(3,3);
	cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);

	double threshold = 5;
	cv::Mat binaryImage;
	cv::threshold(accImg, binaryImage, threshold, 255, cv::THRESH_BINARY);

	binaryImage.convertTo(binaryImage, CV_8U);

	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

	cv::Mat stats, centroids, labelImage;
	connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

	cv::Scalar colors[] = {	cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
								cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)};

	for (int i1 = 1; i1 < stats.rows; i1++) { 
		double cx = centroids.at<double>(i1, 0); 
		double cy = centroids.at<double>(i1, 1); 
		cv::Scalar lineColor = colors[i1 % (sizeof(colors)/sizeof(colors[1]))];
		int nHoodRho = 15;
		int nHoodAlpha = 15;
		for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++) { 
			if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho && fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha) { 
			cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first); 
			cv::line(colorImage, poi, poi, lineColor, 3); 
			} 
		}

	}

	//ZaK leave this out
	if (changeDetectioIsOn && m_prev_image.size() != cv::Size()){
		double threshold = 30;

		cv::Mat diffImage;
		cv::absdiff(*image, m_prev_image, diffImage);

		cv::Mat binaryImage;
		cv::threshold(diffImage, binaryImage, threshold, 255, cv::THRESH_BINARY);

		cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
		cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

		cv::Mat stats, centroids, labelImage;
		connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

		cv::Mat resultImage = image->clone();
		for (int i = 1; i < stats.rows; i++) {
			int topLeftx = stats.at<int>(i, 0);
			int topLefty = stats.at<int>(i, 1);
			int width = stats.at<int>(i, 2);
			int height = stats.at<int>(i, 3);
			int area = stats.at<int>(i, 4);
			double cx = centroids.at<double>(i, 0);
			double cy = centroids.at<double>(i, 1);

			cv::Rect rect(topLeftx, topLefty, width, height);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));

			cv::Point2d cent(cx, cy);
			cv::circle(resultImage, cent, 5, cv::Scalar(128, 0, 0), -1);

		}

		
		*m_proc_image[0] = resultImage;
		*m_proc_image[1] = binaryImage;
		*m_proc_image[2] = labelImage;
		
		

		//  cv::imwrite("dx.png", *m_proc_image[0]);
		//  cv::imwrite("dy.png", *m_proc_image[1]);
	} 

	//ZaK keep this out of the change detection loop
	if (changeDetectioIsOn == false){
		*m_proc_image[0] = imgCanny;
		*m_proc_image[1] = accImg;
		*m_proc_image[2] = colorImage;
	}

	m_prev_image = grayImage.clone();

	return(SUCCESS);
}









