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
        
		cv::Mat grayImage;
		cv::Mat colorImage;
		cv::Mat imgDx;
		cv::Mat imgDy;
		cv::Mat imgCanny;
		cv::Mat rhoBins;
		cv::Mat alphaBins;
		cv::Mat imgDr;
		cv::Mat binaryImage;
		cv::Mat edgeYw;
		cv::Mat edgeXw;
		float deltaAlpha = M_PI/180*1.5;
		float deltaRho = 1.5;
		cv::Size sz = image->size();
		int imageWidth = sz.width;
		int imageHeight = sz.height;
		int imageDiameter = sqrt(pow(imageWidth,2) + pow(imageHeight,2));
		std::vector<std::pair<int, int> > vecRowCol;
		std::vector<std::pair<int, int> > vecRhoAlpha;
		int64 startTic = cv::getTickCount();
		
		


		if(image->channels() > 1) {
 			cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
			colorImage = *image;
		} else {
			grayImage = *image;
			cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
		}

		if(m_prev_image.size() != cv::Size()){

			cv::Sobel(grayImage, imgDx, CV_16S, 1, 0);
			cv::Sobel(grayImage, imgDy, CV_16S, 0, 1);
			//imgDr = sqrt(imgDx*imgDx+imgDy*imgDy);

			double threshold1 = 50;
			double threshold2 = 200;
			cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

			int alphaBins = 1+floor(1+2*3.14/deltaAlpha);
			int rhoBins = floor(1+2*imageDiameter/deltaRho);
			cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);
			for(int rows = 0; rows < imgCanny.rows; rows++) {
				for(int cols = 0; cols < imgCanny.cols; cols++) {
					if(imgCanny.at<uint8_t>(rows, cols) != 0) {
						double dx = imgDx.at<int16_t>(rows, cols);
						double dy = imgDy.at<int16_t>(rows, cols);
						double alpha = atan2(dx,dy);
						//double dr = imgDr.at<int16_t>(rows, cols);
						double dr = std::sqrt(dx*dx+dy*dy);
						double rho = cols*dx/dr + rows*dy/dr;
						double alphaDerive = (alphaBins - 1) / (2*M_PI);
                        int indexAlpha = round(alphaDerive * (alpha + M_PI));
        				double RhoDerive = (rhoBins - 1) / (2*imageDiameter);
                        int indexRho = round(RhoDerive * (rho + imageDiameter));
						vecRowCol.push_back(std::vector<std::pair<int, int> >::value_type(rows, cols));
						vecRhoAlpha.push_back(std::vector<std::pair<double, double> >::value_type(indexRho, indexAlpha));
						accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;
					}
				}
			}
			cv::Size filterBlur(3,3);
			cv::GaussianBlur(accImg, accImg, filterBlur, 1.5); 
			double threshold = 5;
			cv::threshold(accImg, binaryImage, threshold, 255,cv::THRESH_BINARY);
			binaryImage.convertTo(binaryImage, CV_8U);
			cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1); 
			cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);
			cv::Mat stats, centroids, labelImage;
			connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
			int minArea = 1;
			int nHoodAlpha = 5;
			int nHoodRho = 5;
			
			for(uint i0 = 0; i0 < vecRhoAlpha.size();i0++){
				for (int i1 = 1; i1 < stats.rows; i1++) {
					int area = stats.at<int>(i1,4);
					if(area > minArea){
						double cx = centroids.at<double>(i1, 0);
 						double cy = centroids.at<double>(i1, 1);
						
						 if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho && fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha){
							cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first);
							int indx = i1%6;
							int cMap[6][3] = {{0,0,255},{255,255,0},{255,255,255},{255,0,255},{255,0,0},{0,255,0}};
							cv::Scalar RGB = cv::Scalar(cMap[indx][0],cMap[indx][1],cMap[indx][2]);
							cv::line(colorImage, poi, poi, RGB, 3);
						 }
					}
				}
			}

				*m_proc_image[0] = imgCanny;
				*m_proc_image[1] = accImg;
				*m_proc_image[2] = colorImage;
				int64 endTic = cv::getTickCount();
				double deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
				std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;
        }
		m_prev_image = grayImage.clone();
	return(SUCCESS);
}