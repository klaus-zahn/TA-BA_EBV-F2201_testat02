

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
	
	//Bilder einlesen
	if(!image) return(EINVALID_PARAMETER);
	cv::Mat grayimage;
	cv::Mat colorImage;
	if(image->channels()>1){
		cv::cvtColor(*image, grayimage, cv::COLOR_RGB2GRAY);
		colorImage = *image;
	}else{
		grayimage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}	

	//Ableitungen kreieren
	int64 startTic = cv::getTickCount();
	cv::Mat imgDx;
	cv::Mat imgDy;
	cv::Sobel(grayimage, imgDx, CV_16S, 1, 0 ,3, 1, 0);//ZaK do not add 128
	cv::Sobel(grayimage, imgDy, CV_16S, 0, 1 ,3, 1, 0);
	int64 endTic = cv::getTickCount();
	double deltaTime = (double)(endTic-startTic)/cv::getTickFrequency();
	std::cout << "time Sobel:" << (int)(1000*deltaTime)<<"ms"<<std::endl;
	//Canny Edgde Filter anwenden und Grenzen bestimmen
	startTic = cv::getTickCount();
	cv::Mat imgCanny;
	double treshold1 = 10;
	double treshold2 = 60;
	cv::Canny(grayimage, imgCanny, treshold1, treshold2, 3, true);
	endTic = cv::getTickCount();
	deltaTime = (double)(endTic-startTic)/cv::getTickFrequency();
	std::cout << "time Canny:" << (int)(1000*deltaTime)<<"ms"<<std::endl;

	//Akkumulatorbild kreieren
	startTic = cv::getTickCount();
	double dx;
	double dy;
	double alpha;
	double rho;
	const double pi = 3.14159;
	uint16_t rhomax = sqrt((grayimage.rows*grayimage.rows)+(grayimage.cols*grayimage.cols));
	uint16_t alphamax = pi;
	uint16_t indrho, indalpha;
	double toleranceRho = 1.5;
	double toleranceAlpha = (0.5/360)*2*pi;
	cv::Mat accImg = cv::Mat::zeros((uint16_t)ceil(2*rhomax/toleranceRho)+1, (uint16_t)ceil(2*alphamax/toleranceAlpha)+1, CV_16U);
	std::vector<std::pair<int, int> > vecRowCol;
	std::vector<std::pair<int, int> > vecRhoAlpha;

	for(int rows = 0; rows < imgCanny.rows; rows++){
		for(int cols = 0; cols < imgCanny.cols; cols++){
			if(imgCanny.at<uint8_t>(rows, cols) != 0){
				dx = imgDx.at<int16_t>(rows, cols);
				dy = imgDy.at<int16_t>(rows, cols);
				alpha = atan2(dy, dx);
				rho = cols*cos(alpha)+rows*sin(alpha);
				indrho = round(rho/toleranceRho)+round(rhomax/toleranceRho)+1;
    			indalpha = round(alpha/toleranceAlpha)+round(alphamax/toleranceAlpha)+1;
				accImg.at<uint16_t>(indrho, indalpha) += 1;
				vecRhoAlpha.push_back(std::vector<std::pair<int, int> >:: value_type(indrho, indalpha));
				vecRowCol.push_back(std::vector<std::pair<int, int> >:: value_type(rows, cols));
			}
		}
	}
	double min = 0;
	double max = 0;
	cv::minMaxIdx(accImg, &min, &max);
	std::cout << "max Acc value: " << (int)(max)<<std::endl;
	endTic = cv::getTickCount();
	deltaTime = (double)(endTic-startTic)/cv::getTickFrequency();
	std::cout << "time Acc:" << (int)(1000*deltaTime)<<"ms"<<std::endl;

	//Bereite Akkumulatorbild auf
	startTic = cv::getTickCount();
	cv::Size filterBlur(3, 3);
	cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);

	double treshold = 5;
	cv::threshold(accImg, binaryimage, treshold, 255, CV_16U);
	binaryimage.convertTo(binaryimage, CV_8U);
	cv::Mat kernel = cv::Mat::ones(10, 10, CV_8UC1);
	cv::morphologyEx(binaryimage, binaryimage, cv::MORPH_CLOSE, kernel);
	
	cv::Mat stats, centroids, labelImage;
	cv::connectedComponentsWithStats(binaryimage, labelImage, stats, centroids);
	
	endTic = cv::getTickCount();
	deltaTime = (double)(endTic-startTic)/cv::getTickFrequency();
	std::cout << "time Morph:" << (int)(1000*deltaTime)<<"ms"<<std::endl;

	//Verbinde Pixel mit ACC-Bild und male die Grenzen
	startTic = cv::getTickCount();
	double cx;
	double cy;
	int const nHoodRho = 15;
	int const nHoodAlpha = 15;

	for(int i1 = 1; i1<stats.rows; i1++){
		cx = centroids.at<double>(i1,0);
		cy = centroids.at<double>(i1,1);

		for(uint i0 = 0; i0<vecRhoAlpha.size(); i0++){
			if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho && fabs(vecRhoAlpha[i0].second-cx)<nHoodAlpha){
				cv::Point poi = cv::Point(vecRowCol[i0].second, vecRowCol[i0].first);
				switch(i1%7){
					case 1:
						cv::line(colorImage, poi, poi, cv::Scalar(255, 0 ,0), 3);
						break;
					case 2:
						cv::line(colorImage, poi, poi, cv::Scalar(255, 255 ,0), 3);
						break;
					case 3:
						cv::line(colorImage, poi, poi, cv::Scalar(0, 255 ,0), 3);
						break;
					case 4:
						cv::line(colorImage, poi, poi, cv::Scalar(0, 0 ,255), 3);
						break;
					case 5:
						cv::line(colorImage, poi, poi, cv::Scalar(0, 255 ,255), 3);
						break;
					case 6:
						cv::line(colorImage, poi, poi, cv::Scalar(255, 0 ,255), 3);
						break;
				}
			}
		}
	}
	endTic = cv::getTickCount();
	deltaTime = (double)(endTic-startTic)/cv::getTickFrequency();
	std::cout << "time Draw:" << (int)(1000*deltaTime)<<"ms"<<std::endl;

	*m_proc_image[0] = imgDx;
	*m_proc_image[1] = imgCanny;
	*m_proc_image[2] = colorImage;

	return(SUCCESS);
}









