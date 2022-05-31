//ZaK :-)
#include "image_processing.h"


CImageProcessor::CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		// index 0 is 3 channels and indicies 1/2 are 1 channel deep
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
	int64_t startTic0 = cv::getTickCount();
	if(!image) return(EINVALID_PARAMETER);	
	cv::Mat grayImage;
	cv::Mat colorImage;
	//Punkt 1. In Graubild konvertieren
	if(image->channels() > 1){
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY);
		colorImage = *image;
	}
	else{
		grayImage = *image;
		cv::cvtColor(*image,colorImage,cv::COLOR_GRAY2RGB);
	}
	int64_t endTic0 = cv::getTickCount();
	//Punkt 2 und 3. Partielle Ableitung bestimmen
	int64_t startTic1 = cv::getTickCount();
	cv::Mat imgDx;
	cv::Mat imgDy;
	cv::Sobel(grayImage,imgDx,CV_16S,1,0,3,1,0);//ZaK do not add 128
	cv::Sobel(grayImage,imgDy,CV_16S,0,1,3,1,0);
	*m_proc_image[0] = imgDx;
	int64_t endTic1 = cv::getTickCount();
	//Punkt 4. Canny Edge Filter implementieren
	int64_t startTic2 = cv::getTickCount();
	double threshhold1 = 50;
	double threshhold2 = 200;
	cv::Mat imgCanny;
	cv::Canny(grayImage, imgCanny, threshhold1, threshhold2, 3, true);
	*m_proc_image[1] = imgCanny;
	int64_t endTic2 = cv::getTickCount();
	//Punkt 5 und 8 Erstellung Akkumulatorbild und abspeichern der zugehörigen Bildkoordinaten zu x und y sowie Alpha und Rho
	int64_t startTic3 = cv::getTickCount();
	std::vector<MyAccPixel> AccPixelLocalisation;	

	int AlphaBins = 360*1.5;//Auflösung von 2/3 Grad
	int RhoBins = 80; //Auflösung von 80 Schritten
	double RhoMin = -imgCanny.cols*cos(CV_PI/4)-imgCanny.rows*sin(CV_PI/4);//Minimal mögliches Rho im Bild
	double RhoMax = imgCanny.cols*cos(CV_PI/4)+imgCanny.rows*sin(CV_PI/4);//Maximal mögliches Rho im Bild

	cv::Mat accImg = cv::Mat::zeros(RhoBins, AlphaBins, CV_16U);
	for(int rows = 0; rows < imgCanny.rows;rows++){
		for(int cols = 0;cols < imgCanny.cols; cols++){
			if(imgCanny.at<uint8_t>(rows, cols) != 0){
				double dx = imgDx.at<int16_t>(rows, cols);
				double dy = imgDy.at<int16_t>(rows, cols);				

				double alpha = atan2(dy,dx);
				double rho = cols*cos(alpha)+rows*sin(alpha);
				int IndexAlpha = round((alpha+CV_PI)/(2*CV_PI)*AlphaBins);
				int IndexRho = round((rho-RhoMin)/(RhoMax-RhoMin)*RhoBins);
				if(IndexRho >= RhoBins){IndexRho = RhoBins-1;}					
				if(IndexAlpha >= AlphaBins){IndexAlpha = AlphaBins-1;}				
				accImg.at<int16_t>(IndexRho,IndexAlpha) +=1;
				//Teil Aufgabe 8 abspeichern der zugehörigen Bildkoordinaten zu x und y sowie Alpha und Rho
				MyAccPixel AccPixel;
				AccPixel.Alpha = IndexAlpha;
				AccPixel.Rho = IndexRho;
				AccPixel.ImX = cols;
				AccPixel.ImY = rows;
				AccPixelLocalisation.push_back(AccPixel);
			}
		}
	}
	int64_t endTic3 = cv::getTickCount();
	//Punkt 6 Aufbereitung des Akkumulatorbildes durch Glättungsfilter
	int64_t startTic4 = cv::getTickCount();
	cv::Size filterBlur(3,3);
	cv::GaussianBlur(accImg, accImg, filterBlur,1.5);
	double threshold = 3;//Threshhold einstellen
	cv::Mat binaryImage;
	cv::threshold(accImg, binaryImage, threshold, 255, cv::THRESH_BINARY);
	binaryImage.convertTo(binaryImage,CV_8U);
	int64_t endTic4 = cv::getTickCount();
	//Punkt 7 Akkumulatorbild aufbereiten und Zentren finden
	int64_t startTic5 = cv::getTickCount();
	cv::Mat kernel = cv::Mat::ones(3,3,CV_8UC1);
	cv::morphologyEx(binaryImage,binaryImage,cv::MORPH_CLOSE,kernel);
	cv::Mat stats, centroids, labelImage;
	connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
	int64_t endTic5 = cv::getTickCount();
	int64_t startTic6;
	int64_t endTic6;
	int64_t startTic7;
	int64_t endTic7;
	for(int i1=1;i1<stats.rows; i1++){
		startTic6 = cv::getTickCount();
		MyAccStats AccStat;
		AccStat.CentRho = centroids.at<double>(i1,1);
		AccStat.CentAlpha = centroids.at<double>(i1,0);
		//Ermitteln der oberen Alpha Toleranz
		int8_t alpha = binaryImage.at<int8_t>((int16_t)AccStat.CentRho,(int16_t)AccStat.CentAlpha);
		int Index = 0;		
		while(alpha == -1){
			++Index;
			if((AccStat.CentAlpha+Index)<binaryImage.cols){
				alpha = binaryImage.at<int8_t>(AccStat.CentRho,AccStat.CentAlpha+Index);
			}
			else{
				break;
			}			
		}
		AccStat.AlphaThreshTop = Index-1;
		//Ermitteln der unteren Alpha Toleranz
		alpha = binaryImage.at<int8_t>(AccStat.CentRho,AccStat.CentAlpha);
		Index = 0;		
		while(alpha == -1){
			--Index;
			if((AccStat.CentAlpha+Index)>=0){
				alpha = binaryImage.at<int8_t>(AccStat.CentRho,AccStat.CentAlpha+Index);
			}
			else{
				break;
			}			
		}
		AccStat.AlphaThreshBottom = Index+1;
		//Ermitteln der oberen Rho Toleranz
		int8_t rho = binaryImage.at<int8_t>(AccStat.CentRho,AccStat.CentAlpha);
		Index = 0;		
		while(rho == -1){
			++Index;
			if((AccStat.CentRho+Index)>=0){
				rho = binaryImage.at<int8_t>(AccStat.CentRho+Index,AccStat.CentAlpha);
			}
			else{
				break;
			}			
		}
		AccStat.RhoThreshTop = Index-1;
		//Ermitteln der unteren Rho Toleranz
		rho = binaryImage.at<int8_t>(AccStat.CentRho,AccStat.CentAlpha);
		Index = 0;		
		while(rho == -1){
			--Index;
			if((AccStat.CentRho+Index)>=0){
				rho = binaryImage.at<int8_t>(AccStat.CentRho+Index,AccStat.CentAlpha);
			}
			else{
				break;
			}			
		}
		AccStat.RhoThreshBottom = Index+1;	
		endTic6 = cv::getTickCount();			
		//Punkt 8 Originalbild markieren
		startTic7 = cv::getTickCount();
		cv::Scalar Color;
		if((i1%7) == 1){
			Color = cv::Scalar(255,0,0);
		}
		else if((i1%7) == 2){
			Color = cv::Scalar(0,255,0);
		}
		else if((i1%7) == 3){
			Color = cv::Scalar(255,255,0);
		}
		else if((i1%7) == 4){
			Color = cv::Scalar(0,0,255);
		}
		else if((i1%7) == 5){
			Color = cv::Scalar(255,0,255);
		}
		else if((i1%7) == 6){
			Color = cv::Scalar(0,255,255);
		}
		else{
			Color = cv::Scalar(255,125,255);
		}
		for(uint i0=0; i0<AccPixelLocalisation.size(); i0++){		
			if((AccPixelLocalisation[i0].Alpha <= AccStat.CentAlpha+2*AccStat.AlphaThreshTop) &&
			(AccPixelLocalisation[i0].Alpha >= AccStat.CentAlpha+2*AccStat.AlphaThreshBottom)&&
			(AccPixelLocalisation[i0].Rho <=  AccStat.CentRho+2*AccStat.RhoThreshTop)&&
			(AccPixelLocalisation[i0].Rho >= AccStat.CentRho+2*AccStat.RhoThreshBottom)){
				cv::Point poi = cv::Point(AccPixelLocalisation[i0].ImX,AccPixelLocalisation[i0].ImY);				
				cv::line(colorImage,poi,poi,Color,3);
			}
		}
		endTic7 = cv::getTickCount();
	}
	//Bild anzeigen
	*m_proc_image[2] = colorImage;
	//Punkt 11	
	double deltaTime = (double)(endTic0-startTic0)/cv::getTickFrequency();
	std::cout<<"time for step 0:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	deltaTime = (double)(endTic1-startTic1)/cv::getTickFrequency();
	std::cout<<"time for step 1:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	deltaTime = (double)(endTic2-startTic2)/cv::getTickFrequency();
	std::cout<<"time for step 2:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	deltaTime = (double)(endTic3-startTic3)/cv::getTickFrequency();
	std::cout<<"time for step 3:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	deltaTime = (double)(endTic4-startTic4)/cv::getTickFrequency();
	std::cout<<"time for step 4:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	deltaTime = (double)(endTic5-startTic5)/cv::getTickFrequency();
	std::cout<<"time for step 5:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	deltaTime = (double)(endTic6-startTic6)/cv::getTickFrequency();
	std::cout<<"time for step 6:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	deltaTime = (double)(endTic7-startTic7)/cv::getTickFrequency();
	std::cout<<"time for step 7:"<<(int) (1000*deltaTime)<<"ms"<<std::endl;
	
	return(SUCCESS);
}







