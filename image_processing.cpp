//ZaK :-)
#include "image_processing.h"
#include <chrono>
#include <iostream>

//namespace zur chrono Verwendung
using namespace std::chrono;




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
        
        
		cv::Mat grayImage;			//Grauwertbild
		cv::Mat colorImage;			//Farbbild
		cv::Mat imgDx;				//Ausgabebild Dx
		cv::Mat imgDy;				//Ausgabebild Dy
		cv::Mat binaryImage; 		//Ausgabebild Binär
		cv::Mat imgCanny;			//Ausgabebild Canny

		//Vektorpaare der korrespondierenden Pixel
		std::vector<std::pair<int, int> > vecRowCol; 
		std::vector<std::pair<int, int> > vecRhoAlpha;



		cv::Scalar Color[8] = {		cv::Scalar(0,0,128),	//navy blue
									cv::Scalar(0,100,0),	//dark green
									cv::Scalar(255,140,0),	//dark orange
									cv::Scalar(139,0,0),	//dark red
									cv::Scalar(139,0,139),	//dark magenta
									cv::Scalar(255,185,15),	//gold
									cv::Scalar(255,255,0),	//yellow
									cv::Scalar(255,192,203)	//pink
		};



		//Zeitmessung step0
		auto start = high_resolution_clock::now();




		// Konversion in Grauwertbild + Vorbereitung Farbbild
		if(image->channels() > 1){
			cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
			colorImage = *image;
		} else { 
			grayImage = *image; 
			cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
		}




		// Ableitung bestimmen mit Sobel Kantenfilter  (Gleitkommaformat!!! 16bit signed => CV_16S)
		// input, output, Format, x-Ableitung, y-Ableitung, Grösse der Filtermaske, Skalierung output, Addition output
		cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);
		cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);

		*m_proc_image[0] = imgDx;



		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);
		std::cout << "Time for step0: " << duration.count() << " microseconds" << std::endl;
		//Zeitmessung step1
		start = high_resolution_clock::now();


		//Implementation Canny Edge Filter
		double threshold1 = 50; 
		double threshold2 = 200;
		cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

		*m_proc_image[1] = imgCanny;
		



		stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(stop - start);
		std::cout << "Time for step1: " << duration.count() << " microseconds" << std::endl;
		//Zeitmessung step2
		start = high_resolution_clock::now();

		int16_t alphaBins = (360+0.75) / 1.5;
		int16_t diameter = sqrt(imgCanny.rows*imgCanny.rows+imgCanny.cols*imgCanny.cols);
		int16_t rhoBins = (2*diameter+0.75) / 1.5;

		//Implementation Akkumulator Bild
		cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);

		//"Füllung" des Akkumulator Bildes
		for(int rows = 0; rows < imgCanny.rows; rows++) { 
			for(int cols = 0; cols < imgCanny.cols; cols++) { 
				if(imgCanny.at<uint8_t>(rows, cols) != 0) { 
					double dx = imgDx.at<int16_t>(rows, cols); 
					double dy = imgDy.at<int16_t>(rows, cols);
					double alpha = atan2(dy, dx)*180.0/CV_PI;
					double dv = sqrt(dx*dx+dy*dy);
					double rho = cols * dx/dv + rows * dy/dv;
					int16_t indexRho = (diameter+rho)/1.5;
					int16_t indexAlpha = (180+alpha)/1.5;
					accImg.at<u_int16_t>(indexRho, indexAlpha) += 1;
					vecRowCol.push_back(std::vector<std::pair<int, int> >:: value_type(rows, cols));
					vecRhoAlpha.push_back(std::vector<std::pair<int, int> >::value_type(indexRho, indexAlpha));
				} 				
			}
		}
		

		//Glättungsfilter
		cv::Size filterBlur(3,3); 
		cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);

		//Schwellwertoperation
		double threshold = 5; 
		cv::threshold(accImg, binaryImage, threshold, 255, 0);

		//Konvertierung auf CV_8U
		binaryImage.convertTo(binaryImage, CV_8U);

		//Nahe beeinander liegende Peaks zusammenfassen
		cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1); 
		cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE,kernel);
		
		//Gemeinsame Regionen extrahieren
		cv::Mat stats, centroids, labelImage; 
		connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

		stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(stop - start);
		std::cout << "Time for step 2: " << duration.count() << " microseconds" << std::endl;
		//Zeitmessung step3
		start = high_resolution_clock::now();



		//Definition Nachbarschaftsregion
		double nHoodRho = 2;
		double nHoodAlpha = 2;

		for (int i1 = 1; i1 < stats.rows; i1++){ 
			double cx = centroids.at<double>(i1, 0); 
			double cy = centroids.at<double>(i1, 1); 
			
			
			for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++) {
				if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho && 
				   fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha) {
						cv::Point poi = cv::Point(vecRowCol[i0].second, 
												  vecRowCol[i0].first);

					cv::line(colorImage, poi, poi, 
										Color[i1%8],3);
				}
			}
		}


		*m_proc_image[2] = colorImage.clone();

		stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(stop - start);
		std::cout << "Time for step3: " << duration.count() << " microseconds" << std::endl;
		


	return(SUCCESS);
}