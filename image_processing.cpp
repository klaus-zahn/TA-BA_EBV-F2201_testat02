

#include "image_processing.h"
#include <chrono>
#include <unistd.h>

using namespace std;


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
	
	auto start = chrono::steady_clock::now();	// time measurement (Task: (g))
	auto start_sobel = chrono::steady_clock::now(); // time measurement (Task: (g))



	if(!image) return(EINVALID_PARAMETER);
// Converting Image channels (gray/RGB)	
    cv::Mat grayImage;    
	cv::Mat colorImage;
    if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		colorImage = *image;
	} else {
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

// Sobel filter (Task: (a))
	cv::Mat imgDx;
	cv::Mat imgDy;
	cv::Sobel(grayImage, imgDx, CV_16S,1,0,3,1,0);//ZaK: do not add 128
	cv::Sobel(grayImage, imgDy, CV_16S,0,1,3,1,0);

	*m_proc_image[0] = imgDx;	// Write Dx Sobel to web-GUI (image processing 1)

	auto end_sobel = chrono::steady_clock::now(); // time measurement (Task: (g))
	auto start_canny = chrono::steady_clock::now(); // time measurement (Task: (g))
	
// Canny filter (Task: (b))
	double threshold1 = 50;
	double threshold2 = 200;
	cv::Mat imgCanny;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2,3,true);

	*m_proc_image[1] = imgCanny; // Write Canny-image to web-GUI (image processing 2)

	auto end_canny = chrono::steady_clock::now();	// time measurement (Task: (g))
	auto start_acc = chrono::steady_clock::now();	// time measurement (Task: (g))

// Task (c)
	// Calculate all necessary components
	double deltaAlpha= M_PI/180*1.5;
	double deltaRho = 1.5;
	float imgDiam = sqrt((imgCanny.rows)*((imgCanny.rows))+(imgCanny.cols)*(imgCanny.cols));
	int rhoBins = floor(1+2*imgDiam/deltaRho);
	int alphaBins = 1+floor(1+2*M_PI/deltaAlpha);
	float alpha_max = M_PI;
	float rho_max = imgDiam;

	cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);

	std::vector<std::pair<int, int> > vecRowCol;
	std::vector<std::pair<int, int> > vecRhoAlpha;

	
	for(int rows = 0; rows < imgCanny.rows; rows++) {
		for(int cols = 0; cols < imgCanny.cols; cols++) {
			if(imgCanny.at<uint8_t>(rows, cols) != 0) { 		// if Canny component is not 0
				double dx = imgDx.at<int16_t>(rows, cols);		// 
				double dy = imgDy.at<int16_t>(rows, cols);		// determine dx and dy at position
				double alpha = atan2(dy,dx);					//
				double rho = cols*cos(alpha) + rows*sin(alpha); // calculate corresponding alpha an rho

				int indexRho=floor((rho/rho_max)*0.5*rhoBins)+ceil(rhoBins/2); // transform rho and alpha to corresponding index
				int indexAlpha=floor((alpha/alpha_max)*0.5*alphaBins)+ceil(alphaBins/2); //

				vecRowCol.push_back(std::vector<std::pair<int, int> >::value_type(rows, cols)); // store coordinates
				vecRhoAlpha.push_back(std::vector<std::pair<double, double> >::value_type(indexRho, indexAlpha)); // store rho-alpha-pairs
				
				accImg.at<u_int16_t>(indexRho, indexAlpha) += 1; // increment accImg at coordinates
			}
		}
	}
	auto end_acc = chrono::steady_clock::now();		// time measurement (Task: (g))
	auto start_morph = chrono::steady_clock::now();	// time measurement (Task: (g))
	
// filter and morphology of accImg (Task: (d))
	cv::Size filterBlur(3,3);
	cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);

	cv::Mat binaryImage;
	double threshold = 5;
	cv::threshold(accImg, binaryImage, threshold, 255,CV_16U);

	binaryImage.convertTo(binaryImage,CV_8U);

	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

	cv::Mat stats, centroids, labelImage;
	connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

	auto end_morph = chrono::steady_clock::now();	// time measurement (Task: (g))
	auto start_lines = chrono::steady_clock::now();	// time measurement (Task: (g))

// draw Hough-lines (Task: (e))
	float nHoodRho = 5 * deltaRho;		// Neighbourhood-size config
	float nHoodAlpha = 5;				// did fixed size for alpha -> stability

	// define color map for displaying different colors 
	cv::Scalar Cmap[9] = {{255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255},{255,128,128},{128,255,128},{128,128,255}};

	// drawing loop
	for (int i1 = 1; i1 < stats.rows; i1++) {
		double cx = centroids.at<double>(i1, 0);
		double cy = centroids.at<double>(i1, 1);

		for(uint i0 = 0; i0 < vecRhoAlpha.size(); i0++) {
			if(fabs(vecRhoAlpha[i0].first-cy) < nHoodRho &&
			fabs(vecRhoAlpha[i0].second-cx) < nHoodAlpha) {
				cv::Point poi = cv::Point(vecRowCol[i0].second,vecRowCol[i0].first);
				cv::line(colorImage, poi, poi, Cmap[i1%9],3);				
			}	
		}
		
	}
	*m_proc_image[2] = colorImage; // Write Hough-image to web-GUI (image processing 3)

	auto  end = chrono::steady_clock::now();	// time measurement (Task: (g))

	// calculate all durations
	int sobel_time = chrono::duration_cast<chrono::milliseconds>(end_sobel - start_sobel).count();
	int canny_time = chrono::duration_cast<chrono::milliseconds>(end_canny - start_canny).count();
	int acc_time = chrono::duration_cast<chrono::milliseconds>(end_acc - start_acc).count();
	int morph_time = chrono::duration_cast<chrono::milliseconds>(end_morph - start_morph).count();
	int lines_time = chrono::duration_cast<chrono::milliseconds>(end - start_lines).count();
	int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	

	//print durations (Task: (g))
	OscLog(NOTICE,"-------------------------------\n");
	OscLog(NOTICE,"Total conversion time   : %d ms\n",total_time);
	OscLog(NOTICE,"Sobel filter time       : %d ms\n",sobel_time);
	OscLog(NOTICE,"Canny filter time       : %d ms\n",canny_time);
	OscLog(NOTICE,"Accu-img filling time   : %d ms\n",acc_time);
	OscLog(NOTICE,"Morphology time         : %d ms\n",morph_time);
	OscLog(NOTICE,"Hough lines drawing time: %d ms\n",lines_time);
	OscLog(NOTICE,"-------------------------------\n");
	


	return(SUCCESS);
}









