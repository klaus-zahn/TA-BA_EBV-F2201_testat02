//ZaK I removed some of the problems but not all

#include "image_processing.h"
#include <iostream>
#include <chrono>


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
	//convert image to grayscale (if not already)
	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
	} 
	else { 
		grayImage = *image;
	}


	auto start_time = std::chrono::high_resolution_clock::now();


	// Aufgabe 1 a
	// calculate partiall deriatives
	cv::Mat sobelImageX, sobelImageY, sobelImage;
	int ddepth = -1;
	int ksize = 3;
	double scale = 1.0;
	double delta = 0.0;
	int borderType = 4;
	cv::Sobel(*image, sobelImageX, ddepth, 1, 0, ksize, scale, delta, borderType);
	cv::Sobel(*image, sobelImageY, ddepth, 0, 1, ksize, scale, delta, borderType);
	cv::Sobel(*image, sobelImage, ddepth, 1, 1, ksize, scale, delta, borderType);


	auto end_time = std::chrono::high_resolution_clock::now();
  	auto time = end_time - start_time;
	std::cout << "time for step 1a: " << time/std::chrono::milliseconds(1) << "ms\n";
	start_time = std::chrono::high_resolution_clock::now();


	// Aufgabe 1 b
	// find edges with canny edge detection
	cv::Mat cannyImage;
	double thres1 = 100;
	double thres2 = 200;
	int apertureSize = 3;
	bool L2gradient = false;
	cv::Canny(sobelImage, cannyImage, thres1, thres2, apertureSize, L2gradient);


	end_time = std::chrono::high_resolution_clock::now();
  	time = end_time - start_time;
	std::cout << "time for step 1b: " << time/std::chrono::milliseconds(1) << "ms\n";
	start_time = std::chrono::high_resolution_clock::now();


	// Aufgabe 1 c
	// calculate alpha and phi for every pixel in cannyImage
	uint rows = image->rows;
	uint cols = image->cols;
	uint size = rows * cols;
	float * rho = new float[size];
	float * alpha = new float[size];
	float maxAlpha, minAlpha, maxRho, minRho;
	float alphaRes = 1.5/180 * CV_PI;//ZaK wrong brackets
	float rhoRes = 1.5;
	// calculate hough parameters
	for (uint i = 0; i < size; i++)
	{
		//ZaK wrong formula for indices
		uint row = i/cols;
		uint col = i%cols;
		// alpha
		alpha[i] = atan2(sobelImageY.at<float>(row, col), sobelImageX.at<float>(row, col));
		// rho
		// rho = Xcoordinates * cos(alpha) + Ycoordinates * sin(alpha)
		rho[i] = col * cos(alpha[i]) + row * sin(alpha[i]);
	}
	// preallocate min & max values
	maxAlpha = alpha[0];
	minAlpha = alpha[0];
	maxRho = rho[0];
	minRho = rho[0];
	// determinate max and min values of alpha and rho
	for (uint i = 0; i < size; i++)
	{
		if (maxAlpha < alpha[i])
		{
			maxAlpha = alpha[i];
		}
		else if (minAlpha > alpha[i])
		{
			minAlpha = alpha[i];
		}
		if (maxRho < rho[i])
		{
			maxRho = rho[i];
		}
		else if (minRho > rho[i])
		{
			minRho = rho[i];
		}
	}
	// create accumulatorImage and initialize all values in it with 0
	int X = round(((maxAlpha-minAlpha)/alphaRes+1));
	int Y = round((maxRho-minRho)/rhoRes+1);
	cv::Mat accumulatorImage = cv::Mat::zeros(cv::Size( X, Y), CV_8UC1);
	/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// bis hier hin konnte das Programm erfolgreich getestet werden.
	// danach wurden diverse, nicht verfolgbare speicherfehler, segfaults ausgelöst (auch ausserhalb dieser Funktion)
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
	// generate data in accumulator Image
	for (uint i = 0; i < size; i++)
	{
		int AccX = round((alpha[i]-minAlpha)/alphaRes)+1;
		int AccY = round((rho[i]-minRho)/rhoRes)+1;
		accumulatorImage.at<uint>(AccY,AccX)++; 
	}


	end_time = std::chrono::high_resolution_clock::now();
  	time = end_time - start_time;
	std::cout << "time for step 1c: " << time/std::chrono::milliseconds(1) << "ms\n";
	start_time = std::chrono::high_resolution_clock::now();


	// Aufgabe 1 d
	// smooth out Image (lowpass 3x3)
	cv::Mat kernel = cv::Mat::ones(cv::Size(3,3), CV_8UC1);
	cv::filter2D(accumulatorImage, accumulatorImage, CV_8U, kernel);
	// optionale schliessung und anschliessende öffnung
	cv::morphologyEx(accumulatorImage, accumulatorImage, cv::MORPH_CLOSE, kernel);
	cv::morphologyEx(accumulatorImage, accumulatorImage, cv::MORPH_OPEN, kernel);
	// get ROI's
	cv::Mat stats, centroids, labelImage;
	cv::connectedComponentsWithStats(accumulatorImage, labelImage, stats, centroids);
	// get the 6 biggest ROI's and the stats of those
	uint maxCount = 6;
	uint maxAcc[maxCount][5];
	for (uint i = 0; i < maxCount; i++)
	{
		maxAcc[i][0] = 0;
	}
	for (int i = 1; i < stats.rows; i++)
	{ 
		uint area = stats.at<uint>(i, 4);
		if (maxAcc[0][0] < area)
		{
			for (uint i = 1; i < maxCount; i++)
			{
				maxAcc[i][0] = maxAcc[i-1][0];
				maxAcc[i][1] = maxAcc[i-1][1];
				maxAcc[i][2] = maxAcc[i-1][2];
				maxAcc[i][3] = maxAcc[i-1][3];
				maxAcc[i][4] = maxAcc[i-1][4];
			}
			maxAcc[0][0] = area;
			maxAcc[0][1] = stats.at<int>(i, 0); // top left x (alpha)
			maxAcc[0][2] = stats.at<int>(i, 2); // width (d alpha)
			maxAcc[0][3] = stats.at<int>(i, 1); // top left y (rho)
			maxAcc[0][4] = stats.at<int>(i, 3); // height (d rho)
		}
	}


	end_time = std::chrono::high_resolution_clock::now();
  	time = end_time - start_time;
	std::cout << "time for step 1d: " << time/std::chrono::milliseconds(1) << "ms\n";
	start_time = std::chrono::high_resolution_clock::now();


	// Aufgabe 1 e
	// define colors
	unsigned char colors[6][3] = {{0,0,255},{0,255,0},{255,0,0},{0,255,255},{255,0,255},{255,255,0}};
	// check which alpha/rho combinations 
	for (uint i = 0; i < size; i++)
	{
		//ZaK wrong formula
		uint y = i/cols;
		uint x = i%cols;
		// alpha    //ZaK wrong order of x/y
		float Alpha = atan2(sobelImageY.at<float>(y, x), sobelImageX.at<float>(y, x));
		// rho
		// rho = Xcoordinates * cos(alpha) + Ycoordinates * sin(alpha)
		float Rho = x * cos(alpha[i]) + y * sin(alpha[i]);

		// check if those values are in a ROI
		for (uint j = 0; j < maxCount; j++)
		{
			// alpha is NOT in the width of this ROI
			if (!(Alpha < maxAcc[j][1] + maxAcc[j][3] && Alpha > maxAcc[j][1]))
			{
				continue;
			}
			// rho is NOT in the height of this ROI
			if (!(Rho < maxAcc[j][3] + maxAcc[j][4] && Rho > maxAcc[j][3]))
			{
				continue;
			}
			// if the programm runs this code, the pixel is in the searched ROI and has to get colored properly
			// if input image is grayscale
			if(image->channels() == 1) 
			{
				// convert to color imgae
				cv::cvtColor(*image, *image, cv::COLOR_GRAY2RGB);
  			}
			cv::Point point = cv::Point(x,y);
			// read color values
			cv::Vec3b color = image->at<cv::Vec3b>(point);
			// rewrite them
			color[0] = colors[j][0];
			color[1] = colors[j][1];
			color[2] = colors[j][2];
			// write them back in to the picture
			image->at<cv::Vec3b>(point) = color;
		}
	}


	end_time = std::chrono::high_resolution_clock::now();
  	time = end_time - start_time;
	std::cout << "time for step 1e: " << time/std::chrono::milliseconds(1) << "ms\n";
	start_time = std::chrono::high_resolution_clock::now();


	// free up memory after usage
	delete rho;
	delete alpha;

	// show requested images on website
	*m_proc_image[0] = sobelImage;
	*m_proc_image[1] = cannyImage;
	*m_proc_image[2] = *image;

	return(SUCCESS);
}









