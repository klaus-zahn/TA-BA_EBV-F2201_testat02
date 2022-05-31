//ZaK :-)

#include "image_processing.h"
#include <chrono>
using namespace std::chrono;

CImageProcessor::CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
}

CImageProcessor::~CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		delete m_proc_image[i];
	}
}

cv::Mat *CImageProcessor::GetProcImage(uint32 i)
{
	if (2 < i)
	{
		i = 2;
	}
	return m_proc_image[i];
}

int CImageProcessor::DoProcess(cv::Mat *image)
{

	auto start = high_resolution_clock::now();

	if (!image)
		return (EINVALID_PARAMETER);

	if (image->channels() > 1)
	{
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		colorImage = *image;
	}
	else
	{
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	//------------------------------------

	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0);
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1);

	*m_proc_image[0] = imgDx.clone();

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Time for step imgDx: " << duration.count() << " microseconds" << std::endl;

	start = high_resolution_clock::now();

	double threshold1 = 50;
	double threshold2 = 200;
	cv::Canny(grayImage, imgCanny, threshold1, threshold2, 3, true);

	*m_proc_image[1] = imgCanny.clone();

	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Time for step Canny image: " << duration.count() << " microseconds" << std::endl;

	start = high_resolution_clock::now();

	int rhoBins = 667;
	int alphaBins = 241;
	double RhoMin = -imgCanny.cols * cos(CV_PI / 4) - imgCanny.rows * sin(CV_PI / 4); // Minimal mögliches Rho im Bild
	double RhoMax = imgCanny.cols * cos(CV_PI / 4) + imgCanny.rows * sin(CV_PI / 4);  // Maximal mögliches Rho im Bild

	cv::Mat accImg = cv::Mat::zeros(rhoBins, alphaBins, CV_16U);

	std::vector<std::pair<int, int>> vecRowCol;
	std::vector<std::pair<int, int>> vecRhoAlpha;

	for (int rows = 0; rows < imgCanny.rows; rows++)
	{
		for (int cols = 0; cols < imgCanny.cols; cols++)
		{
			if (imgCanny.at<uint8_t>(rows, cols) != 0)
			{
				double dx = imgDx.at<int16_t>(rows, cols);
				double dy = imgDy.at<int16_t>(rows, cols);

				double alpha = atan2(dy, dx);
				double rho = (cols * cos(alpha) + rows * sin(alpha));

				int indexAlpha = round((alpha + CV_PI) / (2 * CV_PI) * alphaBins);
				int indexRho = round((rho - RhoMin) / (RhoMax - RhoMin) * rhoBins);

				if (indexRho >= rhoBins)
				{
					indexRho = rhoBins - 1;
				}
				if (indexAlpha >= alphaBins)
				{
					indexAlpha = alphaBins - 1;
				}
				accImg.at<int16_t>(indexRho, indexAlpha) += 1;

				vecRowCol.push_back(std::vector<std::pair<int, int>>::
										value_type(rows, cols));
				vecRhoAlpha.push_back(std::vector<std::pair<double, double>>::value_type(indexRho, indexAlpha));
			}
		}
	}

	cv::Size filterBlur(3, 3);
	cv::GaussianBlur(accImg, accImg, filterBlur, 1.5);

	double threshold = 5;
	cv::Mat binaryImage;
	cv::threshold(accImg, binaryImage, threshold, 255, cv::THRESH_BINARY);
	binaryImage.convertTo(binaryImage, CV_8U);

	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE,
					 kernel);

	cv::Mat stats, centroids, labelImage;
	connectedComponentsWithStats(binaryImage, labelImage,
								 stats, centroids);

	*m_proc_image[1] = binaryImage.clone();

	int nHoodRho = 4;
	int nHoodAlpha = 4;

	cv::Scalar rgbColor[6] = {cv::Scalar(255, 0, 0),
							  cv::Scalar(0, 255, 0),
							  cv::Scalar(0, 0, 255),
							  cv::Scalar(0, 255, 255),
							  cv::Scalar(255, 255, 0),
							  cv::Scalar(126, 47, 142)};

	for (int i1 = 1; i1 < stats.rows; i1++)
	{
		double cx = centroids.at<double>(i1, 0);
		double cy = centroids.at<double>(i1, 1);

		for (uint i0 = 0; i0 < vecRhoAlpha.size(); i0++)
		{
			if (fabs(vecRhoAlpha[i0].first - cy) < nHoodRho &&
				fabs(vecRhoAlpha[i0].second - cx) < nHoodAlpha)
			{
				cv::Point poi = cv::Point(vecRowCol[i0].second,
										  vecRowCol[i0].first);

				cv::line(colorImage, poi, poi,
						 rgbColor[i1 % 6], 3);
			}
		}
	}

	*m_proc_image[2] = colorImage.clone();
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Time for step color lines: " << duration.count() << " microseconds" << std::endl;

	return (SUCCESS);
}
