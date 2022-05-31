/*! @file image_processing.h
 * @brief Image Manipulation class
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "opencv.hpp"

#include "includes.h"
#include "camera.h"

#include <cmath>



class CImageProcessor {
public:
	CImageProcessor();
	~CImageProcessor();
	
	int DoProcess(cv::Mat* image);

	cv::Mat* GetProcImage(uint32 i);

private:
	cv::Mat* m_proc_image[3];/* we have three processing images for visualization available */

	double deltaAlpha = M_PI / 180 * 1.5;	// Angle bin size in radiant
	double deltaRho	= 1.5;					// Rho bin size in pixel
	int64 startTicAcc, endTicAcc, startTicLine, endTicLine, startTic, endTic;	// Variables for time measurement
};


#endif /* IMAGE_PROCESSING_H_ */
