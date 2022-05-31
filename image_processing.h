/*! @file image_processing.h
 * @brief Image Manipulation class
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "opencv.hpp"

#include "includes.h"
#include "camera.h"




class CImageProcessor {
public:
	CImageProcessor();
	~CImageProcessor();
	
	int DoProcess(cv::Mat* image);
	

	cv::Mat* GetProcImage(uint32 i);
	

private:
	cv::Mat* m_proc_image[3];/* we have three processing images for visualization available */

	cv::Mat m_prev_image;
};


#endif /* IMAGE_PROCESSING_H_ */
