/*! @file image_processing.h
 * @brief Image Manipulation class
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "opencv.hpp"

#include "includes.h"
#include "camera.h"
#include <stdio.h>      /* printf */
#include <math.h>       /* atan2 */


class CImageProcessor {
public:
	CImageProcessor();	// Konstruktor braucht es immer
	~CImageProcessor();	// Destructor auch, brauchts wenn gelöscht wird
	
	int DoProcess(cv::Mat* image);	//Methode

	cv::Mat* GetProcImage(uint32 i);	//Methode

private:
	cv::Mat m_prev_image;	//Member
	cv::Mat* m_proc_image[3];/* we have three processing images for visualization available */
		
};


#endif /* IMAGE_PROCESSING_H_ */
