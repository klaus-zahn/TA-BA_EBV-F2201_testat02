/*! @file image_processing.h
 * @brief Image Manipulation class
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "opencv.hpp"

#include "includes.h"
#include "camera.h"

typedef struct{
	int16_t CentRho;
	int16_t CentAlpha;
	int16_t AlphaThreshTop;
	int16_t AlphaThreshBottom;
	int16_t RhoThreshTop;
	int16_t RhoThreshBottom;
} MyAccStats;
typedef struct{
	int16_t ImX;
	int16_t ImY;
	int16_t Alpha;
	int16_t Rho;	
} MyAccPixel;

class CImageProcessor {
public:
	CImageProcessor();
	~CImageProcessor();
	
	int DoProcess(cv::Mat* image);

	cv::Mat* GetProcImage(uint32 i);

private:
	cv::Mat* m_proc_image[3];/* we have three processing images for visualization available */
};


#endif /* IMAGE_PROCESSING_H_ */
