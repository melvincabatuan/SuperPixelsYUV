#include "com_cabatuan_superpixelsyuv_MainActivity.h"
#include <android/log.h>
#include <android/bitmap.h>
#include <stdlib.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "Superpixels.hpp"

using namespace cv;

#define  LOG_TAG    "SuperpixelsYUV"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)


void extractVU( Mat &image,  Mat &V, Mat &U){
   
    // accept only char type matrices
    CV_Assert(image.depth() != sizeof(uchar));

	int nRows = image.rows;   // number of lines
    int nCols = image.cols;   // number of columns  

    if (image.isContinuous()) {
        // then no padded pixels
        nCols = nCols * nRows;
		nRows = 1; // it is now a 1D array
	}   

    // for all pixels
    for (int j=0; j<nRows; j++) {

        // pointer to first column of line j
        uchar* data   = image.ptr<uchar>(j);
        uchar* colorV = V.ptr<uchar>(j);
        uchar* colorU = U.ptr<uchar>(j);

		for (int i = 0; i < nCols; i += 2) {
		           // process each pixel; assign to V and U alternately
                    *colorV++ =  *data++; // converts [0,255]  
                	*colorU++ =  *data++; // converts [0,255]     
        }
    }
}


/*       Global Variables        */

Mat imageU, imageV, VU, imageYUV; 
Mat scaledY; 
Mat mbgr; 


/*
 * Class:     com_cabatuan_superpixelsyuv_MainActivity
 * Method:    filter
 * Signature: (Landroid/graphics/Bitmap;[B)V
 */
JNIEXPORT void JNICALL Java_com_cabatuan_superpixelsyuv_MainActivity_filter
  (JNIEnv *pEnv, jobject clazz, jobject pTarget, jbyteArray pSource){

   AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent; // Links to Bitmap content

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   Mat srcNV21(bitmapInfo.height + bitmapInfo.height/2, bitmapInfo.width, CV_8UC1, (unsigned char *)source);
   Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);

/***********************************************************************************************/

   if (imageV.empty())
       imageV = Mat(bitmapInfo.height/2, bitmapInfo.width/2, CV_8UC1);

   if (imageU.empty())
       imageU = Mat(bitmapInfo.height/2, bitmapInfo.width/2, CV_8UC1);
 
   if (VU.empty())
       VU = srcNV21(cv::Rect( 0, bitmapInfo.height, bitmapInfo.width, bitmapInfo.height/2));

   //LOGI("VU.size() = [%d, %d]", VU.size().height, VU.size().width);

   extractVU( VU, imageV, imageU);
   
   // Scale down Luminance Y
   if (scaledY.empty())
        scaledY = Mat(bitmapInfo.height/2, bitmapInfo.width/2, CV_8UC1);

    pyrDown(srcNV21(cv::Rect(0,0,bitmapInfo.width,bitmapInfo.height)), scaledY);

    // merge YUV image
    std::vector<Mat> channels;

    channels.push_back(scaledY);
    channels.push_back(imageU);
    channels.push_back(imageV);

    if(imageYUV.empty())
       imageYUV = Mat(bitmapInfo.height/2, bitmapInfo.width/2, CV_8UC3);

    merge(channels, imageYUV);

    float before = static_cast<float>(getTickCount());
    Superpixels sp(imageYUV);    
    Mat boundaries = sp.viewSuperpixels();
    float after = static_cast<float>(getTickCount());

    if (mbgr.empty())
       mbgr = Mat(bitmapInfo.height, bitmapInfo.width, CV_8UC3);
    pyrUp(boundaries, mbgr);

    float duration_ms = 1000 * (after - before)/getTickFrequency();
    LOGI("Superpixels duration_ms = %0.2f ms", duration_ms);

    cvtColor(mbgr, mbgra, CV_BGR2BGRA);

/************************************************************************************************/ 
   
   /// Release Java byte buffer and unlock backing bitmap
   pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();

}
