#include "opencv2/imgproc/imgproc.hpp" 
// Note: core.hpp is automatically with imgproc.hpp

#include <math.h>
//#include <map>


// m is introduced in Ds allowing us to control the compactness of superpixel.
// The greater the value of m, the more spatial proximity is emphasized and 
// the more compact the cluster. This value can be in the range [1, 20]. 
// Authors of the algorithm have chosen m = 10.

#define DEFAULT_M 20   // compactness parameter
//#define DEFAULT_M 10

#define DEFAULT_S -1   // window size (-1 means 'to be set...')


using namespace cv;


#ifndef SUPERPIXELS_H
#define SUPERPIXELS_H




class Superpixels{


public:

    Superpixels(Mat& img, float m = DEFAULT_M, float S = DEFAULT_S);     
    Mat viewSuperpixels(); // returns image displaying superpixel boundaries
    Mat colorSuperpixels(); // recolors image with average color in each cluster
    std::vector<Point> getCenters() const; // centers indexed by label
    Mat getLabels() const; // per pixel label
    ~Superpixels();


protected:

    // Member variables

    Mat img; // src original image
    Mat img_f; // scaled to [0,1]
    Mat img_lab; // converted to LAB colorspace

    // Store the calculated results
    Mat show;
    Mat labels; 
    
    float m; // compactness parameter
    float S; // window size

    int nx, ny; // cell cols and rows
    float dx, dy; // steps
    
    std::vector<Point> centers; // superpixel centers
    
    void calculateSuperpixels();
    float dist(Point p1, Point p2); // 5-D distance between pixels in LAB space
    const static Mat sobel;
};

#endif
