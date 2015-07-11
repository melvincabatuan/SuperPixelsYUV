#include "Superpixels.hpp"


// Sobel  y-direction 3x3 kernel
const Mat Superpixels::sobel = (Mat_<float>(3,3) << -1/16., -2/16., -1/16., 0, 0, 0, 1/16., 2/16., 1/16.);


// Constructor
Superpixels::Superpixels(Mat& img, float m, float S){
    this->img = img.clone();
    this->m = m;
    if(S == DEFAULT_S){
        //this->nx = 15; // cols
        //this->ny = 15; // rows
        this->nx = 7;
        this->ny = 7;
        this->dx = img.cols / float(nx); // x steps
        this->dy = img.rows / float(ny); // y steps
        this->S = (dx + dy + 1.0)/2; // default window size
    }
    else
        this->S = S;
        
    calculateSuperpixels();
}


Superpixels::~Superpixels(){

}


Mat Superpixels::viewSuperpixels(){    

    // Draw boundaries on original image
     std::vector<Mat> rgb(3);

     split(this->img_f, rgb);

     for (int i = 0; i < 3; i++){
        rgb[i] = rgb[i].mul(this->show);
     }
    
     Mat output = this->img_f.clone();

     merge(rgb, output);

     output = 255 * output;

     output.convertTo(output, CV_8UC3);
    
     return output;
}


Mat Superpixels::colorSuperpixels(){
    
    int n = nx * ny;
    std::vector<Vec3b> avg_colors(n);
    std::vector<int> num_pixels(n);
    
    std::vector<long> b(n), g(n), r(n);
    
    for(int y = 0; y < (int) labels.rows; ++y){
        for(int x = 0; x < (int) labels.cols; ++x){

            Vec3b pix = img.at<Vec3b>(y, x);
            int lbl = labels.at<int>(y, x);
            
            b[lbl] += (int) pix[0];
            g[lbl] += (int) pix[1];
            r[lbl] += (int) pix[2];
            
            ++num_pixels[lbl];
        }
    }

    for(int i = 0; i < n; ++i){
        int num = num_pixels[i];
        avg_colors[i] = Vec3b(b[i] / num, g[i] / num, r[i] / num);
    }
    
    Mat output = this->img.clone();
    for(int y = 0; y < (int) output.rows; ++y){
        for(int x = 0; x < (int) output.cols; ++x){
            int lbl = labels.at<int>(y, x);
            if(num_pixels[lbl])
                output.at<Vec3b>(y, x) = avg_colors[lbl];
        }
    }
    
    return output;
}


std::vector<Point> Superpixels::getCenters() const{
    return centers;
}


Mat Superpixels::getLabels() const{
    return labels;
}


void Superpixels::calculateSuperpixels(){

    Mat temp_rgb;

    // #TODO
    cvtColor(this->img, temp_rgb, CV_YUV2BGR);

    // Scale img to [0,1] CV_32F
    temp_rgb.convertTo(this->img_f, CV_32F, 1/255.);

    // Convert to l-a-b colorspace
    // cvtColor(this->img_f, this->img_lab, CV_BGR2Lab);

    int n = nx * ny;  // total labels per cell; e.x. 15*15 = 225
    int w = img.cols; // img width
    int h = img.rows; // img height
    
        // Initialize cluster/cell centers
	for (int i = 0; i < ny; i++) {    // i cell rows
		for (int j = 0; j < nx; j++) { // j cell cols
		      this->centers.push_back( Point2f(j*dx+dx/2, i*dy+dy/2));
		}
	}

	// Initialize labels and distance maps
	std::vector<int> label_vec(n);

       //for (int i = 0; i < n; i++)
       //      label_vec[i] = i*255*255/n; // Why? {0, 289, 578, ..., 64447, 64736}

        for (int i = 0; i < n; i++)
             label_vec[i] = i;

	Mat labels = -1 * Mat::ones(this->img_f.size(), CV_32S); // -1 for uninitialized
	Mat dists  = -1 * Mat::ones(this->img_f.size(), CV_32F); // -1 for uninitialized

	Mat window;
	Point2i p1, p2;
	Vec3f p1_lab, p2_lab;

	// Iterate 10 times. In practice more than enough to converge
	for (int i = 0; i < 10; i++) {
	     // For each center...
	     for (int c = 0; c < n; c++)
             {
                int label = label_vec[c]; // ?

                // Current center
                p1 = centers[c]; 

                // Localize the search to 2S x 2S cell/box
                int xmin = max<int>(p1.x - S, 0);
                int ymin = max<int>(p1.y - S, 0);
                int xmax = min<int>(p1.x + S, w - 1);
                int ymax = min<int>(p1.y + S, h - 1);

                // Search in a 2S x 2S window around the center
                window = this->img_f(Range(ymin, ymax), Range(xmin, xmax));
			
                // Reassign pixels to nearest center
                for (int i = 0; i < window.rows; i++) {
                    for (int j = 0; j < window.cols; j++) {
                        p2 = Point2i(xmin + j, ymin + i);    // current point for clustering
                        float d = dist(p1, p2);             // distance from current center 
                        float last_d = dists.at<float>(p2);// previous distance entry 

                        if (d < last_d || last_d == -1) {   // update distance if less than previous
                                                            // or uninitialized (-1)

                            dists.at<float>(p2) = d;       // update distance
                            labels.at<int>(p2) = label;    // update label 

                        }
                    }
                }
            }
	}

    // Store the labels for each pixel
    this->labels = labels.clone();
    //this->labels =  n * this->labels / (255 * 255);

    // Calculate superpixel boundaries
    labels.convertTo(labels, CV_32F);

    Mat gx, gy, grad;

    filter2D(labels, gx, -1, sobel);
    filter2D(labels, gy, -1, sobel.t());

    magnitude(gx, gy, grad);

    grad = (grad > 1e-4)/255;

    Mat show = 1 - grad;

    show.convertTo(show, CV_32F);
        
    // Store the result
    this->show = show.clone();
}


float Superpixels::dist(Point p1, Point p2){
    Vec3f p1_lab = this->img.at<Vec3f>(p1);
    Vec3f p2_lab = this->img.at<Vec3f>(p2);
    
    float dl = p1_lab[0] - p2_lab[0];
    float da = p1_lab[1] - p2_lab[1];
    float db = p1_lab[2] - p2_lab[2];

    // lab distance, d_lab
    float d_lab = sqrtf(dl*dl + da*da + db*db);

    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;

    // xy plane euclidean distance, d_xy
    float d_xy = sqrtf(dx*dx + dy*dy);

    // Ds is the sum of the lab distance and the xy plane distance 
    // normalized by the grid interval S.
    return d_lab + m/S * d_xy;  
}
    // m is introduced in Ds allowing us to control the compactness of superpixel.
    // The greater the value of m, the more spatial proximity is emphasized and 
    // the more compact the cluster. This value can be in the range [1, 20]. 
    // Authors of the algorithm have chosen m = 10.


