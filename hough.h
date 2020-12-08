// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

#define MINRADIUS 20

void hough_main( std::string imageName );

void Sobel(
	cv::Mat &input, 
	int size,
	cv::Mat &magOutput,
    cv::Mat &xOutput,
    cv::Mat &yOutput,
    cv::Mat &dirOutput);

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

Mat threshold(
    cv::Mat &input,
    int thresh);

void houghCircle(
    Mat &input,
    Mat &gradient, 
    Mat &direction, 
    int ***houghSpace,
    int radius,
    const int MAXRADIUS);

void houghLines(Mat &input, 
    Mat &gradient, 
    Mat &direction);

void houghNaiveLines(Mat &input, Mat &gradient, Mat &direction);

void houghToImage(Mat &lineSpace, Mat &input);

int ***malloc3dArray(int dim1, int dim2, int dim3) {
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	    for (j = 0; j < dim2; j++) {
  	        array[i][j] = (int *) malloc(dim3 * sizeof(int));
	    }

    }
    return array;
}

int **malloc2dArray(int dim1, int dim2) {
    int i, j;
    int **array = (int **) malloc(dim1 * sizeof(int *));

    for (i = 0; i < dim1; i++) {
        array[i] = (int *) malloc(dim2 * sizeof(int));
    }
    return array;
}

void houghMain( std::string imageName ) {
    
    Mat image;
    image = imread( imageName, 1 );

    // CONVERT COLOUR, BLUR AND SAVE
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );

    Mat blurred;
    GaussianBlur(gray_image,8,blurred);

    Mat xSobel(image.rows,image.cols, CV_32FC1);
    Mat ySobel(image.rows,image.cols, CV_32FC1);
    Mat magSobel(image.rows,image.cols, CV_32FC1);
    Mat dirSobel(image.rows,image.cols, CV_32FC1);         
    Sobel(blurred, 3, magSobel, xSobel, ySobel, dirSobel);

    Mat resultX(image.rows,image.cols, CV_8UC1);
    Mat resultY(image.rows,image.cols, CV_8UC1);
    Mat resultMag(image.rows,image.cols, CV_8UC1);
    Mat resultDir(image.rows,image.cols, CV_8UC1);

    normalize(xSobel,resultX,0,255,NORM_MINMAX, CV_8UC1);
    normalize(ySobel,resultY,0,255,NORM_MINMAX, CV_8UC1);
    normalize(magSobel,resultMag,0,255,NORM_MINMAX);
    normalize(dirSobel,resultDir,0,255,NORM_MINMAX);
    imwrite("grad_x.jpg",resultX);
    imwrite("grad_y.jpg",resultY);
    imwrite("mag.jpg",resultMag);
    imwrite("dir.jpg", resultDir);

    const int MAXRADIUS = min(image.rows/4, image.cols/4);

    int ***houghSpace = malloc3dArray(image.rows, image.cols, MAXRADIUS);

    // loading in magnitude
    Mat mag = imread("mag.jpg", 1);
    Mat mag_gray;
    cvtColor( mag, mag_gray, CV_BGR2GRAY );
    // thresholding magnitude
    Mat thresh = threshold(mag_gray, 60);

    imwrite("mag_thresh.jpg", thresh);

    // finding circles 
    houghCircle(image, thresh, dirSobel, houghSpace, MAXRADIUS-MINRADIUS, MAXRADIUS);

    // finding lines
    houghLines(image, thresh, dirSobel);
    
    // loading in hough line transform
    Mat line_hough = imread("houghLineOuput.jpg", 1);

    // thresholding hough line transform 
    Mat line_hough_gray;
    cvtColor( line_hough, line_hough_gray, CV_BGR2GRAY );
    Mat thresh_line = threshold(line_hough_gray, 10);

    imwrite("FALO.jpg", thresh_line);

    // getting lines back
    houghToImage(thresh_line, image);
}

void Sobel(cv::Mat &input, int size, cv::Mat &magOutput, cv::Mat &xOutput, cv::Mat &yOutput, cv::Mat &dirOutput) {

    int kernelX[3][3] = {{-1, 0, 1},
                            {-2, 0, 2},
                            {-1, 0, 1}};

    int kernelY[3][3] = {{-1, -2, -1},
                            {0, 0, 0},
                            {1, 2, 1}};

    cv::Mat kernelXMat = cv::Mat(size, size, cv::DataType<int>::type, kernelX);
    cv::Mat kernelYMat = cv::Mat(size, size, cv::DataType<int>::type, kernelY);

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( size - 1 ) / 2;
	int kernelRadiusY = ( size - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			float sumX = 0.0;
            float sumY = 0.0;

			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					int kernalvalX = kernelXMat.at<int>(kernelx, kernely);
                    int kernalvalY = kernelYMat.at<int>(kernelx, kernely);

					// do the multiplication
					sumX += imageval * kernalvalX;		
                    sumY += imageval * kernalvalY;					
				}
			}
			// set the output value as the sum of the convolution
			xOutput.at<float>(i, j) = (float) sumX;
            yOutput.at<float>(i, j) = (float) sumY;

            magOutput.at<float>(i, j) = (float) sqrt((sumX * sumX) + (sumY * sumY));

            dirOutput.at<float>(i, j) = (float) atan2(sumY, sumX);
		}
	}
}

Mat threshold(cv::Mat &input, int thresh) {
    Mat threshed(input.rows, input.cols, CV_8UC1);

    for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
            int imageval = ( int ) input.at<uchar>( i, j );
            
            if (imageval > thresh) {
                threshed.at<uchar>( i, j ) = 255;
            } else {
                threshed.at<uchar>( i, j ) = 0;
            }
        }
    }
    imwrite("thresh.jpg", threshed);

    return threshed;
}

// Optimised hough line transform
void houghLines(Mat &input, Mat &gradient, Mat &direction) {

    // length of diagonal
    int rho = sqrt((gradient.rows*gradient.rows) + (gradient.cols * gradient.cols));
    int **houghSpace = malloc2dArray(rho, 360);

    // initalising hough space accumulator
    for (int i = 0; i < rho; i++) {
        for (int j = 0; j < 360; j++) {
            houghSpace[i][j] = 0;
        }
    }

    for (int x = 0; x < gradient.rows; x++) {
        for (int y = 0; y < gradient.cols; y++) {
            // if edge is found
            if (gradient.at<uchar>(x, y) == 255) {
                // get the angle from direction vector
                float theta = direction.at<float>(x,y);
                // convert to degrees for easier check on nearby angles for a small degree of error (in this case 5), also add 180 to keep all angles between 0-360
                int thetaDeg = (theta*180)/M_PI + 180;
                for (int deg = thetaDeg - 5; deg < thetaDeg + 6; deg++) {
                    // make sure degrees don't go above 360 or negative
                    int newDeg;
                    if (deg < 0) newDeg = deg + 360;
                    else if (deg >= 360) newDeg = deg % 360;
                    else newDeg = deg;
                    // convert back to radians for sin, cos calculation
                    float rad = (newDeg-180)*M_PI/180;
                    int p = (y * cos(rad)) + (x * sin(rad));
                    if (p >= 0 && p < rho) {
                        houghSpace[p][newDeg]++;
                    }
                }
            }
        }
    }
    std::cout << "passed" << std::endl;

    Mat houghSpaceOutput(rho, 360, CV_32FC1);

    for (int x = 0; x < rho; x++) {
        for (int y = 0; y < 360; y++) {
            houghSpaceOutput.at<float>(x,y) += houghSpace[x][y];  
        }
    }

    Mat houghSpaceConvert(rho, 360, CV_8UC1);
    normalize(houghSpaceOutput, houghSpaceConvert, 0, 255, NORM_MINMAX);

    imwrite( "houghLineOuput.jpg", houghSpaceConvert );
}

// https://stackoverflow.com/questions/28351804/hough-transform-converted-polar-coordinates-back-to-cartesian-but-still-cant
void houghToImage(Mat &lineSpace, Mat &input) {
    Mat output(input.rows, input.cols, CV_32FC1, Scalar(0)); 

    for (int rho = 0; rho < lineSpace.rows; rho++) {
        for (int theta = 0; theta < lineSpace.cols; theta++) {
            if (lineSpace.at<uchar>(rho,theta) == 255) {
                float thetaRad = (theta-180)*M_PI/180;

                float m = -cos(thetaRad)/sin(thetaRad);
                float c = rho/sin(thetaRad);

                for (int x = 0; x < output.cols; x++) {
                    int y = m*x + c;
                    if (y >= 0 && y < output.rows) {
                        //std::cout << output.at<float>(x, y) << std::endl;
                        output.at<float>(y, x)++;
                    }
                }
            }
        }
    }

    Mat houghSpaceConvert(input.rows, input.cols, CV_8UC1, Scalar(0));
    normalize(output, houghSpaceConvert, 0, 255, NORM_MINMAX);

    imwrite( "houghLines.jpg", houghSpaceConvert );
}

void houghNaiveLines(Mat &input, Mat &gradient, Mat &direction) {
    int rho = sqrt((gradient.rows*gradient.rows) + (gradient.cols * gradient.cols));
    int **houghSpace = malloc2dArray(rho, 180);

    for (int i = 0; i < rho; i++) {
        for (int j = 0; j < 180; j++) {
            houghSpace[i][j] = 0;
        }
    }

    for (int x = 0; x < gradient.rows; x++) {
        for (int y = 0; y < gradient.cols; y++) {
            if (gradient.at<uchar>(x, y) == 255) {
                for (int theta = 0; theta < 180; theta++) {
                    int p = (x * cos(theta*M_PI/180)) + (y * sin(theta*M_PI/180));
                    if (p >= 0) {
                        houghSpace[p][theta] ++;
                    }
                }
            }
        }
    }

    std::cout << "passed" << std::endl;

    Mat houghSpaceOutput(rho, 180, CV_32FC1);

    for (int x = 0; x < rho; x++) {
        for (int y = 0; y < 180; y++) {
                houghSpaceOutput.at<float>(x,y) += houghSpace[x][y];
        }
    }

    Mat houghSpaceConvert(rho, 180, CV_8UC1);
    normalize(houghSpaceOutput, houghSpaceConvert, 0, 255, NORM_MINMAX);

    imwrite( "houghNaiveOuput.jpg", houghSpaceConvert );
}

void houghCircle(Mat &input, Mat &gradient, Mat &direction, int ***houghSpace, int radius, const int MAXRADIUS) {
    for (int i = 0; i < gradient.rows; i++) {
        for (int j = 0; j < gradient.cols; j++) {
            for (int k = 0; k < MAXRADIUS; k++) {
                houghSpace[i][j][k] = 0;
            }
        }
    }


    for (int x = 0; x < gradient.rows; x++) {
        for (int y = 0; y < gradient.cols; y++) {
            if (gradient.at<uchar>(x, y) == 255) {
                for (int r = MINRADIUS; r < MAXRADIUS; r++) {
                    int b =  y - (r * cos(direction.at<float>(x, y)));
                    int a =  x - (r * sin(direction.at<float>(x, y)));

                    int d =  y + (r * cos(direction.at<float>(x, y)));
                    int c =  x + (r * sin(direction.at<float>(x, y)));
                    
                    if (b >= 0 && b < gradient.cols && a >= 0 && a < gradient.rows) {
                        houghSpace[a][b][r] += 1;
                    }
                    if (d >= 0 && d < gradient.cols && c >= 0 && c < gradient.rows) {
                        houghSpace[c][d][r] += 1;
                    }
                }
            }
        }
    }

    Mat houghSpaceOutput(gradient.rows, gradient.cols, CV_32FC1);

    for (int x = 0; x < gradient.rows; x++) {
        for (int y = 0; y < gradient.cols; y++) {
            for (int r = MINRADIUS; r < MAXRADIUS; r++) {
                houghSpaceOutput.at<float>(x,y) += houghSpace[x][y][r];
                if (houghSpace[x][y][r] > 20) {
                    circle(input, Point(y, x), r, Scalar(0, 255, 255), 2);
                }
            }
            
        }
    }

    Mat houghSpaceConvert(gradient.rows, gradient.cols, CV_8UC1);
    normalize(houghSpaceOutput, houghSpaceConvert, 0, 255, NORM_MINMAX);

    imwrite( "houghOutput.jpg", houghSpaceConvert );
    imwrite("output3.jpg", input);
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput) {
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}
