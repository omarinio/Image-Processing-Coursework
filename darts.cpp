/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - f1_darts.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <sstream>
#include <fstream>

#include "hough.h"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, string num, string file );
vector<string> split( const std::string &line, char delimiter );
float findIOU( Rect found, Rect truth );
float findF1Score( float true_positives, float false_positives, float false_negatives );
vector<Rect> readFile( string num );

/** Global variables */
String cascade_name = "dartboard.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ) {
    string num = argv[1];

	if (num == "all") {
		for (int i = 0; i < 16; i++) {
			string file = "images/dart" + to_string(i) + ".jpg";
	
			Mat frame = imread(file, CV_LOAD_IMAGE_COLOR);

			// 2. Load the Strong Classifier in a structure called `Cascade'
			if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

			// 3. Detect Faces and Display Result
			detectAndDisplay( frame, to_string(i), file );

			// 4. Save Result Image
			imwrite( "detected" + to_string(i) + ".jpg", frame );
		}
	} else {
		string file = "images/dart" + num + ".jpg";
       
		Mat frame = imread(file, CV_LOAD_IMAGE_COLOR);


		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

		// 3. Detect Faces and Display Result
		detectAndDisplay( frame, num, file );

		// 4. Save Result Image
		imwrite( "detected" + num + ".jpg", frame );
	}

	return 0;
}

// split function taken from computer graphics coursework
vector<string> split( const std::string &line, char delimiter ) {
	auto haystack = line;
	std::vector<std::string> tokens;
	size_t pos;
	while ((pos = haystack.find(delimiter)) != std::string::npos) {
		tokens.push_back(haystack.substr(0, pos));
		haystack.erase(0, pos + 1);
	}
	// Push the remaining chars onto the vector
	tokens.push_back(haystack);
	return tokens;
}

vector<Rect> readFile( string num ) {
    // load in file
    string filename = "dartboard_coordinates/" + num + ".csv";
    std::ifstream infile(filename);
    string line;
    // initalise empty vector for output
    vector<Rect> truths;

    while(std::getline(infile, line)) {
        // split each line by the delimiter (',') 
        vector<string> tokens = split(line, ',');
        // store each value in a Rect
        Rect temp(stoi(tokens[0]), stoi(tokens[1]), stoi(tokens[2]), stoi(tokens[3]));
        // add to vector
        truths.push_back(temp);
    }

    infile.close();

    return truths;
}

// https://medium.com/koderunners/intersection-over-union-516a3950269c
float findIOU( Rect found, Rect truth ) {
	float width = min(found.x + found.width, truth.x + truth.width) - max(found.x, truth.x);
	float height = min(found.y + found.height, truth.y + truth.height) - max(found.y, truth.y);

	// no overlap
	if (width <= 0 || height <= 0) return 0;

	float intersection = width * height;
	float uni = (found.width * found.height) + (truth.width * truth.height) - intersection;

	return intersection/uni;
}

// formula from: https://en.wikipedia.org/wiki/F-score 
float findF1Score( float true_positives, float false_positives, float false_negatives ) {
	if (true_positives == 0 && false_positives == 0 && false_negatives == 0) return 0;

	float f1_score = true_positives/(true_positives + 0.5*(false_positives+false_negatives));

	return f1_score;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, string num, string file ) {
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

    vector<Rect> truths = readFile(num);

    // 5. Draw ground truth
    for (int j = 0; j < truths.size(); j++) {
        rectangle(frame, Point(truths[j].x, truths[j].y), Point(truths[j].x + truths[j].width, truths[j].y + truths[j].height), Scalar(0,0,255), 2);
    }

	float iou_threshold = 0.4;
	float true_positives = 0;

	houghMain(file);

	Mat houghCircles = imread("houghOutput.jpg", 1);
	Mat houghLines = imread("houghLines.jpg", 1);

	Mat linesGray;
    cvtColor( houghLines, linesGray, CV_BGR2GRAY );
    // hough lines threshold
    Mat threshLines = threshold(linesGray, 140);

	Mat circlesGray;
	cvtColor( houghCircles, circlesGray, CV_BGR2GRAY );
	// hough circles threshold
	Mat threshCircles = threshold(circlesGray, 140);

	/* 
	 *	   FIND AVERAGE OF FOUND LINES AND CIRCLES
	 * 	   FIND IOU OF ALL OTHER RECTANGLES
	 * 	   IF OVERLAP, FIND AVERAGE WIDTH AND HEIGHT AND CENTRE OF RECTANGLES AND DRAW A NEW ONE BASED ON THOSE
	 */ 

	// Rect is defined as (x,y,width,height)
	std::vector<Rect> found;

	for( int i = 0; i < faces.size(); i++ ) {
		int counter = 0;
		for (int x = faces[i].x; x <= faces[i].x + faces[i].width; x++) {
			for (int y = faces[i].y; y <= faces[i].y + faces[i].height; y++) {
				// counts up how many centre pixels are found
				if (threshLines.at<uchar>(y, x) == 255) {
					counter++;
				}
				if (threshCircles.at<uchar>(y, x) == 255) {
					counter++;
				}
			}
		}

		// if above threshold, add to list of rectangles
		if (counter >= 80) {
			// check if new rectangle overlaps with any others
			bool isFound = false;
			Rect temp;
			for (int j = 0; j < found.size(); j++) {
				float iou = findIOU(faces[i], found[j]);
				// if overlap is found, break out of for loop and remove the rectangle from the vector
				if (iou > 0) {
					isFound = true;
					temp = found[j];
					found.erase(found.begin() + j);
					break;
				}
			}
			// if intersecting rectangles found, find new rectangle which is the average of the 2 found
			if (isFound == true) {
				int avgX = (temp.x + faces[i].x) / 2;
				int avgY = (temp.y + faces[i].y) / 2;
				int avgWidth = (temp.width + faces[i].width) / 2;
				int avgHeight = (temp.height + faces[i].height) / 2;

				// create new rectangle and push back to vector
				Rect avg(avgX, avgY, avgWidth, avgHeight);
				found.push_back(avg);
			} else {
				// if no overlap found, simply push back new rectangle into found vector
				found.push_back(faces[i]);
			}

		}
	}

	float true_positive_rate;

		for (int x = 0; x < truths.size(); x++) {
			for (int y = 0; y < found.size(); y++) {
				float iou = findIOU(found[y], truths[x]);
				std::cout << iou << std::endl;
				if (iou > iou_threshold) {
					true_positives++;
					break;
				}
			}
		}

	cout << "IMAGE " << num << endl;

	if (truths.size() > 0){
		true_positive_rate = true_positives/(float)truths.size();
	} else {
		true_positive_rate = 0;
	}

	cout << "TPR: " << true_positive_rate << endl;
	cout << "true positives: " << true_positives << endl;

	float false_positives = found.size() - true_positives;

	cout << "false positives: " << false_positives << endl;

	float false_negatives = truths.size() - true_positives;

	cout << "false negatives: " << false_negatives << endl;

	float f1_score = findF1Score(true_positives, false_positives, false_negatives);

	cout << "f1: " << f1_score << endl;

	// displays found triangles
	for (int i = 0; i < found.size(); i++) {

		rectangle(frame, Point(found[i].x, found[i].y), Point(found[i].x + found[i].width, found[i].y + found[i].height), Scalar( 0, 255, 0 ), 2);
	}
}
