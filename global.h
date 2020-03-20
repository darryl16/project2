#include "myHeader.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
 
#include <iostream>
#include <ctype.h>
class global{
private:
    
public:
	void help();
	Mat keypointImage,keypointImage2,resultImg;
	Mat img;
	void  main();
	void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/);
	vector<KeyPoint> keypoints;
	Mat featuremap;
	Mat map;
};