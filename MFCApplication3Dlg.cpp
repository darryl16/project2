
// MFCApplication3Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "global.h"
#include "MFCApplication3.h"
#include "MFCApplication3Dlg.h"
#include "afxdialogex.h"
#include "myHeader.h"



#ifdef _DEBUG
#define new DEBUG_NEW
#endif

global myglobal;
// 對 App About 使用 CAboutDlg 對話方塊
 void onMouse(int Event,int x,int y,int flags,void* param);
int count1;
int count2;
 vector<vector<Point2f>>pointall;

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 對話方塊資料
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCApplication3Dlg 對話方塊



CMFCApplication3Dlg::CMFCApplication3Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CMFCApplication3Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFCApplication3Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMFCApplication3Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON2, &CMFCApplication3Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CMFCApplication3Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON1, &CMFCApplication3Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON5, &CMFCApplication3Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CMFCApplication3Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON4, &CMFCApplication3Dlg::OnBnClickedButton4)
	//ON_BN_CLICKED(IDC_BUTTON7, &CMFCApplication3Dlg::OnBnClickedButton7)
END_MESSAGE_MAP()


// CMFCApplication3Dlg 訊息處理常式

BOOL CMFCApplication3Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	AllocConsole();
	freopen ("CONOUT$", "w", stdout );

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void CMFCApplication3Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void CMFCApplication3Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR CMFCApplication3Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CMFCApplication3Dlg::OnBnClickedButton2()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	/*
	initModule_nonfree();//if use SIFT or SURF
	Ptr<FeatureDetector> detector= FeatureDetector::create( "SIFT" );
	 
	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );
	if( detector.empty() || descriptor_extractor.empty() )
		throw runtime_error("fail to create detector!");
 
	Mat img1 = imread("Bird1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("Bird2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
 
	//detect keypoints;
	vector<KeyPoint> keypoints1,keypoints2;
	detector->detect( img1, keypoints1 );
	detector->detect( img2, keypoints2 );
	
	cout <<"img1:"<< keypoints1.size() << " points  img2:" <<keypoints2.size() 
		<< " points" << endl << ">" << endl;
 
	//compute descriptors for keypoints;
	cout << "< Computing descriptors for keypoints from images..." << endl;
	Mat descriptors1,descriptors2;
	descriptor_extractor->compute( img1, keypoints1, descriptors1 );
	descriptor_extractor->compute( img2, keypoints2, descriptors2 );

 
	cout<<endl<<"Descriptors Size: "<<descriptors2.size()<<" >"<<endl;
	cout<<endl<<"Descriptor's Column: "<<descriptors2.cols<<endl
		<<"Descriptor's Row: "<<descriptors2.rows<<endl;
	cout << ">" << endl;
	Mat img_keypoints1,img_keypoints2;
	drawKeypoints(img1,keypoints1,img_keypoints1,Scalar::all(-1),0);
	drawKeypoints(img2,keypoints2,img_keypoints2,Scalar::all(-1),0);
	imshow("FeatureBird1.jpg",img_keypoints1);
	imshow("FeatureBird2.jpg",img_keypoints2);
	descriptor_extractor->compute( img1, keypoints1, descriptors1 );  
	vector<DMatch> matches;
	descriptor_matcher->match( descriptors1, descriptors2, matches );
 
	Mat img_matches;
	drawMatches(img1,keypoints1,img2,keypoints2,matches,img_matches,Scalar::all(-1),CV_RGB(255,255,255),Mat(),4);
 
	imshow("Match",img_matches);
	
	*/

	//成功取得5個點
/*	
vector<KeyPoint> keypoints,keypoints2;

//The SIFT feature extractor and descriptor
//SiftDescriptorExtractor detector;   
SiftDescriptorExtractor detector(5);
Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );
Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );
Mat input,input2;    
Mat feature,feature2;

//open the file
input = imread("Bird1.jpg", 0); 
input2 = imread("Bird2.jpg", 0);
//detect feature points
detector.detect(input, keypoints);
detector.detect(input2, keypoints2);

cout <<"img1:"<< keypoints.size() << " points  img2:" <<keypoints2.size() 
		<< " points" << endl << ">" << endl;
///Draw Keypoints
Mat keypointImage,keypointImage2;
keypointImage.create( input.rows, input.cols, CV_8UC3 );
keypointImage2.create( input2.rows, input2.cols, CV_8UC3 );
drawKeypoints(input, keypoints, keypointImage, Scalar::all(-1),0);
drawKeypoints(input2, keypoints2, keypointImage2, Scalar::all(-1),0);
imshow("FeatureBird1.jpg", keypointImage);
imshow("FeatureBird2.jpg", keypointImage2);

//Matched feature points

Mat descriptors1,descriptors2;
	descriptor_extractor->compute( keypointImage, keypoints, descriptors1 );
	descriptor_extractor->compute( keypointImage2, keypoints2, descriptors2 );
descriptor_extractor->compute( keypointImage, keypoints, descriptors1 );  
	vector<DMatch> matches;
	descriptor_matcher->match( descriptors1, descriptors2, matches );
 
	Mat img_matches;
	drawMatches(keypointImage,keypoints,keypointImage2,keypoints2,matches,img_matches,Scalar::all(-1),0,Mat());
 
	imshow("Match",img_matches);*/
	
 
	Mat imgObject = imread( "Bird1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	Mat imgScene = imread( "Bird2.jpg", CV_LOAD_IMAGE_GRAYSCALE );
 
	if( !imgObject.data || !imgScene.data )
	{ 
		cout<< " --(!) Error reading images "<<endl;
		//return -1; 
	}
 
	//double begin = clock();
 
	///-- Step 1: 使用SIFT算子检测特征点
	//int minHessian = 400;
	double threshold=500;
	double edgeThreshold=500;
		 SiftFeatureDetector detector;//( minHessian );
	 


	vector<KeyPoint> keypointsObject, keypointsScene,goodpointsobject,goodpointsScene,testhomography;
	detector.detect( imgObject, keypointsObject );
	detector.detect( imgScene, keypointsScene );
	cout<<"object--number of keypoints: "<<keypointsObject.size()<<endl;
	cout<<"scene--number of keypoints: "<<keypointsScene.size()<<endl;
	
	///-- Step 2: 使用SIFT算子提取特征（计算特征向量）
	SiftDescriptorExtractor extractor;
	Mat descriptorsObject, descriptorsScene,descriptorsObjecthomography;
	extractor.compute( imgObject, keypointsObject, descriptorsObject );
	extractor.compute( imgScene, keypointsScene, descriptorsScene );
  
	///-- Step 3: 使用FLANN法进行匹配
	FlannBasedMatcher matcher;
	vector< DMatch > allMatches,allmatches2;
	matcher.match( descriptorsObject, descriptorsScene, allMatches );
	matcher.match( descriptorsScene,descriptorsObject,  allmatches2 );
	cout<<"number of matches before filtering: "<<allMatches.size()<<endl;

	//-- 计算关键点间的最大最小距离
	double maxDist = 0,maxDist2 = 0;
	double minDist=95,minDist2 = 95;
	for( int i = 0; i < descriptorsObject.rows; i++ )
	{
		double dist = allMatches[i].distance;
		if( dist < minDist )
			minDist = dist;
		if( dist > maxDist )
			maxDist = dist;
	}
	for( int i = 0; i < descriptorsScene.rows; i++ )
	{
		double dist = allmatches2[i].distance;
		if( dist < minDist2 )
			minDist2 = dist;
		if( dist > maxDist2 )
			maxDist2 = dist;
	}
	printf("	max dist : %f \n", maxDist );
	printf("	min dist : %f \n", minDist );
 
	//-- 过滤匹配点，保留好的匹配点（这里采用的标准：distance<3*minDist）
	vector< DMatch > goodMatches;
	for( int i = 0; i < descriptorsObject.rows; i++ )
	{
		if( allMatches[i].distance < 2*minDist2 ){
			goodMatches.push_back( allMatches[i]); 
		goodpointsobject.push_back( keypointsObject[i]); 
		
			}
	}
	
	testhomography.push_back( keypointsObject[114]);
	cout<<descriptorsObject.row(114)<<endl;
	descriptorsObjecthomography.push_back( descriptorsObject.row(309));//save特徵點的值
	
	for( int i = 0; i < descriptorsScene.rows; i++ )
	{
		if( allmatches2[i].distance < 2*minDist ){
			
	
		goodpointsScene.push_back(keypointsScene[i]);
			}
	}
	cout<<"number of matches after filtering: "<<goodMatches.size()<<endl;
 
	//-- 显示匹配结果
	//Mat resultImg;
	drawMatches( imgObject, keypointsObject, imgScene, keypointsScene, 
		goodMatches, myglobal.resultImg, Scalar::all(-1), Scalar::all(-1), vector<char>(), 
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS //不显示未匹配的点
		); 

	//-- 输出匹配点的对应关系
	for( int i = 0; i < goodMatches.size(); i++ ){
		printf( "	good match %d: keypointsObject [%d]  -- keypointsScene [%d]\n", i, 
		goodMatches[i].queryIdx, goodMatches[i].trainIdx );
	///--homography
	  //cout<<descriptorsObject.queryIdx<<endl;
		//cout<<(goodpointsobject[i].pt.x,goodpointsobject[i].pt.y)<<endl;
		//cout<<(goodpointsScene[i].pt.x,goodpointsScene[i].pt.y)<<endl;
	}
	
	///-- Step 4: 使用findHomography找出相应的透视变换
	vector<Point2f> object;
	vector<Point2f> scene;
	for( size_t i = 0; i < goodMatches.size(); i++ )
	{
		//-- 从好的匹配中获取关键点: 匹配关系是关键点间具有的一 一对应关系，可以从匹配关系获得关键点的索引
		//-- e.g. 这里的goodMatches[i].queryIdx和goodMatches[i].trainIdx是匹配中一对关键点的索引
		object.push_back( keypointsObject[ goodMatches[i].queryIdx ].pt );
		scene.push_back( keypointsScene[ goodMatches[i].trainIdx ].pt ); 
	}
	Mat H = findHomography( object, scene, CV_RANSAC );
	//homography
	Mat keypointImage,keypointImage2;
keypointImage.create( imgObject.rows, imgObject.cols, CV_8UC3 );
//drawKeypoints(imgObject, goodpointsobject, keypointImage, Scalar::all(-1),0);
drawKeypoints(imgObject, goodpointsobject, keypointImage, Scalar::all(-1),0);
//circle(keypointImage, Point(179.9,114.0), 5, Scalar(255,0,0), 1, 8);
imshow("test.jpg", keypointImage);
//imwrite("test.jpg", myglobal.keypointImage);
keypointImage2.create( imgScene.rows, imgScene.cols, CV_8UC3 );
drawKeypoints(imgScene, goodpointsScene,keypointImage2, Scalar::all(-1),0);
/*
for(int i = 0 ; i<keypointsObject.size() ; i++)
{
	cout << keypointsObject[i].pt << endl;
	cout<<i<<endl;
}*/

imshow("test2.jpg", keypointImage2);
//imwrite("test2.jpg", myglobal.keypointImage2);
	//-- 显示检测结果
	//imshow("detection result", myglobal.resultImg );
	//imwrite("detection result", myglobal.resultImg);
	 Mat img(500, 500, CV_8UC3, Scalar(255,255,255));
	  line(img, Point(10,300), Point(394,300), 0, 1);
	   line(img, Point(10,300), Point(10,10), 0, 1);
	    line(img, Point(10,10), Point(394,10), 0, 1);
		 line(img, Point(394,300), Point(394,10), 0, 1);
	 for(int i=0;i<128;i++){
    line(img, Point(10+3*i,300), Point(10+3*i,300-descriptorsObjecthomography.at<float>(i)), Scalar(255,0,0), 3);
	 //line(img, Point(25,400), Point(25,400-descriptorsObjecthomography.at<float>(21)), Scalar(255,0,0), 3);
	}
	   putText(img,"0" ,Point(10,310),FONT_HERSHEY_COMPLEX,0.3,0 );
	 putText(img,"20" ,Point(70,310),FONT_HERSHEY_COMPLEX,0.3,0 );
	    putText(img,"40" ,Point(130,310),FONT_HERSHEY_COMPLEX,0.3,0 );
		 putText(img,"60" ,Point(190,310),FONT_HERSHEY_COMPLEX,0.3,0 );
		  putText(img,"80" ,Point(250,310),FONT_HERSHEY_COMPLEX,0.3,0 );
		   putText(img,"100" ,Point(320,310),FONT_HERSHEY_COMPLEX,0.3,0 );
		    putText(img,"120" ,Point(370,310),FONT_HERSHEY_COMPLEX,0.3,0 );
			putText(img,"0" ,Point(0,300),FONT_HERSHEY_COMPLEX,0.3,0 );
	 putText(img,"25" ,Point(0,275),FONT_HERSHEY_COMPLEX,0.3,0 );
	    putText(img,"50" ,Point(0,250),FONT_HERSHEY_COMPLEX,0.3,0 );
		 putText(img,"75" ,Point(0,225),FONT_HERSHEY_COMPLEX,0.3,0 );
		  putText(img,"100" ,Point(0,200),FONT_HERSHEY_COMPLEX,0.3,0 );
		   putText(img,"125" ,Point(0,175),FONT_HERSHEY_COMPLEX,0.3,0 );
		    putText(img,"150" ,Point(0,150),FONT_HERSHEY_COMPLEX,0.3,0 );
	// imshow("window", img);
	
	circle(imgObject, Point(179.9,114.0), 5, Scalar(255,0,0), 1, 8);
	myglobal.featuremap=imgObject;
	myglobal.map=img;
	
}


void CMFCApplication3Dlg::OnBnClickedButton3()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	imshow("feature point", myglobal.featuremap);
	imshow("detection result", myglobal.map );
}


void CMFCApplication3Dlg::OnBnClickedButton1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	imshow("detection result", myglobal.resultImg );
	
}

 Point2f point;
bool addRemovePt = false;
void CMFCApplication3Dlg::OnBnClickedButton5()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	    myglobal.help();

/*
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10, 10), winSize(21, 21);
 
    //const int MAX_COUNT = 500;
	const int MAX_COUNT = 7;
    bool needToInit = false;
    bool nightMode = false;
+
 
    cap.open("featureTracking.mp4");
 
    if (!cap.isOpened())
    {
        cout << "Could notinitialize capturing...\n";
        
    }
 
    namedWindow("LK", 1);
    setMouseCallback("LK", onMouse,0);
 
    Mat gray, prevGray, image;
    vector<Point2f> points[2];
 
    for (;;)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
 
        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);
 
        if (nightMode)
            image = Scalar::all(0);
 
        if (needToInit)
        {
            // automaticinitialization
            goodFeaturesToTrack(gray, points[1],100, 0.01, 10, Mat(), 3, 0, 0.04);
            cornerSubPix(gray, points[1],subPixWinSize, Size(-1, -1), termcrit);
            addRemovePt = false;
        }
        else if(!points[0].empty())
        {
            vector<uchar> status;
            vector<float> err;
            if (prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray,points[0], points[1], status, err, winSize,
                3, termcrit, 0, 0.001);
            size_t i, k;
            for (i = k = 0; i <points[1].size(); i++)
            {
                if (addRemovePt)
                {
                    if (norm(point -points[1][i]) <= 5)
                    {
                        addRemovePt = false;
                        continue;
                    }
                }
 
                if (!status[i])
                    continue;
 
                points[1][k++] = points[1][i];
                circle(image, points[1][i], 3, Scalar(0, 0 ,255), -1, 8);
				line( image, points[1][i], points[1][i], CV_RGB(255,0,0),2 );
				
            }
            points[1].resize(k);
        }
 
        if (addRemovePt&& points[1].size() < (size_t)MAX_COUNT)
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix(gray, tmp, winSize,cvSize(-1, -1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }
 
        needToInit = false;
        imshow("LK", image);
 
        char c = (char)waitKey(100);
        if (c == 27)
            break;
        switch (c)
        {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        }
 
        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
	}
*/
	 // TODO: Add your control notification handler code here
    Mat image1, image2;
    vector<Point2f> point1, point2, pointCopy;
	
    vector<uchar> status;
    vector<float> err;
	vector<KeyPoint> keypoints;

    for(int i=0; i<myglobal.keypoints.size(); i++){

		point1.push_back(myglobal.keypoints[i].pt);

			}
	


    VideoCapture video("featureTracking.mp4");
    video >> image1;
	cv::SimpleBlobDetector::Params params2;
			params2.minDistBetweenBlobs = 1.0f;
			params2.filterByInertia = true;
			params2.minInertiaRatio = 0.4;
			params2.filterByConvexity = true;
			params2.minConvexity = 0.87;
			params2.filterByColor = true;
			params2.blobColor = 0;
			params2.filterByCircularity = 0.83;
			params2.filterByArea = true;
			params2.minArea = 30.0f;
			params2.maxArea = 500.0f;

			// Set up the detector with default parameters.
			SimpleBlobDetector detector2(params2);
    Mat image1Gray, image2Gray;
    cvtColor(image1, image1Gray, CV_RGB2GRAY);

    //goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
    pointCopy = point1;
    for (int i = 0; i < point1.size(); i++)    //绘制特征点位  
    {
        circle(image1, point1[i], 1, Scalar(0, 0, 255), 2);
    }
    namedWindow("光流特征圖");
    while (true)
    {
		
        video >> image2;
		
        if (waitKey(33) == ' ')  //按下空格选择当前画面作为标定图像  
		{
           // cvtColor(image2, image1Gray, CV_RGB2GRAY);
            //goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
			 cvtColor(image2, image1Gray, CV_RGB2GRAY);
                //goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
				//Mat im = image1Gray;
 
			
 


			// Detect blobs.
			//std::vector<KeyPoint> keypoints;
			Mat imtemt=image2;
			detector2.detect( imtemt, keypoints);
			   
			point1.erase(point1.begin(),point1.end());
			
			  for(int i=0; i<keypoints.size(); i++){

			point1.push_back(keypoints[i].pt);

			}
            pointCopy = point1;
			cout<<"fuckkkkkkkkkkkk"<<endl;
		}
        cvtColor(image2, image2Gray, CV_RGB2GRAY);
        calcOpticalFlowPyrLK(image1Gray, image2Gray, point1, point2, status, err, Size(11, 11), 3); //LK金字塔       
        int tr_num = 0;
        vector<unsigned char>::iterator status_itr = status.begin();
        while (status_itr != status.end()) {
            if (*status_itr > 0)
                tr_num++;
            status_itr++;
        }
        if (tr_num < 6) {
            cout << "you need to change the feat-img because the background-img was all changed" << endl;
            if (waitKey(0) == ' ') {
                cvtColor(image2, image1Gray, CV_RGB2GRAY);
                //goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());

                pointCopy = point1;
            }
        }
        for (int i = 0; i < point2.size(); i++)
        {
            circle(image2, point2[i], 1, Scalar(0, 0, 255), 6);

			
			//cout<<i<<endl;
            //line(image2, pointCopy[i], point2[i], Scalar(255, 0, 0), 1,CV_AA);
        }

	
		count1++;
		if( (count1%3) ==0)
		{
			pointall.push_back(point2) ;
			count2++;
		}
		
		
		if( count2 >= 2)
		{
			for (int i = 0; i < count2 ; i++)
			{

				for( int j = 0 ; j < i ; j++ )
				{
					line(image2, pointall[j][0], pointall[j+1][0], Scalar(0, 97, 255), 1,CV_AA);
					line(image2, pointall[j][1], pointall[j+1][1], Scalar(0, 97, 255), 1,CV_AA);
					line(image2, pointall[j][2], pointall[j+1][2], Scalar(0, 97, 255), 1,CV_AA);
					line(image2, pointall[j][3], pointall[j+1][3], Scalar(0, 97, 255), 1,CV_AA);
					line(image2, pointall[j][4], pointall[j+1][4], Scalar(0, 97, 255), 1,CV_AA);
					line(image2, pointall[j][5], pointall[j+1][5], Scalar(0, 97, 255), 1,CV_AA);
					line(image2, pointall[j][6], pointall[j+1][6], Scalar(0, 97, 255), 1,CV_AA);
					//line(image2, pointall[j][7], pointall[j+1][7], Scalar(0, 97, 255), 1,CV_AA);
	
				}
			
			}
		}
		
		cout << count1 << endl ;

		


		//cout<<pointall<<endl;

        imshow("光流特征图", image2);
        swap(point1, point2);
        image1Gray = image2Gray.clone();
		
    }

}

	void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
	}

	void CMFCApplication3Dlg::OnBnClickedButton6()
	{
		// TODO: 在此加入控制項告知處理常式程式碼
		/*
		 Mat image1, image2;
    vector<Point2f> point1, point2, pointCopy;
    vector<uchar> status;
    vector<float> err;


    VideoCapture cap;
	cap.open("featureTracking.mp4");
    cap >> image1;
    Mat image1Gray, image2Gray;
    cvtColor(image1, image1Gray, CV_RGB2GRAY);
    goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
    pointCopy = point1;
    for (int i = 0; i < point1.size(); i++)    //绘制特征点位  
    {
        circle(image1, point1[i], 1, Scalar(0, 0, 255), 2);
    }
    namedWindow("光流特征图");
    while (true)
    {
        cap >> image2;
        if (waitKey(33) == ' ')  //按下空格选择当前画面作为标定图像  
        {
            cvtColor(image2, image1Gray, CV_RGB2GRAY);
            goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
            pointCopy = point1;
        }
        cvtColor(image2, image2Gray, CV_RGB2GRAY);
        calcOpticalFlowPyrLK(image1Gray, image2Gray, point1, point2, status, err, Size(21, 21), 3); //LK金字塔       
        int tr_num = 0;
        vector<unsigned char>::iterator status_itr = status.begin();
        while (status_itr != status.end()) {
            if (*status_itr > 0)
                tr_num++;
            status_itr++;
        }
        if (tr_num < 6) {
            cout << "you need to change the feat-img because the background-img was all changed" << endl;
            if (waitKey(0) == ' ') {
                cvtColor(image2, image1Gray, CV_RGB2GRAY);
                goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
                pointCopy = point1;
            }
        }
        for (int i = 0; i < point2.size(); i++)
        {
            circle(image2, point2[i], 1, Scalar(0, 0, 255), 2);
            line(image2, pointCopy[i], point2[i], Scalar(255, 0, 0), 1, CV_AA);
			
        }

        imshow("光流特征图", image2);
        swap(point1, point2);
        image1Gray = image2Gray.clone();
    }
	*/
		VideoCapture video("featureTracking.mp4");
    //if (!video.isOpened()){
    //    return -1;
    //}
		vector<KeyPoint>keypoints;
    Size videoSize = Size((int)video.get(CV_CAP_PROP_FRAME_WIDTH),(int)video.get(CV_CAP_PROP_FRAME_HEIGHT));
    namedWindow("video demo", CV_WINDOW_AUTOSIZE);
    Mat videoFrame;
	Mat firstFrame;
	int i = 1;
    

	
	
	//play every frame about video
	while(true){
        video >> videoFrame;
		
        if(videoFrame.empty()){
            break;
        }


		

		


	    if(i==1){

			firstFrame = videoFrame;

			imshow ("first image", firstFrame);
			//SimpleBlobDetector
			Mat im = firstFrame;
 
			cv::SimpleBlobDetector::Params params;
			params.minDistBetweenBlobs = 1.0f;
			params.filterByInertia = true;
			params.minInertiaRatio = 0.4;
			params.filterByConvexity = true;
			params.minConvexity = 0.87;
			params.filterByColor = true;
			params.blobColor = 0;
			params.filterByCircularity = 0.83;
			params.filterByArea = true;
			params.minArea = 30.0f;
			params.maxArea = 500.0f;

			// Set up the detector with default parameters.
			SimpleBlobDetector detector(params);
 


			// Detect blobs.
			//std::vector<KeyPoint> keypoints;
			detector.detect( im, keypoints);
			cout<<keypoints.size()<<endl;

			//keypoints.push_back(keypoints[1]);
		    //keypoints[5].pt.x = 0;
			//keypoints[5].pt.y = 0;
			Mat im_with_keypoints;
			for(int i=0; i<keypoints.size(); i++){
				/*if(i!=5){*/
				 rectangle(im, Point(keypoints[i].pt.x,keypoints[i].pt.y), Point(keypoints[i].pt.x-5,keypoints[i].pt.y-5), Scalar(0,0,255), 1);
				rectangle(im, Point(keypoints[i].pt.x,keypoints[i].pt.y), Point(keypoints[i].pt.x+5,keypoints[i].pt.y+5), Scalar(0,0,255), 1);
				rectangle(im, Point(keypoints[i].pt.x,keypoints[i].pt.y), Point(keypoints[i].pt.x-5,keypoints[i].pt.y+5), Scalar(0,0,255), 1);
				rectangle(im, Point(keypoints[i].pt.x,keypoints[i].pt.y), Point(keypoints[i].pt.x+5,keypoints[i].pt.y-5), Scalar(0,0,255), 1);
				/*}*/

				cout<<keypoints[i].pt<<endl;

			}
			

			
			//keypoints.erase(keypoints[5]);
			//cout<<keypoints.back().pt<<endl;
			     

            

			// Draw detected blobs as red circles.
			// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
			
			drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
 myglobal.keypoints=keypoints;
			// Show blobs
			imshow("keypoints", im_with_keypoints );
			//waitKey(0);


		}


        imshow("video demo", videoFrame);

		


		//cout<<i<<endl;
        waitKey(33);
		i++;
    }
	}


	void CMFCApplication3Dlg::OnBnClickedButton4()
	{
		// TODO: 在此加入控制項告知處理常式程式碼
		 cv::VideoCapture capture("bgSub.mp4");
	// check if video successfully opened
	/*if (!capture.isOpened())

		return 0;*/
 
	// current video frame
	cv::Mat frame, frameGray; 
	// foreground binary image
	cv::Mat foreground;
 
	cv::namedWindow("Extracted Foreground");
 
	// The Mixture of Gaussian object
	// used with all default parameters
	Ptr<BackgroundSubtractor> pBackSub;
     pBackSub = new BackgroundSubtractorMOG2(50, 100); //MOG2 approach
	bool stop(false);
	// for all frames in video
	while (!stop) {
 
		// read next frame if any
		if (!capture.read(frame))
			break;
 
		// update the background
		// and return the foreground
		cvtColor(frame, frameGray, CV_BGR2GRAY);
		pBackSub ->operator()(frame,foreground);
 
		// show foreground
		cv::imshow("Extracted Foreground",foreground);
		cv::imshow("image",frame);
 
		// introduce a delay
		// or press key to stop
		if (cv::waitKey(10)>=0)
				stop= true;
	}

	}

	/*
	void CMFCApplication3Dlg::OnBnClickedButton7()
	{
		// TODO: 在此加入控制項告知處理常式程式碼
		vector<KeyPoint> keypoints,keypoints2;

//The SIFT feature extractor and descriptor
//SiftDescriptorExtractor detector;   
SiftDescriptorExtractor detector(8);
SiftDescriptorExtractor detector2(5);
Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );
Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );
Mat input,input2;    
Mat feature,feature2;

//open the file
input = imread("Bird1.jpg", 0); 
input2 = imread("Bird2.jpg", 0);
//detect feature points
detector.detect(input, keypoints);
detector2.detect(input2, keypoints2);

cout <<"img1:"<< keypoints.size() << " points  img2:" <<keypoints2.size() 
		<< " points" << endl << ">" << endl;
///Draw Keypoints
Mat keypointImage,keypointImage2;
keypointImage.create( input.rows, input.cols, CV_8UC3 );
keypointImage2.create( input2.rows, input2.cols, CV_8UC3 );
drawKeypoints(input, keypoints, keypointImage, Scalar::all(-1),0);
drawKeypoints(input2, keypoints2, keypointImage2, Scalar::all(-1),0);
imshow("FeatureBird1.jpg", keypointImage);
imshow("FeatureBird2.jpg", keypointImage2);

//Matched feature points

Mat descriptors1,descriptors2;
	descriptor_extractor->compute( keypointImage, keypoints, descriptors1 );
	descriptor_extractor->compute( keypointImage2, keypoints2, descriptors2 );
descriptor_extractor->compute( keypointImage, keypoints, descriptors1 );  
	vector<DMatch> matches;
	descriptor_matcher->match( descriptors1, descriptors2, matches );
 
	Mat img_matches;
	drawMatches(keypointImage,keypoints,keypointImage2,keypoints2,matches,img_matches,Scalar::all(-1),0,Mat());
 
	imshow("Match",img_matches);
	}
	*/