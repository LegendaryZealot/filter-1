#undef DBG_NEW
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/imgcodecs.hpp"
#include <Windows.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <algorithm>


using namespace std;
using namespace cv;


int cho = 1;
int gama = 0;
int alpha = 50;
int gama2 = 0;
Mat tmp2;
Mat solv;
Mat tmp;
Mat src;
int beta;
Mat dst;
Mat record;
int rec = 1;
int sigma = 5;
int up, dw, ru, lu;

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
CascadeClassifier& nestedCascade,
double scale, bool tryflip);


void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip){
	int i = 0;
	double t = 0;

	vector<Rect> faces, faces2;
	//定义一些颜色，用来标示不同的人脸

	Mat gray;
	//转成灰度图像，Harr特征基于灰度图
	cvtColor(img, gray, CV_BGR2GRAY);
	equalizeHist(gray,gray);

	cascade.detectMultiScale(gray, faces,
		1.1, 2, 0
		|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		| CV_HAAR_SCALE_IMAGE
		,
		Size(30, 30));
	//如果使能，翻转图像继续检测
	if (tryflip)
	{
		flip(gray, gray, 1);
		cascade.detectMultiScale(gray, faces2,
			1.1, 2, 0
			|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			| CV_HAAR_SCALE_IMAGE
			,
			Size(30, 30));
		for (int r = 0; r < faces2.size(); r++)
		{
			faces.push_back(Rect(gray.cols - faces2[r].x - faces2[r].width, faces2[r].y, faces2[r].width, faces2[r].height));
		}
	}
	for (i = 0; i < faces.size(); i++)
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = Scalar(0, 255, 0);
		Point pt1(faces[i].x + faces[i].width * 1.02, faces[i].y + faces[i].height * 1.1);
		Point pt2(faces[i].x - faces[i].width * 0.02, faces[i].y - faces[i].height * 0.1);
		//		ellipse(img, Point(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5), Size(faces[i].width * 0.5 + img.cols * 0.03, faces[i].height * 0.5 + img.rows * 0.1), 0, 0, 360, Scalar(0, 255, 0, 0), 3, 8, 0);
//		rectangle(img, pt1, pt2, cvScalar(0, 255, 0, 0), 3, 8, 0);
		int ch = img.channels();
		lu = max(0, faces[i].x - faces[i].width * 0.05);
		ru = min(img.cols, faces[i].x + faces[i].width * 1.05);
		up = max(0, faces[i].y - faces[i].height * 0.1);
		dw = min(img.rows, faces[i].y + faces[i].height * 1.1);
		for (int ii = up ; ii < dw; ii++) {
			uchar* ds = solv.ptr<uchar>(ii);
			const uchar* sr = img.ptr<uchar>(ii);
			for (int ji = lu; ji < ru; ji++)
				for (int c = 0; c < ch; c++)
				ds[ji * ch + c] = sr[ji * ch + c];
		}
				
		if (nestedCascade.empty())
			continue;
	}
	if (faces.size() <= 0)record = img.clone();
}


void mopiop(int, void*);
void sharpop(int, void*);
void recov(int, void*);
void lvse(int, void*);
void softlight(int, void*);
void stronglight(int, void*);
void recove();
//void lvjing();
void mopi();
void ImageSharp(Mat &src, Mat &dst, int nAmount = 200);


void local(Mat & src, Mat & dst, int sigma, int n, int m) {
	Mat tmpp = Mat::zeros(src.size(), src.type());
	Mat tmpp2 = Mat::zeros(src.size(), src.type());
	const int ro = src.rows;
	blur(src, tmpp, Size(2 * m + 1, 2 * n + 1), Point(-1, -1));
	//将src中值滤波
	long long int ** re = new long long int*[ro];
	int nc = src.channels();
	for (int i = 0; i < ro; i++)
		re[i] = new long long int[src.cols * nc];
//	Mat del = tmpp(Range(up, dw), Range(lu, ru));
	for (int i = up; i < dw; i++) {
		uchar* ds = dst.ptr<uchar>(i);
		const uchar* I1 = src.ptr<uchar>(i);
		const uchar* I2 = tmpp.ptr<uchar>(i);

		for (int j = lu; j < ru; j++) {
//			if (I1[j * 3] == 128 && I1[j * 3 + 1] == 128 && I1[j * 3 + 2] == 128)continue;
 			if (i == up) {
				if (j == lu) {
					for (int c = 0; c < nc; c++)
						re[i][j * nc + c] = (I1[j * nc + c] - I2[j * nc + c]) * (I1[j * nc + c] - I2[j * nc + c]);
				}

				else {
					for (int c = 0; c < nc; c++)
						re[i][j * nc + c] = re[i][(j - 1) * nc + c] + (I1[j * nc + c] - I2[j * nc + c]) * (I1[j * nc + c] - I2[j * nc + c]);
				}
			}
			else {
				if (j == lu)
					for (int c = 0; c < nc; c++)
						re[i][j * nc + c] = re[i - 1][j * nc + c] + (I1[j * nc + c] - I2[j * nc + c]) * (I1[j * nc + c] - I2[j * nc + c]);
				else
					for (int c = 0; c < nc; c++)
						re[i][j * nc + c] = re[i - 1][j * nc + c] + re[i][(j - 1) * nc + c] - re[i - 1][(j - 1) * nc + c] + (I1[j * nc + c] - I2[j * nc + c]) * (I1[j * nc + c] - I2[j * nc + c]);

			}
		}
	}
//	del.copyTo(tmpp(Range(lu, ru), Range(up, dw)));
	for (int i = up; i < dw; i++) {
		uchar * ds = dst.ptr<uchar>(i);
		const uchar* I1 = src.ptr<uchar>(i);
		//		uchar* I2 = tmpp2.ptr<uchar>(i);
		const uchar* I3 = tmpp.ptr<uchar>(i);
		int nc = src.channels();
		for (int j = lu; j < ru; j++)
			for (int c = 0; c < nc; c++) {
				double sig;
				int rx = j + n, ry = i + m, lx = j - n - 1, ly = i - m - 1;
				if (rx >= ru)rx = ru - 1;
				if (ly < up)ly = up;
				if (lx < lu)lx = lu;
				if (ry >= dw)ry = dw - 1;
				sig = (re[ry][rx * nc + c] - re[ly][rx * nc + c] - re[ry][lx * nc + c] + re[ly][lx * nc + c]) / ((2 * m + 1) * (2 * n + 1));
				double k = (sig * 1.0) / (sig + sigma);
				ds[j * nc + c] = saturate_cast<uchar>((1 - k) * I3[j * nc + c] + k * I1[j * nc + c]);
			}
	}
}


void loc(int, void*) {
	int rad = max((ru - lu), (dw - up)) * 0.02;
	local(solv, tmp, 10 + sigma * sigma * 5, rad, rad);
}
void scan(Mat & src, Mat & dst, int flag) {
	double r, b, g, lu, ll, w;//openCV存储顺序是bgr

	int R, G, B;
	int nc = src.channels();
	for (int i = 0; i < src.rows; i++) {
		uchar * ds = dst.ptr<uchar>(i);
		const uchar* I1 = src.ptr<uchar>(i);
		const uchar* I2 = record.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			int flag1 = 0, flag2 = 0;
				int B = I1[j * nc];
				int R = I1[j * nc + 2];
				int G = I1[j * nc + 1];
				/*				if ((R - G) >= 45) {//requirement 1: R > G > B
									if ((B < G)) {//2 R-G>=45
										int sum = R + G + B;
										r = (R * 1.0) / sum;
										g = (G * 1.0) / sum;
										b = (B * 1.0) / sum;
										w = (r - 1 / 3) * (r - 1 / 3) + (g - 1 / 3)*(g - 1 / 3);

										if (w >= 0.0004) {//req 3
											lu = -1.3767 * r * r + 1.0743 * r + 0.1452;
											if (g < lu) {
												ll = -0.776 * r * r + 0.5601 * r + 0.1766;
												if (g > ll) {
													//判断为肤色区域
													ds[j * nc] = I1[j * nc];
													ds[j * nc + 1] = I1[j * nc + 1];
													ds[j * nc + 2] = I1[j*nc + 2];
													flag1 = 1;
												}
											}
										}
									}
				*/


				if (R > 95) {
					if (G > 40)
						if (B > 20)
							if (R > G)
								if (R > B)
									if (max(R, G, B) - min(R, G, B) > 15)
										if (abs(R - G) > 15) {
											flag1 = 1;
											if (flag == 1) {
												ds[j * nc] = I1[j * nc];
												ds[j * nc + 1] = I1[j * nc + 1];
												ds[j * nc + 2] = I1[j*nc + 2];
											}
										}
				}			
				if (!flag1) {
				if (flag == 1) {
					for (int c = 0; c < 3; c++)
						ds[j * nc + c] = 128;
				}
				else {
					for (int c = 0; c < 3; c++)
						ds[j * nc + c] = I2[j * nc + c];
				}
			}
		}
	}
}


int main(int argc, char** argv) {

	//	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	/// Load the source image
	src = imread("0.jpg", 1);
	int ch = src.channels();
	tmp = Mat::zeros(src.size(), src.type());
	record = src.clone();
	solv = src.clone();

	namedWindow("try", 1);


	dst = record.clone();
//	scan(src, tmp);
		mopi();
	//	lvjing();
	     //	meibai();
	//	recove();
	//	recov(0, 0);
	imshow("try", tmp);
	waitKey(0);



	//	waitKey(0);
	return 0;
}
void recove() {
	createTrackbar("recover", "try",
		&rec, 2,
		recov);
	recov(0, 0);
}
void recov(int, void*) {
	if (rec == 0)
		dst = record.clone();
	if (rec == 2) {
		record = dst.clone();
	}
	imshow("try", dst);
}
void ImageSharp(Mat &src, Mat &dst, int nAmount)
{
	double sigma = 3;
	int threshold = 1;
	float amount = nAmount / 100.0f;

	Mat imgBlurred;
	GaussianBlur(src, imgBlurred, Size(), sigma, sigma);

	Mat lowContrastMask = abs(src - imgBlurred)<threshold;
	dst = src*(1 + amount) + imgBlurred*(-amount);
	src.copyTo(dst, lowContrastMask);
}
void choseMopi(int, void*) {
	
	Mat solvv = solv.clone();
	//这一块将人脸通过肤色识别扫描出来

	if (cho == 1) {
		Mat tmmp = solvv(Range(up, dw), Range(lu, ru));

		tmp = solvv.clone();
		Mat tmpp = Mat::zeros(tmmp.size(), tmmp.type());

		bilateralFilter(tmmp, tmpp, 40, 80, 10);
		tmpp.copyTo(tmp(Range(up, dw), Range(lu, ru)));


	}
	else {
		createTrackbar("local", "try",
			&sigma, 10,
			loc);
		loc(0, 0);
	}
	scan(solvv, solv, 1);
	solvv = tmp.clone();
	scan(solv, tmp, 0);
	for (int i = 0; i < up; i++) {
		uchar* ds = tmp.ptr<uchar>(i);
		const uchar* sc = record.ptr<uchar>(i);
		for (int j = 0; j < src.cols * src.channels(); j++)
			ds[j] = sc[j];

	}
	for (int i = 0; i < src.rows; i++) {
		uchar* ds = tmp.ptr<uchar>(i);
		const uchar* sc = record.ptr<uchar>(i);
		for (int j = 0; j < lu * src.channels(); j++)
			ds[j] = sc[j];

	}
	for (int i = 0; i < src.rows; i++) {
		uchar* ds = tmp.ptr<uchar>(i);
		const uchar* sc = record.ptr<uchar>(i);
		for (int j = ru * src.channels(); j < src.cols * src.channels(); j++)
			ds[j] = sc[j];

	}
	for (int i = dw; i < src.rows; i++) {
		uchar* ds = tmp.ptr<uchar>(i);
		const uchar* sc = record.ptr<uchar>(i);
		for (int j = 0; j < src.cols * src.channels(); j++)
			ds[j] = sc[j];

	}

	tmp2 = tmp - record + 128;

	//	namedWindow("try", 1);
	GaussianBlur(tmp2, tmp, Size(3, 3), 5, 5);
	tmp2 = record + 2 * tmp - 255;
}
void mopi() {
	int c = 0, ch = record.channels();
	for (int i = 0; i < record.rows; i++) {
		uchar * ds = solv.ptr<uchar>(i);
		for (int j = 0; j < record.cols; j++)
			for (c = 0; c < ch; c++)
				ds[j * ch + c] = 128;

	}
	Mat frame = record.clone();

	CascadeClassifier cascade, nestedCascade;
	bool stop = false;
	//训练好的文件名称，放置在可执行文件同目录下
	cascade.load("haarcascade_frontalface_alt.xml");
	//	nestedCascade.load("haarcascade_eye_tree_eyeglasses.xml");
	detectAndDraw(frame, cascade, nestedCascade, 2, 0);
	//	waitKey(0);

	createTrackbar("mopi type", "try",
		&cho, 1,
		choseMopi);

	createTrackbar("mix range", "try",
		&alpha, 100,
		mopiop);

	//	tmp = (src * 50 + tmp2 * 50) / 100;




	createTrackbar("sharp range", "try",
		&beta, 100,
		sharpop);
	choseMopi(0, 0);
	mopiop(0, 0);
	sharpop(0, 0);
}



void mopiop(int, void*) {
	tmp = (tmp2 * alpha + record * (100 - alpha)) / 100;
	dst = tmp.clone();
	imshow("try", dst);
}
void sharpop(int, void*) {
	int i = beta * 3;
	ImageSharp(tmp, dst, i);
	imshow("try", dst);
}

/*
void Light(Mat & src, Mat &src2, Mat & dst, int type) {
	int i, j;
	Size size = src.size();
	int chns = src.channels();
	if (src.isContinuous() && dst.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	for (i = 0; i<size.height; ++i)
	{
		const uchar* srcc = (const uchar*)(src.data + src.step*i);
		const uchar* srcc2 = (const uchar*)(src2.data + src2.step * i);
		uchar* dstt = (uchar*)(dst.data + dst.step*i);
		for (j = 0; j<size.width * chns; ++j)
		{
			if (type == 9)//实色混合
				dstt[j] = saturate_cast<uchar>((srcc2[j] < 128 ? (srcc2[j] == 0 ? 2 * srcc2[j] : max(0, (255 - ((255 - srcc[j]) << 8) / (2 * srcc2[j])))) :
					((2 * (srcc2[j] - 128)) == 255 ? (2 * (srcc2[j] - 128)) : min(255, ((srcc[j] << 8) / (255 - (2 * (srcc2[j] - 128)))))))< 128 ? 0 : 255);
			if (type == 8)//点光
				dstt[j] = saturate_cast<uchar>(max(0, max(2 * srcc[j] - 255, min(srcc2[j], 2 * srcc[j]))));
			if (type == 7)//亮光
				dstt[j] = saturate_cast<uchar>(srcc2[j] < 128 ? (srcc2[j] == 0 ? 2 * srcc2[j] : max(0, (255 - ((255 - srcc[j]) << 8) / (2 * srcc2[j])))) :
					((2 * (srcc2[j] - 128)) == 255 ? (2 * (srcc2[j] - 128)) : min(255, ((srcc[j] << 8) / (255 - (2 * (srcc2[j] - 128)))))));
			if (type == 6)//叠加
				dstt[j] = saturate_cast<uchar>((srcc2[j] < 128) ? (2 * srcc[j] * srcc2[j] / 255) : (255 - 2 * (255 - srcc[j]) * (255 - srcc2[j]) / 255));
			if (type == 5)//线性叠加减淡
				dstt[j] = saturate_cast<uchar>(min(255, (srcc[j] + srcc2[j])));
			if (type == 4)//减弱
				dstt[j] = saturate_cast<uchar>(srcc2[j] == 255 ? srcc2[j] : min(255, ((srcc[j] << 8) / (255 - srcc2[j]))));
			if (type == 3)//强光
				dstt[j] = saturate_cast<uchar>((srcc[j] < 128) ? (2 * srcc[j] * srcc2[j] / 255) : (255 - 2 * (255 - srcc[j]) * (255 - srcc2[j]) / 255));
			if (type == 2)//弱光
				dstt[j] = saturate_cast<uchar>(srcc2[j] < 128 ? (2 * ((srcc[j] >> 1) + 64)) * (srcc2[j] / 255) : (255 - (2 * (255 - ((srcc[j] >> 1) + 64)) * (255 - srcc2[j]) / 255)));
			if (type == 1)//滤色
				dstt[j] = saturate_cast<uchar>(255 - ((255 - srcc[j]) * (255 - srcc2[j])) / 255);
		}
	}
}

void typ(int, void*) {
	Mat pic2 = imread("2.jpg", 1);
	if (!pic2.empty())
		tmp = pic2.clone();
	else
		tmp = record.clone();
	Light(record, tmp, tmp2, gama + 1);
	dst = (record * (100 - gama2) + tmp2 * gama2) / 100;
	imshow("try", dst);
}
void lvjing() {
	tmp2 = Mat::zeros(src.size(), src.type());
	createTrackbar("type: 0 lvse; 1 softli; 2 strli; 3 ColorDodge; 4  LinearjDodge|Add; 5 overlay; 6 vividlight; 7 pinlight; 8 HardMix", "try",
		&gama, 10,
		typ);
	createTrackbar("depth", "try",
		&gama2, 100,
		typ);
	typ(0, 0);
}
*/