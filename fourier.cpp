#undef DBG_NEW
#include <math.h>    

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
#define PI2   6.283185307179586476925286766559   

using namespace std;
using namespace cv;
Mat padded;
void swap(float& a1, float& a2) {
	float tmp = a2;
	a2 = a1;
	a1 = tmp;
}

void bitreverse(float xreal[], float xima[], int n) {	//按位反序，得到的数字为实际的位置坐标
	int i, j, a, b, p;
	int num = log2(n);
	for (i = 0; i < n; i++) {
		a = i;
		b = 0;
		for (j = 0; j < num; j++) {
			b = (b << 1) | (a & 1);
			a >>= 1;
		}
		if (b > i) {
			swap(xreal[i], xreal[b]);
			swap(xima[i], xima[b]);
		}
	}
}
void dft(float xreal[], float xima[], int n, bool rev = false) {
	float tmpr;
	float tmpi;
	float tmp2r, tmp2i;
	int flag = 1;
	if (rev)flag = -1;
	bitreverse(xreal, xima, n);

	for (int i = 2; i <= n; i *= 2) {
		for (int k = 0; k < n; k += i) {//对每个长度有k个串
			for (int j = 0; j < i / 2; j++) {//开始计算第k个串
				int index1 = k + j;
				int index2 = k + j + i / 2;
				float ar = (PI2 * j) / i;
				float w1r = cos(flag * ar), w1i = -sin(flag * ar);

				tmpr = xreal[index1] - xima[index2] * w1i + xreal[index2] * w1r;
				tmpi = xima[index1] + xreal[index2] * w1i + xima[index2] * w1r;
				tmp2r = xreal[index1] - xreal[index2] * w1r + xima[index2] * w1i;
				tmp2i = xima[index1] - xreal[index2] * w1i - xima[index2] * w1r;

				xreal[index1] = tmpr;
				xima[index1] = tmpi;
				xreal[index2] = tmp2r;
				xima[index2] = tmp2i;
			}
		}
	}
	if (rev) {
		for (int i = 0; i < n; i++) {
			xreal[i] /= n;
			xima[i] /= n;
		}
	}
}

void fft(Mat& image, bool rev = false) {
	int m, n;

	int p = log2(image.rows);
	if (1 << p != image.rows)
		m = 1 << (p + 1);
	p = log2(image.cols);
	if (1 << p != image.cols)
		n = 1 << (p + 1);



	//	Mat padded;
	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));


	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };

	for (int ii = 0; ii < m; ii++) {
		float* xreal = planes[0].ptr <float>(ii);
		float* xima = planes[1].ptr<float>(ii);
		dft(xreal, xima, n,rev);
	}

	for (int ii = 0; ii < n; ii++) {
		float* xreal = new float[m];
		float* xima = new float[m];
		for (int j = 0; j < m; j++) {
			xreal[j] = planes[0].at<float>(j, ii);
			xima[j] = planes[1].at<float>(j, ii);
		}
		dft(xreal, xima, m, rev);
		for (int j = 0; j < m; j++) {
			planes[0].at<float>(j, ii) = xreal[j];
			planes[1].at<float>(j, ii) = xima[j];
		}
	}
	for (int ii = 0; ii < m; ii++) {
		float* xreal = planes[0].ptr <float>(ii);
		float* xima = planes[1].ptr<float>(ii);
		for (int j = 0; j < n; j++) {
			if (xreal[j] < 10)xreal[j] = 10;
			if (xima[j] < 10)xima[j] = 10;
		}
	}



	for (int ii = 0; ii < n; ii++) {
		float* xreal = new float[m];
		float* xima = new float[m];
		for (int j = 0; j < m; j++) {
			xreal[j] = planes[0].at<float>(j, ii);
			xima[j] = planes[1].at<float>(j, ii);
		}
		dft(xreal, xima, m, true);
		for (int j = 0; j < m; j++) {
			planes[0].at<float>(j, ii) = xreal[j];
			planes[1].at<float>(j, ii) = xima[j];
		}
	}

	for (int ii = 0; ii < m; ii++) {
		float* xreal = planes[0].ptr <float>(ii);
		float* xima = planes[1].ptr<float>(ii);
		dft(xreal, xima, n, true);
	}

/*
	Mat rrrtmp = Mat_<uchar>(planes[0]);
	Mat magI = planes[0].clone();

	magnitude(planes[1], planes[0], magI);

	magI += Scalar::all(1);
	log(magI, magI);
	magI = magI(Rect(0, 0, image.cols,image.rows));
	Mat _magI = magI.clone();
	normalize(_magI, _magI, 0, 1, CV_MINMAX);

	// rearrange the quadrants of Fourier image so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));    // Top-Left
	Mat q1(magI, Rect(cx, 0, cx, cy));   // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));   // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy));  // Bottom-Right

	// exchange Top-Left and Bottom-Right
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	// exchange Top-Right and Bottom-Left
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX);
*/
}

int main() {

	Mat channels[3];
	Mat dst;
	Mat srcImage = imread("0.jpg");
	split(srcImage, channels);
	fft(channels[0], false);
	merge(channels, 3, dst);
	return 0;
}