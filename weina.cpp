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

void fft(Mat planes[], int m, int n, bool rev = false) {


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
}


void weina(Mat& src, Mat& dst) {
	double k = 0.01;
	double R = 5;
	int m = src.rows, n = src.cols;
	int p = log2(m);
	if ((1 << p) != m)
		m = 1 << (p + 1);
	p = log2(n);
	if ((1 << p) != n)
		n = 1 << (p + 1);

	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat::zeros(padded.size(), CV_32F), Mat::zeros(padded.size(), CV_32F) };

	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			double tmp;
			double t = pow(1.0 *i, 2) + pow(1.0*j, 2);
			if (t < R*R || t == R*R)
				tmp = 2.0 / (PI2*R*R);
			else
				tmp = 0;
			planes[0].at<float>(i, j) = tmp;
		}
	fft(planes, m, n);

	Mat plane[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };

	fft(plane, m, n);

	Mat dstt[] = { Mat::zeros(padded.size(), CV_32F), Mat::zeros(padded.size(), CV_32F) };

	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			double a = planes[0].at<float>(i, j);
			double b = planes[1].at<float>(i, j);
			double c = plane[0].at<float>(i, j);
			double d = plane[1].at<float>(i, j);

			double t = a*a + b*b + k;
			dstt[0].at<float>(i, j) = (a*c + b*d) / t;
			dstt[1].at<float>(i, j) = (a*d - b*c) / t;

		}

	fft(dstt, m, n, true);

	dst = Mat_<uchar>(dstt[0]);
}

int main() {
	Mat G = imread("0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat W = Mat::zeros(G.size(), G.type());
	weina(G, W);
	return 0;
}
