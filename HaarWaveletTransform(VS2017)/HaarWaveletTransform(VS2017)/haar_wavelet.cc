#include "haar_wavelet.hh"

#include <cmath>
#include <iostream>


void haar_wavelet::compute(int levels)
{
	this->levels = levels;
	tmp.release();
	if (img.channels() > 1)	cvtColor(img, img, CV_BGR2GRAY);

	int w = 1;
	int h = 1;

	for (; w < img.cols; w *= 2);
	for (; h < img.rows; h *= 2);

	int max = w > h ? w : h;
	w = h = max;

	Mat img_pad = Mat(h, w, CV_32FC1);

	Range colRange = Range(0, min(img.cols, img_pad.cols));
	Range rowRange = Range(0, min(img.rows, img_pad.rows));
	img(rowRange, colRange).copyTo(img_pad(rowRange, colRange));

	img = img_pad.clone();

	haarcount = 0;
	for (int i = 0; i < levels; i++)
		haar(w / pow(2, i), h / pow(2, i)); // pwo(2, i) = 2^i
}

//haar 웨이블릿 변환 식(그레이스케일용)
void haar_wavelet::haar(int w, int h)
{

	int w_2 = w / 2;
	int h_2 = h / 2;

	tmp = img.clone();

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w_2; j++)
		{
			auto a = tmp.at<float>(j * 2 + 1, i);
			auto b = tmp.at<float>(j * 2, i);

			img.at<float>(j + w_2, i) = (a - b) / norm; // norm = 2
			img.at<float>(j, i) = (a + b) / norm;
		}

	tmp = img.clone();
	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		imshow("tmp" + std::to_string(haarcount), output);
		cv::imwrite("./image/tmp" + std::to_string(haarcount) + ".png", output);
		haarcount++;
	}

	for (int i = 0; i < h_2; i++)
		for (int j = 0; j < w; j++)
		{
			auto a = tmp.at<float>(j, i * 2 + 1);
			auto b = tmp.at<float>(j, i * 2);

			img.at<float>(j, h_2 + i) = (a - b) / norm;
			img.at<float>(j, i) = (a + b) / norm;
		}

	tmp = img.clone();
	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		imshow("tmp" + std::to_string(haarcount), output);
		cv::imwrite("./image/tmp" + std::to_string(haarcount) + ".png", output);
		haarcount++;
	}
}

//그레이스케일 이미지 웨이블릿 역변환
void haar_wavelet::Inverse()
{

	tmp.release();

	int w = 1;
	int h = 1;

	for (; w < img.cols; w *= 2);
	for (; h < img.rows; h *= 2);

	int max = w > h ? w : h;
	w = h = max;

	haarcount = 0;
	for (int i = levels - 1; i >= 0; i--)
		haar_Inverse(w / pow(2, i), h / pow(2, i)); // pwo(2, i) = 2^i
}

//haar 웨이블릿 역변환 식(그레이스케일용)
void haar_wavelet::haar_Inverse(int w, int h)
{

	int w_2 = w / 2;
	int h_2 = h / 2;

	tmp = img.clone();

	for (int i = h_2 - 1; i >= 0; i--)
		for (int j = w - 1; j >= 0; j--)
		{
			/*auto a = tmp.at<char>(j, i * 2 + 1);
			auto b = tmp.at<char>(j, i * 2);

			img.at<char>(j, h_2 + i) = (a - b) / norm;
			img.at<char>(j, i) = (a + b) / norm;*/
			img.at<float>(j, i * 2 + 1) = tmp.at<float>(j, i) + tmp.at<float>(j, h_2 + i);
			img.at<float>(j, i * 2) = tmp.at<float>(j, i) - tmp.at<float>(j, h_2 + i);
		}

	tmp = img.clone();
	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		imshow("Inverse" + std::to_string(haarcount), output);
		cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", output);
		haarcount++;
	}


	for (int i = h - 1; i >= 0; i--)
		for (int j = w_2 - 1; j >= 0; j--)
		{
			//auto a = tmp.at<char>(j * 2 + 1, i);
			//auto b = tmp.at<char>(j * 2, i);

			//img.at<char>(j + w_2, i) = (a - b) / norm; // norm = 2
			//img.at<char>(j, i) = (a + b) / norm;
			img.at<float>(j * 2 + 1, i) = tmp.at<float>(j, i) + tmp.at<float>(j + w_2, i);
			img.at<float>(j * 2, i) = tmp.at<float>(j, i) - tmp.at<float>(j + w_2, i);
		}

	tmp = img.clone();
	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		imshow("Inverse" + std::to_string(haarcount), output);
		cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", output);
		haarcount++;
	}
}

//컬러이미지 웨이블릿 변환
void haar_wavelet::compute_rgb(int levels, int im)
{
	this->insertmode = im;
	this->levels = levels;
	tmp.release();

	int w = 1;
	int h = 1;

	for (; w < img.cols; w *= 2);
	for (; h < img.rows; h *= 2);

	int max = w > h ? w : h;
	w = h = max;

	Mat img_pad = Mat(h, w, CV_32FC3);

	Range colRange = Range(0, min(img.cols, img_pad.cols));
	Range rowRange = Range(0, min(img.rows, img_pad.rows));
	img(rowRange, colRange).copyTo(img_pad(rowRange, colRange));

	img = img_pad.clone();

	haarcount = 0;

	cv::Mat channelsrc[3];
	split(img.clone(), channelsrc);

	for (int n = 0; n < 3; n++)
		for (int i = 0; i < levels; i++)
		{
			//channelsrc[n].convertTo(channelsrc[n], CV_32FC1);
			haar_rgb(&channelsrc[n], w / pow(2, i), h / pow(2, i)); // pwo(2, i) = 2^i
																	//channelsrc[n].convertTo(channelsrc[n], CV_8UC1);
		}
	merge(channelsrc, 3, img);
}

//haar 웨이블릿 변환 식(컬러이미지용)
void haar_wavelet::haar_rgb(Mat* img, int w, int h)
{

	int w_2 = w / 2;
	int h_2 = h / 2;

	tmp = img->clone();

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w_2; j++)
		{
			auto a = tmp.at<float>(j * 2 + 1, i);
			auto b = tmp.at<float>(j * 2, i);

			img->at<float>(j + w_2, i) = (a - b) / norm; // norm = 2
			img->at<float>(j, i) = (a + b) / norm;
		}

	tmp = img->clone();
	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		imshow("tmp" + std::to_string(haarcount), output);
		cv::imwrite("./image/tmp" + std::to_string(haarcount) + ".png", output);
		haarcount++;
	}

	for (int i = 0; i < h_2; i++)
		for (int j = 0; j < w; j++)
		{
			auto a = tmp.at<float>(j, i * 2 + 1);
			auto b = tmp.at<float>(j, i * 2);

			img->at<float>(j, h_2 + i) = (a - b) / norm;
			img->at<float>(j, i) = (a + b) / norm;
		}

	tmp = img->clone();
	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		imshow("tmp" + std::to_string(haarcount), output);
		cv::imwrite("./image/tmp" + std::to_string(haarcount) + ".png", output);
		haarcount++;
	}
}

//컬러이미지 웨이블릿 역변환
void haar_wavelet::Inverse_rgb()
{

	tmp.release();

	int w = 1;
	int h = 1;

	for (; w < img.cols; w *= 2);
	for (; h < img.rows; h *= 2);

	int max = w > h ? w : h;
	w = h = max;

	haarcount = 0;

	cv::Mat channelsrc[3];
	split(img.clone(), channelsrc);

	zero = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

	for (int n = 0; n < 3; n++)
		for (int i = levels - 1; i >= 0; i--)
		{
			//channelsrc[n].convertTo(channelsrc[n], CV_32FC1);
			haar_Inverse_rgb(&channelsrc[n], w / pow(2, i), h / pow(2, i)); // pwo(2, i) = 2^i
			if (i == 0)channelsrc[n].convertTo(channelsrc[n], CV_8UC1);
		}
	merge(channelsrc, 3, img);
}

//haar 웨이블릿 역변환 식(컬러이미지용)
void haar_wavelet::haar_Inverse_rgb(Mat* img, int w, int h)
{

	int w_2 = w / 2;
	int h_2 = h / 2;

	tmp = img->clone();

	for (int i = 0; i < h_2; i++)
		for (int j = 0; j < w; j++)
		{
			/*auto a = tmp.at<char>(j, i * 2 + 1);
			auto b = tmp.at<char>(j, i * 2);

			img.at<char>(j, h_2 + i) = (a - b) / norm;
			img.at<char>(j, i) = (a + b) / norm;*/
			img->at<float>(j, i * 2 + 1) = tmp.at<float>(j, i) + tmp.at<float>(j, h_2 + i);
			img->at<float>(j, i * 2) = tmp.at<float>(j, i) - tmp.at<float>(j, h_2 + i);
		}

	tmp = img->clone();

	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		if (haarcount >= 0 && haarcount <= 3)
		{
			Mat matArray[] = { output, zero, zero };
			Mat result; merge(matArray, 3, result);
			imshow("Inverse" + std::to_string(haarcount), result);
			cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", result);
		}
		else if (haarcount >= 4 && haarcount <= 7)
		{
			Mat matArray[] = { zero, output, zero };
			Mat result; merge(matArray, 3, result);
			imshow("Inverse" + std::to_string(haarcount), result);
			cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", result);
		}
		else if (haarcount >= 8 && haarcount <= 11)
		{
			Mat matArray[] = { zero, zero, output };
			Mat result; merge(matArray, 3, result);
			imshow("Inverse" + std::to_string(haarcount), result);
			cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", result);
		}
		haarcount++;
	}

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w_2; j++)
		{
			//auto a = tmp.at<char>(j * 2 + 1, i);
			//auto b = tmp.at<char>(j * 2, i);

			//img.at<char>(j + w_2, i) = (a - b) / norm; // norm = 2
			//img.at<char>(j, i) = (a + b) / norm;
			img->at<float>(j * 2 + 1, i) = tmp.at<float>(j, i) + tmp.at<float>(j + w_2, i);
			img->at<float>(j * 2, i) = tmp.at<float>(j, i) - tmp.at<float>(j + w_2, i);
		}

	tmp = img->clone();

	if (debug)
	{
		tmp.convertTo(output, CV_8UC1);
		if (haarcount >= 0 && haarcount <= 3)
		{
			Mat matArray[] = { output, zero, zero };
			Mat result; merge(matArray, 3, result);
			imshow("Inverse" + std::to_string(haarcount), result);
			cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", result);
		}
		else if (haarcount >= 4 && haarcount <= 7)
		{
			Mat matArray[] = { zero, output, zero };
			Mat result; merge(matArray, 3, result);
			imshow("Inverse" + std::to_string(haarcount), result);
			cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", result);
		}
		else if (haarcount >= 8 && haarcount <= 11)
		{
			Mat matArray[] = { zero, zero, output };
			Mat result; merge(matArray, 3, result);
			imshow("Inverse" + std::to_string(haarcount), result);
			cv::imwrite("./image/Inverse" + std::to_string(haarcount) + ".png", result);
		}
		haarcount++;
	}
}

void haar_wavelet::set_image(Mat img)
{
	this->img = img;
}

void haar_wavelet::set_norm_factor(float n)
{
	norm = n;
}

Mat haar_wavelet::get_haar_pyramid()
{
	return img;
}
