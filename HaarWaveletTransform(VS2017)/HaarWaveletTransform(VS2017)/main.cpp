#include "haar_wavelet.hh"

void main()
{
	haar_wavelet haar;
	Mat input,output;
	input = imread("./image/dodo(512x512).png");	

	//gray
	haar.set_image(input.clone());
	cvtColor(input, output, CV_BGR2GRAY);
	imshow("gray_input", output);
	haar.compute(2);
	haar.get_haar_pyramid().convertTo(output, CV_8UC1);
	imshow("Haar_gray", output);
	imwrite("./image/wavelet_gray.png", output);
	haar.Inverse();	
	haar.get_haar_pyramid().convertTo(output, CV_8UC1);
	imshow("Inverse_gray", output);
	imwrite("./image/inversed_gray.png", output);

	//color
	haar.set_image(input.clone());
	imshow("color_input", input);
	haar.compute_rgb(2);
	haar.get_haar_pyramid().convertTo(output, CV_8UC1);
	imshow("Haar_rgb", output);
	imwrite("./image/wavelet_rgb.png", output);
	haar.Inverse_rgb();
	haar.get_haar_pyramid().convertTo(output, CV_8UC1);
	imshow("Inverse_rgb", output);
	imwrite("./image/inversed_rgb.png", output);

	waitKey(0);
}