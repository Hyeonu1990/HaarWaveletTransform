#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

class haar_wavelet
{
public:

	/*그레이스케일 이미지 웨이블릿변환 함수
	할당된 haar_wavelet.img 변수를 웨이블릿변환에 맞게 크기를 조절하고 단계에 맞게 haar 함수를 호출
	@param levels : 웨이블릿 단계
	*/
	void compute(int levels);

	/*그레이스케일 이미지 웨이블릿역변환 함수
	haar_wavelet.img 변수를 역변환하여 재할당
	*/
	void Inverse();

	/*RGB 이미지 웨이블릿변환 함수
	할당된 haar_wavelet.img 변수를 웨이블릿변환에 맞게 크기를 조절하고 단계에 맞게 haar_rgb 함수를 호출
	@param levels : 웨이블릿 단계
	@param im : 추후 R, G, B 중 하나에만 워터마크를 심을 때 사용할 변수
	*/
	void compute_rgb(int levels, int im = -1);

	/*RGB 이미지 웨이블릿역변환 함수
	haar_wavelet.img 변수를 역변환하여 재할당
	*/
	void Inverse_rgb();

	/*norm 변수 변경 함수
	@param n : private 변수인 haar_wavelet.norm에 할당할 값
	*/
	void set_norm_factor(float n);

	/*웨이블릿변환할 이미지 할당 함수
	@param img : private 변수인 haar_wavelet.img 변수에 해당 매개변수 할당
	*/
	void set_image(Mat img);

	/* 웨이블릿변환(역변환) 이미지 출력 함수
	private 변수인 haar_wavelet.img를 return
	*/
	Mat get_haar_pyramid();

private:

	/*haar 웨이블릿변환 함수(그레이스케일 이미지용)
	입력된 width와 height 만큼 haar 웨이블릿 변환
	@param w : 웨이블릿 변환을 시도할 width 길이
	@param h : 웨이블릿 변환을 시도할 height 길이
	*/
	void haar(int w, int h);

	/*haar 웨이블릿변환 함수(RGB 이미지용)
	입력된 width와 height 만큼 haar 웨이블릿 변환
	@param img : 웨이블릿 변환할 이미지. haar_wavelet.img에서 분할된 R, G, B 이미지 중 하나를 받게됨
	@param w : 웨이블릿 변환을 시도할 width 길이
	@param h : 웨이블릿 변환을 시도할 height 길이
	*/
	void haar_rgb(Mat* img, int w, int h);

	/*haar 웨이블릿역변환 함수(그레이스케일 이미지용)
	입력된 width와 height 만큼 haar 웨이블릿 역변환
	@param w : 웨이블릿 역변환을 시도할 width 길이
	@param h : 웨이블릿 역변환을 시도할 height 길이
	*/
	void haar_Inverse(int w, int h);

	/*haar 웨이블릿역변환 함수(RGB 이미지용)
	입력된 width와 height 만큼 haar 웨이블릿 역변환
	@param img : 웨이블릿 역변환할 이미지. haar_wavelet.img에서 분할된 R, G, B 이미지 중 하나를 받게됨
	@param w : 웨이블릿 역변환을 시도할 width 길이
	@param h : 웨이블릿 역변환을 시도할 height 길이
	*/
	void haar_Inverse_rgb(Mat* img, int w, int h);

	//웨이블릿 변환(역변환)에 사용될 이미지
	Mat img;

	//웨이블릿 변환 도중 임시 데이터 보관용 temp변수
	Mat tmp;

	//R,G,B 이미지를 출력하기 위한 모든값이 0인 이미지파일
	Mat zero;

	//작업중인 이미지를 출력하기 위한 임시 변수
	Mat output;

	/*웨이블릿 변환에 사용되는 변수
	일부 논문에서는 2가 아닌 sqrt(2)를 사용
	*/
	float norm = 2;

	//웨이블릿 단계
	int levels;

	//compute_rgb함수에서 할당되는 변수
	int insertmode = 0;

	//이미지 출력 정보 기록용 임시 변수
	int haarcount = 0;

	bool debug = false;

};