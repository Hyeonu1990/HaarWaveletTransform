#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

class haar_wavelet
{
public:

	/*�׷��̽����� �̹��� ���̺���ȯ �Լ�
	�Ҵ�� haar_wavelet.img ������ ���̺���ȯ�� �°� ũ�⸦ �����ϰ� �ܰ迡 �°� haar �Լ��� ȣ��
	@param levels : ���̺� �ܰ�
	*/
	void compute(int levels);

	/*�׷��̽����� �̹��� ���̺�����ȯ �Լ�
	haar_wavelet.img ������ ����ȯ�Ͽ� ���Ҵ�
	*/
	void Inverse();

	/*RGB �̹��� ���̺���ȯ �Լ�
	�Ҵ�� haar_wavelet.img ������ ���̺���ȯ�� �°� ũ�⸦ �����ϰ� �ܰ迡 �°� haar_rgb �Լ��� ȣ��
	@param levels : ���̺� �ܰ�
	@param im : ���� R, G, B �� �ϳ����� ���͸�ũ�� ���� �� ����� ����
	*/
	void compute_rgb(int levels, int im = -1);

	/*RGB �̹��� ���̺�����ȯ �Լ�
	haar_wavelet.img ������ ����ȯ�Ͽ� ���Ҵ�
	*/
	void Inverse_rgb();

	/*norm ���� ���� �Լ�
	@param n : private ������ haar_wavelet.norm�� �Ҵ��� ��
	*/
	void set_norm_factor(float n);

	/*���̺���ȯ�� �̹��� �Ҵ� �Լ�
	@param img : private ������ haar_wavelet.img ������ �ش� �Ű����� �Ҵ�
	*/
	void set_image(Mat img);

	/* ���̺���ȯ(����ȯ) �̹��� ��� �Լ�
	private ������ haar_wavelet.img�� return
	*/
	Mat get_haar_pyramid();

private:

	/*haar ���̺���ȯ �Լ�(�׷��̽����� �̹�����)
	�Էµ� width�� height ��ŭ haar ���̺� ��ȯ
	@param w : ���̺� ��ȯ�� �õ��� width ����
	@param h : ���̺� ��ȯ�� �õ��� height ����
	*/
	void haar(int w, int h);

	/*haar ���̺���ȯ �Լ�(RGB �̹�����)
	�Էµ� width�� height ��ŭ haar ���̺� ��ȯ
	@param img : ���̺� ��ȯ�� �̹���. haar_wavelet.img���� ���ҵ� R, G, B �̹��� �� �ϳ��� �ްԵ�
	@param w : ���̺� ��ȯ�� �õ��� width ����
	@param h : ���̺� ��ȯ�� �õ��� height ����
	*/
	void haar_rgb(Mat* img, int w, int h);

	/*haar ���̺�����ȯ �Լ�(�׷��̽����� �̹�����)
	�Էµ� width�� height ��ŭ haar ���̺� ����ȯ
	@param w : ���̺� ����ȯ�� �õ��� width ����
	@param h : ���̺� ����ȯ�� �õ��� height ����
	*/
	void haar_Inverse(int w, int h);

	/*haar ���̺�����ȯ �Լ�(RGB �̹�����)
	�Էµ� width�� height ��ŭ haar ���̺� ����ȯ
	@param img : ���̺� ����ȯ�� �̹���. haar_wavelet.img���� ���ҵ� R, G, B �̹��� �� �ϳ��� �ްԵ�
	@param w : ���̺� ����ȯ�� �õ��� width ����
	@param h : ���̺� ����ȯ�� �õ��� height ����
	*/
	void haar_Inverse_rgb(Mat* img, int w, int h);

	//���̺� ��ȯ(����ȯ)�� ���� �̹���
	Mat img;

	//���̺� ��ȯ ���� �ӽ� ������ ������ temp����
	Mat tmp;

	//R,G,B �̹����� ����ϱ� ���� ��簪�� 0�� �̹�������
	Mat zero;

	//�۾����� �̹����� ����ϱ� ���� �ӽ� ����
	Mat output;

	/*���̺� ��ȯ�� ���Ǵ� ����
	�Ϻ� �������� 2�� �ƴ� sqrt(2)�� ���
	*/
	float norm = 2;

	//���̺� �ܰ�
	int levels;

	//compute_rgb�Լ����� �Ҵ�Ǵ� ����
	int insertmode = 0;

	//�̹��� ��� ���� ��Ͽ� �ӽ� ����
	int haarcount = 0;

	bool debug = false;

};