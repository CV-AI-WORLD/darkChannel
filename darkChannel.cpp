#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
using namespace cv;
int main(int argc,char*argv[])
{
	Mat image=imread(argv[1],1);
	CV_Assert(!image.empty() && image.channels() == 3);
	//图片的归一化
	Mat fImage;
	image.convertTo(fImage,CV_32FC3,1.0/255,0);
	//规定patch的大小,且均为奇数
	int hPatch = 15;
	int vPatch = 15;
	//给归一化的图片添加边界
	Mat fImageBorder;
	copyMakeBorder(fImage,fImageBorder,vPatch/2,vPatch/2,hPatch/2,hPatch/2,BORDER_REPLICATE);
	//分离通道
	vector<Mat> fImageBorderVector(3);
	split(fImageBorder,fImageBorderVector);
	//创建darkChannel
	Mat darkChannel(image.rows,image.cols,CV_32FC1);
	double minTemp ,minPixel;
	//根据darkChannel的定义
	for(unsigned int r = 0;r < darkChannel.rows;r++)
	{
		for(unsigned int c = 0;c < darkChannel.cols;c++)
		{
			minPixel = 1.0;
			for(vector<Mat>::iterator it = fImageBorderVector.begin() ;it != fImageBorderVector.end();it++)
			{
				Mat roi(*it,Rect(c,r,hPatch,vPatch));
				minMaxLoc(roi,&minTemp);
				minPixel = min(minPixel,minTemp);
			}
			darkChannel.at<float>(r,c) = float(minPixel);
		}
	}
	namedWindow("darkChannel",1);
	imshow("darkChannel",darkChannel);
	Mat darkChannel8U;
	darkChannel.convertTo(darkChannel8U,CV_8UC1,255,0);
	imwrite("darkChannel.jpg",darkChannel8U);
	/*第2步：求出 A(global atmospheric light)*/
	//2.1 计算出darkChannel中,前top个亮的值,论文中取值为0.1%
	float top = 0.001;
	float numberTop = top*darkChannel.rows*darkChannel.cols;
	Mat darkChannelVector;
	darkChannelVector = darkChannel.reshape(1,1);
	Mat_<int> darkChannelVectorIndex;
	sortIdx(darkChannelVector,darkChannelVectorIndex,CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	//制作掩码
	Mat mask(darkChannelVectorIndex.rows,darkChannelVectorIndex.cols,CV_8UC1);//注意mask的类型必须是CV_8UC1
	for(unsigned int r = 0;r < darkChannelVectorIndex.rows;r++)
	{
		for(unsigned int c = 0;c < darkChannelVectorIndex.cols;c++)
		{
			if(darkChannelVectorIndex.at<int>(r,c) <= numberTop)
				mask.at<uchar>(r,c) = 1;
			else 
				mask.at<uchar>(r,c) = 0;
		}
	}
	Mat darkChannelIndex = mask.reshape(1,darkChannel.rows);
	vector<double> A(3);//分别存取A_b,A_g,A_r
	vector<double>::iterator itA = A.begin();
	vector<Mat>::iterator it = fImageBorderVector.begin();
	//2.2在求第三步的t(x)时，会用到以下的矩阵，这里可以提前求出
	vector<Mat> fImageBorderVectorA(3);
	vector<Mat>::iterator itAA = fImageBorderVectorA.begin();
	for( ;it != fImageBorderVector.end() && itA != A.end() && itAA != fImageBorderVectorA.end();it++,itA++,itAA++)
	{
		Mat roi(*it,Rect(hPatch/2,vPatch/2,darkChannel.cols,darkChannel.rows));
		minMaxLoc(roi,0,&(*itA),0,0,darkChannelIndex);//
		(*itAA) = (*it)/(*itA); //[注意：这个地方有除号，但是没有判断是否等于0]
	}
	/*第三步：求t(x)*/
	Mat darkChannelA(darkChannel.rows,darkChannel.cols,CV_32FC1);
	float omega = 0.95;//0<w<=1,论文中取值为0.95
	//代码和求darkChannel的时候,代码差不多
	for(unsigned int r = 0;r < darkChannel.rows;r++)
	{
		for(unsigned int c = 0;c < darkChannel.cols;c++)
		{
			minPixel = 1.0;
			for(itAA = fImageBorderVectorA.begin() ;itAA != fImageBorderVectorA.end();itAA++)
			{
				Mat roi(*itAA,Rect(c,r,hPatch,vPatch));
				minMaxLoc(roi,&minTemp);
				minPixel = min(minPixel,minTemp);
			}
			darkChannelA.at<float>(r,c) = float(minPixel);
		}
	}
	Mat tx = 1.0 - omega*darkChannelA;
	/*第四步：我们可以求J(x)*/
	float t0  = 0.1;//论文中取t0 = 0.1
	Mat jx(image.rows,image.cols,CV_32FC3);
	for(size_t r = 0;r < jx.rows;r++)
	{
		for(size_t c =0;c<jx.cols;c++)
		{
			jx.at<Vec3f>(r,c) = Vec3f((fImage.at<Vec3f>(r,c)[0] - A[0])/max(tx.at<float>(r,c),t0)+A[0],(fImage.at<Vec3f>(r,c)[1] - A[1])/max(tx.at<float>(r,c),t0)+A[1],(fImage.at<Vec3f>(r,c)[2] - A[2])/max(tx.at<float>(r,c),t0)+A[2]);
		}
	}
	namedWindow("jx",1);
	imshow("jx",jx);
	Mat jx8U;
	jx.convertTo(jx8U,CV_8UC3,255,0);
	imwrite("jx.jpg",jx8U);
	waitKey(0);
	system("pause");
	return 0;
}
