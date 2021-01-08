/**************************************************************

MIT License

Copyright (c) 2018 SEU-SuperNova-CVRA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors:	Rick_Hang, <213162574@seu.edu.cn>
		BinYan Hu
**************************************************************/
#include"ArmorDetector.h"
#include<iostream>
#include<vector>
#include<math.h>
#include<string>

#include"../General/opencv_extended.h"
#include"../General/numeric_rm.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace rm
{

enum
{
	WIDTH_GREATER_THAN_HEIGHT,
	ANGLE_TO_UP
};
/*
*	@Brief:		// regulate the rotated rect
*	@Input:		// rotated rec
*				// regulation mode
*	@Return:	// regulated rec
*/
cv::RotatedRect& adjustRec(cv::RotatedRect& rec, const int mode)
{
	using std::swap;

	float& width = rec.size.width;
	float& height = rec.size.height;
	float& angle = rec.angle;

	if(mode == WIDTH_GREATER_THAN_HEIGHT)
	{
		if(width < height)
		{
			swap(width, height);
			angle += 90.0;
		}
	}

	while(angle >= 90.0) angle -= 180.0;
	while(angle < -90.0) angle += 180.0;

	if(mode == ANGLE_TO_UP)
	{
		if(angle >= 45.0)
		{
			swap(width, height);
			angle -= 90.0;
		}
		else if(angle < -45.0)
		{
			swap(width, height);
			angle += 90.0;
		}
	}

	return rec;
}


ArmorDescriptor::ArmorDescriptor()
{
	rotationScore = 0;
	sizeScore = 0;
	vertex.resize(4);
	for(int i = 0; i < 4; i++)
	{
		vertex[i] = cv::Point2f(0, 0);
	}
	type = UNKNOWN_ARMOR;
}

ArmorDescriptor::ArmorDescriptor(const LightDescriptor & lLight, const LightDescriptor & rLight, const int armorType, const cv::Mat & grayImg, float rotaScore, ArmorParam _param)
{
	//handle two lights
	lightPairs[0] = lLight.rec();
	lightPairs[1] = rLight.rec();

	cv::Size exLSize(int(lightPairs[0].size.width), int(lightPairs[0].size.height * 2));
	cv::Size exRSize(int(lightPairs[1].size.width), int(lightPairs[1].size.height * 2));
	cv::RotatedRect exLLight(lightPairs[0].center, exLSize, lightPairs[0].angle);
	cv::RotatedRect exRLight(lightPairs[1].center, exRSize, lightPairs[1].angle);

	cv::Point2f pts_l[4];
	exLLight.points(pts_l);
	cv::Point2f upper_l = pts_l[2];
	cv::Point2f lower_l = pts_l[3];

	cv::Point2f pts_r[4];
	exRLight.points(pts_r);
	cv::Point2f upper_r = pts_r[1];
	cv::Point2f lower_r = pts_r[0];

	vertex.resize(4);
	vertex[0] = upper_l;
	vertex[1] = upper_r;
	vertex[2] = lower_r;
	vertex[3] = lower_l;

	//set armor type
	type = armorType;

	//get front view
	getFrontImg(grayImg);
	rotationScore = rotaScore;

	// calculate the size score
	float normalized_area = contourArea(vertex) / _param.area_normalized_base;
	sizeScore = exp(normalized_area);

	// calculate the distance score
	Point2f srcImgCenter(grayImg.cols / 2, grayImg.rows / 2);
	float sightOffset = cvex::distance(srcImgCenter, cvex::crossPointOf(array<Point2f, 2>{vertex[0],vertex[2]}, array<Point2f, 2>{vertex[1],vertex[3]}));
	distScore = exp(-sightOffset / _param.sight_offset_normalized_base);

}

void ArmorDescriptor::getFrontImg(const Mat& grayImg)
{
	using cvex::distance;
	const Point2f&
		tl = vertex[0],
		tr = vertex[1],
		br = vertex[2],
		bl = vertex[3];

	int width, height;
	if(type == BIG_ARMOR)
	{
		width = 92;
		height = 50;
	}
	else
	{
		width = 50;
		height = 50;
	}

	Point2f src[4]{Vec2f(tl), Vec2f(tr), Vec2f(br), Vec2f(bl)};
	Point2f dst[4]{Point2f(0.0, 0.0), Point2f(width, 0.0), Point2f(width, height), Point2f(0.0, height)};
	const Mat perspMat = getPerspectiveTransform(src, dst);
	cv::warpPerspective(grayImg, frontImg, perspMat, Size(width, height));
}


ArmorDetector::ArmorDetector()
{
    _flag = ARMOR_NO;
    _roi = Rect(cv::Point(0, 0), _srcImg.size());
    _isTracking = false;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
    _debugWindowName = "debug info";
#endif // DEBUG_DETECTION || SHOW_RESULT
}

ArmorDetector::ArmorDetector(const ArmorParam & armorParam)
{
	_param = armorParam;
	_flag = ARMOR_NO;
	_roi = Rect(cv::Point(0, 0), _srcImg.size());
	_isTracking = false;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
	_debugWindowName = "debug info";
#endif // DEBUG_DETECTION || SHOW_RESULT
}

void ArmorDetector::init(const ArmorParam &armorParam)
{
    _param = armorParam;
}

void ArmorDetector::loadImg(const cv::Mat & srcImg)
{
	_srcImg = srcImg;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
	_debugImg = srcImg.clone();
#endif // DEBUG_DETECTION || SHOW_RESULT

	Rect imgBound = Rect(cv::Point(0, 0), _srcImg.size());

    if(_flag == ARMOR_LOCAL && _trackCnt != _param.max_track_num) // 局部搜索
    {
        cv::Rect bRect = boundingRect(_targetArmor.vertex) + _roi.tl();
        bRect = cvex::scaleRect(bRect, Vec2f(3,2));	//以中心为锚点放大2倍
        _roi = bRect & imgBound;
        _roiImg = _srcImg(_roi).clone();
    }
    else // 全局搜索
    {
		_roi = imgBound;
		_roiImg = _srcImg.clone();
		_trackCnt = 0;
    }

#ifdef DEBUG_DETECTION
	rectangle(_debugImg, _roi, cvex::YELLOW);
#endif // DEBUG_DETECTION
}


int ArmorDetector::detect()
{
	/*
	*	Detect lights and build light bars' desciptors
	*/
	_armors.clear();
	std::vector<LightDescriptor> lightInfos;
	{
		/*
		*	pre-treatment
		*   阈值化处理得到二值图，之后可以进行膨胀处理让图像中的轮廓更明显
		*/
		cv::Mat binBrightImg;
		//阈值化
        cvtColor(_roiImg, _grayImg, COLOR_BGR2GRAY, 1);
		cv::threshold(_grayImg, binBrightImg, _param.brightness_threshold, 255, cv::THRESH_BINARY);

        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		//膨胀
        dilate(binBrightImg, binBrightImg, element);	

    #ifdef DEBUG_PRETREATMENT
        imshow("brightness_binary", binBrightImg);
        waitKey();
    #endif // DEBUG_PRETREATMENT

		/*
		*	find and filter light bars
		*   找轮廓，这步是整个算法中最耗时的部分，如果预处理做的好，可以极大地减少找轮廓中花费的时间
		*/
		vector<vector<Point>> lightContours;
		//找轮廓
		cv::findContours(binBrightImg.clone(), lightContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// 找到轮廓后开始遍历轮廓提取灯条
		for(const auto& contour : lightContours)
		{
			//得到面积
			float lightContourArea = contourArea(contour);
			//面积太小的不要
			if(contour.size() <= 5 ||
			   lightContourArea < _param.light_min_area) continue;

			// 将外接矩形拟合成椭圆
			// https://www.cnblogs.com/hsy1941/p/7923323.html
			// angle：在opencv中，坐标的原点在左上角，与x轴平行的方向为角度为0，
			// 逆时针旋转角度为负，顺时针旋转角度为正。角度是水平轴（x轴）顺时针旋转，与碰到的第一个边的夹角度数。而opencv默认把这个边的边长作为height。
			RotatedRect lightRec = fitEllipse(contour);  // angle、width、height、center
			//矫正灯条，调用函数adjustRec
			adjustRec(lightRec, ANGLE_TO_UP);
			
			//宽高比、凸度筛选灯条
			//float solidity = lightContourArea / lightRec.size.area();
            		if(lightRec.size.width / lightRec.size.height > _param.light_max_ratio ||
			   lightContourArea / lightRec.size.area() < _param.light_contour_min_solidity) continue;

			//Mat temp;
			//cvex::showRectangle("light_right_position", _srcImg, temp, lightRec, cvex::GREEN,0, _roi.tl());
			//对灯条范围适当扩大
			lightRec.size.width *= _param.light_color_detect_extend_ratio;
			lightRec.size.height *= _param.light_color_detect_extend_ratio;

			// boundingRect()得到包覆此轮廓的最小正矩形
			// minAreaRect()得到包覆轮廓的最小斜矩形
			Rect lightRect = lightRec.boundingRect();
			const Rect srcBound(Point(0, 0), _roiImg.size());
			// 两个Rect取并集
			lightRect &= srcBound;
			Mat lightImg = _roiImg(lightRect);
			Mat lightMask = Mat::zeros(lightRect.size(), CV_8UC1);  // 灯条的外接矩形掩膜
			Point2f lightVertexArray[4];
			lightRec.points(lightVertexArray);  //Rec的四个顶点
			std::vector<Point> lightVertex;
			for(int i = 0; i < 4; i++)
			{
				lightVertex.emplace_back(Point(lightVertexArray[i].x - lightRect.tl().x,
											   lightVertexArray[i].y - lightRect.tl().y));
			}
			// 填充凸多边形，用lightVertex四个点填充lightMask
			fillConvexPoly(lightMask, lightVertex, 255);

            		if(lightImg.size().area() <= 0 || lightMask.size().area() <= 0) continue;
            		cv::dilate(lightMask, lightMask, element);
			// Scalar设置单一颜色，这里单一颜色提取，获得是掩膜区域颜色的三个通道各自的均值。
			const Scalar meanVal = mean(lightImg, lightMask);

			//Mat debugColorImg = _srcImg.clone();
			//const String BGR = "(" + std::to_string(int(meanVal[0])) + ", " + std::to_string(int(meanVal[1])) + ", " + std::to_string(int(meanVal[2])) + ")";

			if(((_enemy_color == BLUE) && (meanVal[BLUE] - meanVal[RED] > 20.0)) || (_enemy_color == RED && meanVal[RED] - meanVal[BLUE] > 20.0))
			{
				lightInfos.push_back(LightDescriptor(lightRec));

				//putText(debugColorImg, BGR, Point(lightVertexArray[0]) + _roi.tl(), FONT_HERSHEY_SIMPLEX, 0.4, cvex::GREEN, 1); //fontScalar 0.34
			}
			//else
			//{
			//	putText(debugColorImg, BGR, Point(lightVertexArray[0]) + _roi.tl(), FONT_HERSHEY_SIMPLEX, 0.4, cvex::CYAN, 1); //fontScalar 0.34
			//}
			//imshow("BGR", debugColorImg);
			//waitKey(0);

		}

	#ifdef DEBUG_DETECTION
		vector<RotatedRect> lightsRecs;
		for(auto & light : lightInfos)
		{
			lightsRecs.emplace_back(light.rec());
		}
        cvex::showRectangles(_debugWindowName, _debugImg, _debugImg, lightsRecs, cvex::MAGENTA, 1, _roi.tl());
	#endif //DEBUG_DETECTION
        //// 没找到灯条就返回没找到
		if(lightInfos.empty())
		{
			return _flag = ARMOR_NO;
		}
	}


	/*
	*	find and filter light bar pairs
	*   对灯条进行匹配筛选
	*/

	{
		//按灯条中心x从小到大排序
		sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2)
		{
			//Lambda函数,作为sort的cmp函数
			return ld1.center.x < ld2.center.x;
		});
		vector<int> minRightIndices(lightInfos.size(), -1);
		for(size_t i = 0; i < lightInfos.size(); i++)
		{//遍历所有灯条进行匹配
			for(size_t j = i + 1; (j < lightInfos.size()); j++)
			{
				const LightDescriptor& leftLight  = lightInfos[i];
				const LightDescriptor& rightLight = lightInfos[j];

		#ifdef DEBUG_DETECTION
				Mat pairImg = _debugImg.clone();
				vector<RotatedRect> curLightPair{leftLight.rec(), rightLight.rec()};
                		cvex::showRectangles("debug pairing",  pairImg, pairImg,curLightPair, cvex::CYAN, 1, _roi.tl());
		#endif // DEBUG_DETECTION

				/*
				*	Works for 2-3 meters situation
				*	morphologically similar: // parallel 
								 // similar height
				*/
			    //角差
				float angleDiff_ = abs(leftLight.angle - rightLight.angle);
				//长度差比率
				float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
				//筛选
				if(angleDiff_ > _param.light_max_angle_diff_ ||
				   LenDiff_ratio > _param.light_max_height_diff_ratio_)
				{
					continue;
				}

				/*
				*	proper location: // y value of light bar close enough 
				*			 // ratio of length and width is proper
				*/
				//左右灯条相距距离
				float dis = cvex::distance(leftLight.center, rightLight.center);
				//左右灯条长度的平均值
				float meanLen = (leftLight.length + rightLight.length) / 2;
				//左右灯条中心点y的差值
				float yDiff = abs(leftLight.center.y - rightLight.center.y);
				//y差比率
				float yDiff_ratio = yDiff / meanLen;
				//左右灯条中心点x的差值
				float xDiff = abs(leftLight.center.x - rightLight.center.x);
				//x差比率
				float xDiff_ratio = xDiff / meanLen;
				//相距距离与灯条长度比值
				float ratio = dis / meanLen;
				//筛选
				if(yDiff_ratio > _param.light_max_y_diff_ratio_ ||
				   xDiff_ratio < _param.light_min_x_diff_ratio_ ||
				   ratio > _param.armor_max_aspect_ratio_ ||
				   ratio < _param.armor_min_aspect_ratio_)
				{
					continue;
				}

				// calculate pairs' info 
				//按比值来确定大小装甲
				int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR;
				// calculate the rotation score
				float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio - ratio, float(0)) : max(_param.armor_small_armor_ratio - ratio, float(0));
				float yOff = yDiff / meanLen;
				float rotationScore = -(ratiOff * ratiOff + yOff * yOff);

				//得到匹配的装甲板
				ArmorDescriptor armor(leftLight, rightLight, armorType, _grayImg, rotationScore, _param);
				_armors.emplace_back(armor);
				break;
			}
		}

	#ifdef  DEBUG_DETECTION
		vector<vector<Point>> armorVertexs;
		for(const auto & armor : _armors)
		{
			vector<Point> intVertex;
			for(const auto& point : armor.vertex)
			{
				intVertex.emplace_back(Point(point.x,point.y));
			}
			armorVertexs.emplace_back(intVertex);
		}
        cvex::showContours(_debugWindowName, _debugImg, _debugImg,armorVertexs,  cvex::WHITE, 1, _roi.tl());
	#endif //  DEBUG_DETECTION

		//没匹配到装甲板则返回没找到
		if(_armors.empty())
		{
			return _flag = ARMOR_NO;
		}
	}

#ifdef GET_ARMOR_PIC
	_allCnt++;
	int i = 0;
	for(const auto & armor : _armors)
    	{
        	Mat regulatedFrontImg = armor.frontImg;
        	if(armor.type == BIG_ARMOR)
        	{
       		     	regulatedFrontImg = regulatedFrontImg(Rect(21, 0, 50, 50));
        	}
        	imwrite("/home/nvidia/Documents/ArmorTrainingSample/" + to_string(_allCnt) + "_" + to_string(i) + ".bmp", regulatedFrontImg);
		i++;
	}
#endif // GET_ARMOR_PIC

	//对找到的装甲板进行筛选
	//delete the fake armors
	_armors.erase(remove_if(_armors.begin(), _armors.end(), [](ArmorDescriptor& i)
	{//lamdba函数判断是不是装甲板，将装甲板中心的图片提取后让识别函数去识别，识别可以用svm或者模板匹配等
		return !(i.isArmorPattern());
	}), _armors.end());

	//全都判断不是装甲板
	if(_armors.empty())
	{
		_targetArmor.clear();

		if(_flag == ARMOR_LOCAL)
		{
			//cout << "Tracking lost" << endl;
			return _flag = ARMOR_LOST;
		}
		else
		{
			//cout << "No armor pattern detected." << endl;
			return _flag = ARMOR_NO;
		}
	}

	//calculate the final score
	for(auto & armor : _armors)
	{
		armor.finalScore = armor.sizeScore + armor.distScore + armor.rotationScore;
	}

	//choose the one with highest score, store it on _targetArmor
	std::sort(_armors.begin(), _armors.end(), [](const ArmorDescriptor & a, const ArmorDescriptor & b)
	{
		return a.finalScore > b.finalScore;
	});
	
	//选择得分最高的目标装甲板
	_targetArmor = _armors[0];

	//update the flag status	
	_trackCnt++;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
	vector<Point> intVertex;
	for(const auto& point : _targetArmor.vertex)
	{
		Point fuckPoint = point;
		intVertex.emplace_back(fuckPoint);
	}
   	cvex::showContour(_debugWindowName, _debugImg, _debugImg, intVertex, cvex::GREEN, -1, _roi.tl());
#endif //DEBUG_DETECTION || SHOW_RESULT

	return _flag = ARMOR_LOCAL;
}

bool ArmorDescriptor::isArmorPattern() const
{
//    // cut the central part of the armor
//    Mat regulatedImg;
//    if(type == BIG_ARMOR)
//    {
//        regulatedImg = frontImg(Rect(21, 0, 50, 50));
//    }
//    else
//    {
//        regulatedImg = frontImg;
//    }

//    resize(regulatedImg,regulatedImg, Size(regulatedImg.size().width / 2, regulatedImg.size().height / 2));
//    // copy the data to make the matrix continuous
//    Mat temp;
//    regulatedImg.copyTo(temp);
//    Mat data = temp.reshape(1, 1);

//    data.convertTo(data, CV_32FC1);

//    Ptr<SVM> svm = StatModel::load<SVM>("/home/nvidia/Robomaster/Robomaster2018/Armor/SVM3.xml");


//    int result = (int)svm->predict(data);
//    if(result == 1) return true;
//    else return false;
	
	
	// to test the svm, uncomment the code block above
	// and comment the code below
	return true;
}


const std::vector<cv::Point2f> ArmorDetector::getArmorVertex() const
{
	vector<cv::Point2f> realVertex;
	for(int i = 0; i < 4; i++)
	{
		realVertex.emplace_back(Point2f(_targetArmor.vertex[i].x + _roi.tl().x, 
										_targetArmor.vertex[i].y + _roi.tl().y));
	}
	return realVertex;
}

int ArmorDetector::getArmorType() const
{
	return _targetArmor.type;
}

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
void ArmorDetector::showDebugImg() const
{
	imshow(_debugWindowName, _debugImg);
}
#endif // DEBUG_DETECTION || SHOW_RESULT
}
