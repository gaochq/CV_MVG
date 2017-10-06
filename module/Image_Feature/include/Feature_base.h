//
// Created by buyi on 17-10-6.
//

#ifndef CV_MVG_FEATURE_BASE_H
#define CV_MVG_FEATURE_BASE_H

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;

namespace CV_MVG
{
class Feature_base
{
public:
    Feature_base();
    ~Feature_base();

    // extract high quality orb keypoints
    void Extract_GoodPoints(cv::Mat imageA, cv::Mat imageB, vector<cv::Point2f> &PointsA, vector<cv::Point2f> &PointsB);
};
} //CV_MVG

#endif //CV_MVG_FEATURE_BASE_H
