//
// Created by buyi on 17-10-3.
//

#ifndef CV_MVG_EPIPOLAR_BASE_H
#define CV_MVG_EPIPOLAR_BASE_H

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
class Epipolar_base
{
public:
    Epipolar_base();
    ~Epipolar_base();

    Eigen::Matrix3d Normalize_Points(vector<cv::Point2f> Points, vector<cv::Point2f> &Norm_Points); // Normalize the keypoints
    Eigen::MatrixXd Point2Vector(vector<cv::Point2f> Points); // Convert the keypoint from cv to eigen
    void Point2Vector(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB,
                      Eigen::MatrixXd &VectorA, Eigen::MatrixXd &VectorB); // Convert the keypoint from cv to eign

    double Sampson_Distance(cv::Point2f PointA, cv::Point2f PointB, Eigen::Matrix3d F); // compute the Sampson distance
};
} // namespace CV_MVG


#endif //CV_MVG_EPIPOLAR_BASE_H
