//
// Created by buyi on 17-10-3.
//

#ifndef CV_MVG_EPIPOLAR_BASE_H
#define CV_MVG_EPIPOLAR_BASE_H

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

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

    // Normalize the keypoints
    Eigen::Matrix3d Normalize_Points(vector<cv::Point2f> Points, vector<cv::Point2f> &Norm_Points);

    // Convert the keypoint from cv to eigen
    Eigen::MatrixXd Point2Vector(vector<cv::Point2f> Points);

    // Convert the keypoint from cv to eigen
    void Point2Vector(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, Eigen::MatrixXd &VectorA, Eigen::MatrixXd &VectorB);

    // compute the Sampson distance
    double Sampson_Distance(cv::Point2f PointA, cv::Point2f PointB, Eigen::Matrix3d F);

    // draw the epipolar lines
    void Draw_EpipolarLines(cv::Mat F, vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, cv::Mat &ImageA, cv::Mat &ImageB);

    // Triangulate single Feature
    void Triangulate_Feature(const cv::Point2f PointA, const cv::Point2f PointB, const cv::Mat P1, const cv::Mat P2, cv::Mat &X);

    // Triangulate Features
    void Triangulate_Features(const vector<cv::Point2f> PointA, const vector<cv::Point2f> PointB, const cv::Mat P1, const cv::Mat P2, cv::Mat &X);

    // Triangulate the points on the Normalized plane
    void Triangulate_Points(const Eigen::Vector3d Point_fre, const Eigen::Vector3d Point_cur, const Eigen::Isometry3d T, Eigen::Vector3d &X);

    // Solve Essential Matrix
    Eigen::Matrix3d Solve_E_Martix(const Eigen::Matrix3d F, const Eigen::Matrix3d K);
};
} // namespace CV_MVG


#endif //CV_MVG_EPIPOLAR_BASE_H
