//
// Created by buyi on 17-10-3.
//

#include "Epipolar_base.h"

namespace CV_MVG
{
    Epipolar_base::Epipolar_base()
    {

    }

    Epipolar_base::~Epipolar_base()
    {

    }

    // Normalize the keypoints
    Eigen::Matrix3d Epipolar_base::Normalize_Points(vector <cv::Point2f> Points, vector <cv::Point2f> &Norm_Points)
    {
        Eigen::Matrix3d transform;
        cv::Point2d Point_mean, Point_absMean;
        for (int i = 0; i < Points.size(); ++i)
        {
            Point_mean.x += Points[i].x/Points.size();
            Point_mean.y += Points[i].y/Points.size();
        }

        for (int j = 0; j < Points.size(); ++j)
        {
            Point_absMean.x += abs(Points[j].x - Point_mean.x)/Points.size();
            Point_absMean.y += abs(Points[j].y - Point_mean.y)/Points.size();
        }

        float sx = 1.0/Point_absMean.x;
        float sy = 1.0/Point_absMean.y;

        Norm_Points.resize(Points.size());
        for (int k = 0; k < Points.size(); ++k)
        {
            float a = sx*(Points[k].x - Point_mean.x);
            Norm_Points[k].x = sx*(Points[k].x - Point_mean.x);
            Norm_Points[k].y = sy*(Points[k].y - Point_mean.y);
        }
        transform << sx, 0, -Point_mean.x*sx,
                0, sy, -Point_mean.y*sy,
                0, 0, 1;
        return transform;
    }

    // Convert the keypoint from cv to eign
    Eigen::MatrixXd Epipolar_base::Point2Vector(vector <cv::Point2f> Points)
    {
        Eigen::MatrixXd Vector(Points.size(), 2);
        for (int i = 0; i < Points.size(); ++i)
        {
            Eigen::Vector2d Point(Points[i].x, Points[i].y);
            Vector.row(i) = Point;
        }

        return Vector;
    }

    void Epipolar_base::Point2Vector(vector <cv::Point2f> PointsA, vector <cv::Point2f> PointsB,
                                     Eigen::MatrixXd &VectorA, Eigen::MatrixXd &VectorB)
    {
        VectorA.resize(PointsA.size(), 2);
        VectorB.resize(PointsA.size(), 2);
        for (int i = 0; i < PointsA.size(); ++i)
        {
            Eigen::Vector2d Point(PointsA[i].x, PointsA[i].y);
            VectorA.row(i) = Point;
            Point << PointsB[i].x, PointsB[i].y;
            VectorB.row(i) = Point;
        }
    }

    double Epipolar_base::Sampson_Distance(cv::Point2f PointA, cv::Point2f PointB, Eigen::Matrix3d F)
    {
        double Sampson_Distance;
        Eigen::Vector3d Pointx, Pointy;

        Pointx << PointA.x, PointA.y, 1;
        Pointy << PointB.x, PointB.y, 1;
        float a = static_cast<float>(Pointy.transpose()*F*Pointx);
        float b = static_cast<float>(F.row(0)*Pointx);
        float c = static_cast<float>(F.row(1)*Pointx);
        float d = static_cast<float>(F.transpose().row(0)*Pointy);
        float e = static_cast<float>(F.transpose().row(1)*Pointy);

        Sampson_Distance = a*a/(b*b + c*c + d*d + e*e);

        return Sampson_Distance;
    }



} // namespace CV_MVG