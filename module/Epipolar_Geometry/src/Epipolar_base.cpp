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

    void Epipolar_base::Draw_EpipolarLines(cv::Mat F, vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB,
                                           cv::Mat &ImageA, cv::Mat &ImageB)
    {
        vector<cv::Vec3f> Epipolar_linesA, Epipolar_linesB;
        cv::Mat F_tr;

        // Get the epipolar lines: I' = Fx
        cv::computeCorrespondEpilines(cv::Mat(PointsA), 1, F, Epipolar_linesB);
        cv::computeCorrespondEpilines(cv::Mat(PointsB), 2, F, Epipolar_linesA);

        cv::RNG& rng = cv::theRNG();
        for (int k = 0; k < Epipolar_linesB.size(); ++k)
        {
            cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));    // Randomly generate color
            cv::line(ImageB,
                     cv::Point(0, -Epipolar_linesB[k][2] / Epipolar_linesB[k][1]),
                     cv::Point(ImageB.cols, -(Epipolar_linesB[k][2] + Epipolar_linesB[k][0] * ImageB.cols) / Epipolar_linesB[k][1]),
                     cv::Scalar(255, 255, 255));
            cv::line(ImageA,
                     cv::Point(0, -Epipolar_linesA[k][2] / Epipolar_linesA[k][1]),
                     cv::Point(ImageA.cols, -(Epipolar_linesA[k][2] + Epipolar_linesA[k][0] * ImageB.cols) / Epipolar_linesA[k][1]),
                     cv::Scalar(255, 255, 255));
        }

        vector<cv::KeyPoint> PointC, PointD;
        cv::KeyPoint::convert(PointsA, PointC);
        cv::KeyPoint::convert(PointsB, PointD);

        cv::drawKeypoints(ImageA, PointC, ImageA, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        cv::drawKeypoints(ImageB, PointD, ImageB, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    }

    // Triangulate single Feature
    /*
     *   |vP2 - P1 |
     *   |P0 - uP2 |
     *   |v'P'2-P'1| X=0
     *   |p'0-u'P'2|
     */
    void Epipolar_base::Triangulate_Feature(const cv::Point2f PointA, const cv::Point2f PointB, const cv::Mat P1,
                                    const cv::Mat P2, cv::Mat &X)
    {
        cv::Mat A(4, 4, CV_32F);

        A.row(0) = PointA.y*P1.row(2) - P1.row(1);
        A.row(1) = P1.row(0) - PointA.x*P1.row(2);
        A.row(2) = PointB.y*P2.row(2) - P2.row(1);
        A.row(3) = P2.row(0) - PointB.x*P2.row(2);

        // using SVD get the single solution
        cv::Mat u,w,vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);

        X = vt.row(3).t();
        X = X.rowRange(0, 3)/X.at<float>(3);
    }

    // Triangulate Features
    void Epipolar_base::Triangulate_Features(const vector<cv::Point2f> PointA, const vector<cv::Point2f> PointB,
                                            const cv::Mat P1, const cv::Mat P2, cv::Mat &X)
    {
        cv::Mat X_tmp(3, int(PointA.size()), CV_32FC1);
        for (int i = 0; i < PointA.size(); ++i)
        {
            cv::Mat X1;
            Triangulate_Feature(PointA[i], PointB[i], P1, P2, X1);
            X_tmp.col(i) = X1;
        }

        X = X_tmp.clone();
    }

    // Triangulate the points on the Normalized plane
    /*
     *  s1x_fre = s2x_curR + t
     *  s1x_fre = s2x_curr + t
     *  |x_fre^Tx_fre   -x_fre^Tx_curr ||s1|=|x_fre^Tt |
     *  |x_fre^Tx_curr  -x_curr^Tx_curr||s2| |x_curr^Tt|
     *
     */
    void Epipolar_base::Triangulate_Points(const Eigen::Vector3d Point_fre, const Eigen::Vector3d Point_cur,
                                           const Eigen::Isometry3d T, Eigen::Vector3d &X)
    {
        Eigen::Vector3d t = T.translation();
        Eigen::Vector3d Point_curr = T.rotation()*Point_cur;
        Eigen::Vector2d b = Eigen::Vector2d(t.dot(Point_fre), t.dot(Point_curr));

        double A[4];
        A[0] = Point_fre.dot(Point_fre);
        A[1] = Point_fre.dot(Point_curr);
        A[2] = -A[1];
        A[3] = Point_curr.dot(Point_curr);

        double d = A[0]*A[3] - A[1]*A[2];
        Eigen::Vector2d Lambdavec =
                Eigen::Vector2d(A[3]*b(0,0) - A[1]*b(0,1),
                                -A[2]*b(0,0) + A[0]*b(0,1));

        /* wait for test
        Eigen::Matrix4d A_tmp;
        A_tmp << Point_fre.dot(Point_fre), Point_fre.dot(Point_curr),
                -A[1], Point_curr.dot(Point_curr);
        Lambdavec = A_tmp.colPivHouseholderQr().solve(b);
        */

        Eigen::Vector3d PointA = Lambdavec(0, 0)*Point_fre;
        Eigen::Vector3d PointB = Lambdavec(0, 1)*Point_curr;
        X = (PointA + PointB)/2.0;
    }

    // Get Essential matrix from fundamental matrix
    // E = K^T*F*K
    Eigen::Matrix3d Epipolar_base::Solve_E_Martix(const Eigen::Matrix3d F, const Eigen::Matrix3d K)
    {
        Eigen::Matrix3d E;
        E = K.transpose()*F*K;
    }
} // namespace CV_MVG