//
// Created by buyi on 17-10-6.
//

#include "Solve_Homography.h"

using namespace std;

namespace CV_MVG
{
    Solve_H_Matrix::Solve_H_Matrix():
    Probe(0.99), Outlier_thld(1.25), Max_Iteration(100)
    {

    }

    Solve_H_Matrix::~Solve_H_Matrix()
    {

    }

    // Solve the homography using DLT
    /*
     *  p2=Hp1  ==>
     *
     *  h1u1+h2v1+h3-h7u1u2-h8v1u2 = u2
     *  h4u1+h5v1+h6-h7u1u2-h8v1v2 = v2  ==>
     *
     *  |0 0 0 -u1 -v1 -1 u1v2 v1v2 v2| H=0
     *  |u1 v1 1 0 0 0 -u1u2 -v1u2 -u2|
     *
     *  h9 = 1
     */
    Eigen::Matrix3d Solve_H_Matrix::DLT_H_Matrix(const vector<cv::Point2f> PointsA,
                                                         const vector<cv::Point2f> PointsB)
    {
        assert(PointsA.size()>=4 && PointsB.size()>=4 && PointsA.size()==PointsB.size());

        // Normalize the keyPoints
        Eigen::Matrix3d Norm_MatA, Norm_MatB;
        vector<cv::Point2f> Norm_PointsA, Norm_PointsB;
        Norm_MatA = epipolar_base.Normalize_Points(PointsA, Norm_PointsA);
        Norm_MatB = epipolar_base.Normalize_Points(PointsB, Norm_PointsB);

        const int N = PointsA.size();
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2*N, 9);

        for (int i = 0; i < N; ++i)
        {
            const float u1 = Norm_PointsA[i].x;
            const float v1 = Norm_PointsA[i].y;
            const float u2 = Norm_PointsB[i].x;
            const float v2 = Norm_PointsB[i].y;

            A(2*i, 0) = 0.0;
            A(2*i, 1) = 0.0;
            A(2*i, 2) = 0.0;
            A(2*i, 3) = -u1;
            A(2*i, 4) = -v1;
            A(2*i, 5) = -1.0;
            A(2*i, 6) = u1*v2;
            A(2*i, 7) = v1*v2;
            A(2*i, 8) = v2;

            A(2*i+1, 0) = u1;
            A(2*i+1, 1) = v1;
            A(2*i+1, 2) = 1.0;
            A(2*i+1, 3) = 0.0;
            A(2*i+1, 4) = 0.0;
            A(2*i+1, 5) = 0.0;
            A(2*i+1, 6) = -u1*u2;
            A(2*i+1, 7) = -v1*u2;
            A(2*i+1, 8) = -u2;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd_A.matrixV();
        Eigen::MatrixXd h = V.col(V.cols() - 1);
        h.resize(3, 3);

        // Denormalization
        Eigen::Matrix3d H = Norm_MatB.inverse()*h.transpose()*Norm_MatA;

        return H;
    }

    Eigen::Matrix3d Solve_H_Matrix::DLT_H_Matrix(const vector<std::pair<cv::Point2f, cv::Point2f> > Matches)
    {
        vector<cv::Point2f> PointsA, PointsB;
        for (int i = 0; i < Matches.size(); ++i)
        {
            PointsA.push_back(Matches[i].first);
            PointsB.push_back(Matches[i].second);
        }

        return DLT_H_Matrix(PointsA, PointsB);
    }

    // Solve the homography using ransac
    Eigen::Matrix3d Solve_H_Matrix::Ransac_H_Matrix(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB,
                                                    cv::Mat &mask)
    {
        assert(PointsA.size()>=4 && PointsB.size()>=4 && PointsA.size()==PointsB.size());
        assert(mMethod==HM_DLT_Ransac);

        vector<std::pair<cv::Point2f, cv::Point2f> > Matches;
        for (int i = 0; i < PointsA.size(); ++i)
        {
            std::pair<cv::Point2f, cv::Point2f> Match;
            Match = make_pair(PointsA[i], PointsB[i]);
            Matches.push_back(Match);
        }

        long N = std::numeric_limits<long>::max();
        double Inner_Num = 0;
        double Sample_Num = 0;

        while (N > Sample_Num || Sample_Num <= Max_Iteration)
        {
            int Inner_times = 0;

            cv::Mat Tmp_mask(PointsA.size(), 1, 0);
            uchar *data = &Tmp_mask.data[0];
            Eigen::Matrix3d Sample_H;

            random_shuffle(Matches.begin(), Matches.end());
            vector<std::pair<cv::Point2f, cv::Point2f> > Sample_Matches(Matches.begin(), Matches.begin() + 4);

            Sample_H = DLT_H_Matrix(Sample_Matches);

            for (int i = 0; i < Matches.size(); ++i)
            {
                float Dist = epipolar_base.Sampson_Distance(PointsA[i], PointsB[i], Sample_H);
                if (Dist > Outlier_thld)
                {
                    Inner_times++;
                    data[i] = true;
                }
                else
                    data[i] = false;
            }

            if (Inner_times > Inner_Num)
            {
                Inner_Num = Inner_times;
                Tmp_mask.copyTo(mask);
            }

            double a = pow(Inner_Num / PointsA.size(), PointsA.size());
            double b = log(1.0 - Probe) / log(1.0 - a);
            N = static_cast<long>(b);

            Sample_Num++;
        }

        vector<cv::Point2f> PointsC, PointsD;
        for (int j = 0; j < PointsA.size(); ++j)
        {
            if (!mask.at<uchar>(j, 0))
                continue;

            PointsC.push_back(PointsA[j]);
            PointsD.push_back(PointsB[j]);
        }

        return DLT_H_Matrix(PointsC, PointsD);
    }

    //Final sovle function
    Eigen::Matrix3d Solve_H_Matrix::Solve(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, Solve_method method, cv::Mat &mask)
    {
        mMethod = method;
        Eigen::Matrix3d H;

        switch(mMethod)
        {
            case HM_DLT:
                H = DLT_H_Matrix(PointsA, PointsB);
                break;

            case HM_DLT_Ransac:
                H = Ransac_H_Matrix(PointsA, PointsB, mask);
                break;

            default:
                H.setIdentity();
                break;
        }

        return H;
    }




} //CV_MVG
