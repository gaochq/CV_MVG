//
// Compute the Fundamental Matrix using 8 Point, 7 Point and 8 Point with ransac
//
#include "Solve_Fundamental.h"

using namespace std;

namespace CV_MVG
{
    Solve_F_Matrix::Solve_F_Matrix(Solve_method method) :
            Probe(0.99), Outlier_thld(1.25), Max_Iteration(100)
    {
        Method = method;
    }

    Solve_F_Matrix::~Solve_F_Matrix()
    {

    }

    vector<Eigen::Matrix3d> Solve_F_Matrix::EightPoint_F_Matrix(vector<cv::Point2f> PointsA,
                                                                vector<cv::Point2f> PointsB)
    {
        assert(PointsA.size()>=8 && PointsB.size()>=8 && PointsA.size()==PointsB.size());

        vector<Eigen::Matrix3d> F_Mat;
        // Normalize the keyPoints
        Eigen::Matrix3d Norm_MatA, Norm_MatB;
        vector<cv::Point2f> Norm_PointsA, Norm_PointsB;
        Norm_MatA = epipolar_base.Normalize_Points(PointsA, Norm_PointsA);
        Norm_MatB = epipolar_base.Normalize_Points(PointsB, Norm_PointsB);

        Eigen::MatrixXd VectorA, VectorB;
        epipolar_base.Point2Vector(Norm_PointsA, Norm_PointsB, VectorA, VectorB);

        // SVD the A matrix
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(VectorA.rows(), 9);
        for (int i = 0; i < VectorA.rows(); ++i)
        {
            A.row(i) << VectorB(i,0)*VectorB(i,0), VectorB(i,0)*VectorA(i,1), VectorB(i,0),
                    VectorB(i,1)*VectorA(i,0), VectorB(i,1)*VectorA(i,1), VectorB(i,1),
                    VectorA(i,0), VectorA(i,1), 1.0;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::MatrixXd V_A = svd_A.matrixV();
        Eigen::MatrixXd f = V_A.col(V_A.cols()-1);
        f.resize(3,3);

        // make the rank of F equals to 2
        Eigen::Matrix<double, 3, 3> F(f.transpose());
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_F(F, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::Matrix3d V_F = svd_F.matrixV(), U_F = svd_F.matrixU();
        Eigen::Matrix3d W_F = U_F.inverse()*F*V_F.transpose().inverse();
        W_F(2,2) = 0.0;
        F = U_F*W_F.diagonal().asDiagonal()*V_F.transpose();

        // relieve the normalize
        F = Norm_MatB.transpose()*F*Norm_MatA;
        F_Mat.push_back(F);

        return F_Mat;
    }

    vector<Eigen::Matrix3d> Solve_F_Matrix::EightPoint_F_Matrix(vector<std::pair<cv::Point2f, cv::Point2f> > Matches)
    {
        vector<cv::Point2f> PointsA, PointsB;
        for (int i = 0; i < Matches.size(); ++i)
        {
            PointsA.push_back(Matches[i].first);
            PointsB.push_back(Matches[i].second);
        }

        return EightPoint_F_Matrix(PointsA, PointsB);
    }

    vector<Eigen::Matrix3d> Solve_F_Matrix::SevenPoint_F_Matrix(vector<cv::Point2f> PointsA,
                                                                vector<cv::Point2f> PointsB)
    {
        assert(PointsA.size()==7 &&PointsA.size()==7);
        vector<Eigen::Matrix3d> F_Mat;
        // construct the constaint condition
        Eigen::MatrixXd VectorA, VectorB;
        epipolar_base.Point2Vector(PointsA, PointsB, VectorA, VectorB);
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(VectorA.rows(), 9);
        for (int i = 0; i < VectorA.rows(); ++i)
        {
            A.row(i) << VectorB(i,0)*VectorB(i,0), VectorB(i,0)*VectorA(i,1), VectorB(i,0),
                    VectorB(i,1)*VectorA(i,0), VectorB(i,1)*VectorA(i,1), VectorB(i,1),
                    VectorA(i,0), VectorA(i,1), 1.0;
        }

        // SVD the A matrix
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::MatrixXd V_A = svd_A.matrixV();

        Eigen::MatrixXd f1 = V_A.col(V_A.cols()-1);
        Eigen::MatrixXd f2 = V_A.col(V_A.cols()-2);

        //https://github.com/yueying/3DReconstruction/blob/6c3f475123df4f57be339230f28e5559278600e6/libs/multiview/src/solver_fundamental_kernel.cpp#L40
        double  a = f1(0, 0), j = f2(0, 0),
                b = f1(1, 0), k = f2(1, 0),
                c = f1(2, 0), l = f2(2, 0),
                d = f1(3, 0), m = f2(3, 0),
                e = f1(4, 0), n = f2(4, 0),
                f = f1(5, 0), o = f2(5, 0),
                g = f1(6, 0), p = f2(6, 0),
                h = f1(7, 0), q = f2(7, 0),
                i = f1(8, 0), r = f2(8, 0);

        double P[4] = {
                a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g,
                a*e*r + a*i*n + b*f*p + b*g*o + c*d*q + c*h*m + d*h*l + e*i*j + f*g*k -
                a*f*q - a*h*o - b*d*r - b*i*m - c*e*p - c*g*n - d*i*k - e*g*l - f*h*j,
                a*n*r + b*o*p + c*m*q + d*l*q + e*j*r + f*k*p + g*k*o + h*l*m + i*j*n -
                a*o*q - b*m*r - c*n*p - d*k*r - e*l*p - f*j*q - g*l*n - h*j*o - i*k*m,
                j*n*r + k*o*p + l*m*q - j*o*q - k*m*r - l*n*p };

        vector<double> P_coeffic(P, P+5);
        vector<double> roots;
        roots.resize(3);
        int num_roots = cv::solveCubic(P_coeffic, roots);

        f1.resize(3, 3);
        f2.resize(3, 3);

        for (int j = 0; j < num_roots; ++j)
        {
            F_Mat.push_back(roots[j]*f1.transpose() + (1 - roots[j])*f2.transpose());
        }

        return F_Mat;
    }

    vector<Eigen::Matrix3d> Solve_F_Matrix::SevenPoint_F_Matrix(vector<std::pair<cv::Point2f, cv::Point2f> > Matches)
    {
        vector<cv::Point2f> PointsA, PointsB;
        for (int i = 0; i < Matches.size(); ++i)
        {
            PointsA.push_back(Matches[i].first);
            PointsB.push_back(Matches[i].second);
        }

        return SevenPoint_F_Matrix(PointsA, PointsB);
    }


    Eigen::Matrix3d Solve_F_Matrix::Ransac_F_Matrix(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB,
                                                    cv::Mat &mask, Solve_method method)
    {
        assert(method==FM_8Point_Ransac | method==FM_7Point);

        Eigen::Matrix3d F_final;

        vector< std::pair<cv::Point2f, cv::Point2f> > Matches;
        for (int i = 0; i < PointsA.size(); ++i)
        {
            std::pair<cv::Point2f, cv::Point2f> Match;
            Match = make_pair(PointsA[i], PointsB[i]);
            Matches.push_back(Match);
        }

        long N = std::numeric_limits<long >::max();
        double Inner_Num = 0;
        double Sample_Num = 0;
        while (N > Sample_Num || Sample_Num<=Max_Iteration)
        {
            int Inner_times = 0;

            cv::Mat Tmp_mask(PointsA.size(), 1, 0);
            uchar* data = &Tmp_mask.data[0];
            vector<Eigen::Matrix3d> Sample_F;
            random_shuffle(Matches.begin(), Matches.end());
            if(method==FM_8Point_Ransac)
            {
                vector< std::pair<cv::Point2f, cv::Point2f> > Sample_Matches(Matches.begin(), Matches.begin()+8);
                Sample_F = EightPoint_F_Matrix(Sample_Matches);
            }
            else
            {
                vector< std::pair<cv::Point2f, cv::Point2f> > Sample_Matches(Matches.begin(), Matches.begin()+7);
                Sample_F = SevenPoint_F_Matrix(Sample_Matches);
            }

            for (int j = 0; j < Sample_F.size(); ++j)
            {
                Eigen::Matrix3d F_tmp = Sample_F[j];

                for (int i = 0; i < Matches.size(); ++i)
                {
                    float Dist = epipolar_base.Sampson_Distance(PointsA[i], PointsB[i], F_tmp);
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
                    F_final = F_tmp;
                }
            }
            double a = pow(Inner_Num/PointsA.size(), PointsA.size());
            double b = log(1.0-Probe)/log(1.0-a);
            N = static_cast<long>(b);

            Sample_Num++;
        }

        vector<cv::Point2f> PointsC, PointsD;
        for (int j = 0; j < PointsA.size(); ++j)
        {
            if(!mask.at<uchar>(j,0))
                continue;

            PointsC.push_back(PointsA[j]);
            PointsD.push_back(PointsB[j]);
        }

        if(method==FM_8Point_Ransac)
            F_final = EightPoint_F_Matrix(PointsC, PointsD).back();

        return F_final;
    }

    Eigen::Matrix3d Solve_F_Matrix::Solve(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, cv::Mat &mask)
    {
        Eigen::Matrix3d F;

        switch(Method)
        {
            case FM_8Point:
                F = EightPoint_F_Matrix(PointsA, PointsB).back();
                break;

            case FM_8Point_Ransac:
                F = Ransac_F_Matrix(PointsA, PointsB, mask, FM_8Point_Ransac);
                break;

            case FM_7Point:
                F = Ransac_F_Matrix(PointsA, PointsB, mask, FM_7Point);
                break;

            default:
                F.setIdentity();
                break;
        }

        return F;
    }

}

