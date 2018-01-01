//
// Created by buyi on 18-1-1.
//

#ifndef CV_MVG_TIC_TOC_H
#define CV_MVG_TIC_TOC_H

#include <ctime>
#include <cstdlib>
#include <chrono>

namespace CV_MVG
{

class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()        //ms
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

} //namespace DSDTM

#endif //CV_MVG_TIC_TOC_H
