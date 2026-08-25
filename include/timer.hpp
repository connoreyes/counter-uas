#pragma once
#include <chrono>

class Timer{
public:
    /*
    starts the timer, stores it in the class member start_
    */
    void start(){
        start_ = std::chrono::steady_clock::now();
    }
    /*
    creates an auto now and then subracts prev now
    @returns double
    */
    double stop(){
        auto now = std::chrono::steady_clock::now();
        // note that duration<...> is a template
        // duration<how its stored, what unit>
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }


private:
    // declare variable start_ as type time_point
    std::chrono::steady_clock::time_point start_;
};