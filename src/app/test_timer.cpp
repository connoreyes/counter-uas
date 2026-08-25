#include <cstdio>
#include <thread>
#include "timer.hpp"

int main(){
    Timer t;
    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    printf("slept 100ms, measured %.3f ms\n", t.stop());
    return 0;
}