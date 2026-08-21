#pragma once
#include <cstddef>
#include <algorithm>

class Stat {
public:
    /*
    Purpose: records elapsed time, how long did this stage take?
    @double value - holds 512 most recent data values
    Wrap value once over 512 data values
    We wrap it by doing modulo over cap size (512)
    */
    void add(double value){
        // let head be first value
        samples_[head_] = value;
        // incr head
        head_ = (head_ + 1) % CAP;
        // incr count if no wrap yet
        if (count_ < CAP) count_++;
    }
    /*
    purpose: returns a value to determine how bad a frame is
    @double p - returns percent 
    */
    double percentile(double p){
        if (count_ == 0) return 0.0;
        // copy samples into scratch
        for ( size_t i = 0; i < count_ ; i ++){
            scratch_[i] = samples_[i];
        }
        std::sort(scratch_, scratch_ + count_);
        // safe guard
        size_t idx = (size_t)(count_ * p);
        if ( idx >= count_) idx = count_ -1;
        return scratch_[idx];
    }

private:
    static constexpr size_t CAP = 512;
    double samples_[CAP] = {};
    double scratch_[CAP] = {};
    size_t head_ = 0;
    size_t count_ = 0;
};