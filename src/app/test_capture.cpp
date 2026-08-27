#include <cstdio>
#include "stat.hpp"
#include "timer.hpp"
#include <opencv2/opencv.hpp>

int main(){
    // Camera outputs YUYV (2 bytes/px). channels=3 means OpenCV
    // converted to BGR on the CPU for every frame. That conversion
    // is one of the costs the fused CUDA kernel will absorb.
    
    // opens device 0, /dev/video0
    cv::VideoCapture cap(0);
    // connection check
    if (!cap.isOpened()){
        printf("failed to open camera\n");
        return 1;
    }

    // a request to set width, height, and fps
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    // an image
    cv::Mat frame;
    // frame is now a matrix holding the image data
    cap >> frame;

    // checking to make sure data arrived
    if (frame.empty()) {
        printf("empty frame\n");
        return 1;
    }
    // simple print lines, width, height, values per pixel, and format
    // if the .cols and .rows are not what we set using cap.set
    // then we know there is an issue
    printf("frame: %dx%d channels=%d type=%d\n",
           frame.cols, frame.rows, frame.channels(), frame.type());

    Stat capture_stat;
    Timer c;

    for (int i = 0; i < 300; ++i){
        c.start();
        cap >> frame;
        capture_stat.add(c.stop());

        if(frame.empty()){
            printf("failed on iteration %d\n", i);
            break;
        }
    }
    printf("300 samples: p50=%2f p95=%2f p99=%2f\n",
    capture_stat.percentile(0.50), 
    capture_stat.percentile(0.95), 
    capture_stat.percentile(0.99));
    return 0;
}