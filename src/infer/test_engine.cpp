#include <cstdio>
#include <NvInfer.h>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

/*
Purpose: TensorRT needs a way to tell us when an error occurs
this is the class that will do that. It will tell us the severity
and what happened
*/
class Logger : public nvinfer1 ::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING){
            printf("[TRT]%s\n", msg);
        }
    }
};

Logger gLogger;

/*
Purpose: The og image is a 640x480 dimension, so we need to scale
both axes by the same factor that is 80, since 80+80=160+480=640, resulting in
640x640 image.
*/
cv::Mat letterbox(const cv::Mat& src, int out_w, int out_h, float& scale, int& pad_x, int& pad_y){
    scale = std::min(out_w / (float)src.cols, out_h / (float)src.rows);
    int new_w = (int)(src.cols * scale);
    int new_h = (int)(src.rows * scale);
    
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    cv::Mat canvas(out_h, out_w, CV_8UC3, cv::Scalar(114,114,114));

    pad_x = (out_w - new_w) / 2;
    pad_y = (out_h - new_h) / 2;

    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return canvas;

}


int main(){
    // read the file in binary
    std:: ifstream file("models/yolov8n_fp16.engine", std::ios::binary);
    if (!file){
        printf("failed to open engine file\n");
        return 1;
    }
    // there exist a cursor that exists at byte 0, so we seek/move that cursor to the end
    // seek to the end
    file.seekg(0, std::ios::end);
    // this tells us what the end byte is giving us the size
    size_t size = file.tellg();
    // seek back to the beginning
    file.seekg(0, std::ios::beg);
    // creates a container data of size(size)
    std::vector<char> data(size);
    // data doesnt hold a pointer so we do .data() to hand out the raw pointer to its storage
    file.read(data.data(), size);
    printf("read %zu bytes\n", size);

    // the deserializer, takes the logger so it can report problems
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if(!runtime){
        printf("failed to create runtime\n");
        return 1;
    }
    // bytes in, engine out
    // note: .data() again, it wants a raw pointer
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), size);
    if(!engine){
        printf("failed to deserialize engine \n");
        delete runtime;
        return 1;
    }
    // working state for the actual running inference
    nvinfer1::IExecutionContext* context = engine -> createExecutionContext();
    if(!context){
        printf("failed to create context\n");
        // note: runtime was created before engine, so it must outlive it
        delete engine;
        delete runtime;

        return 1;
    }

    printf("engine loaded \n");


    /*
    for following code purpose:
    we have an engine loaded, to run inference we need to give it memory,
    a place to put the input image, and a place to write results.
    instead of hardcoding and having silent failures, we will interogate
    the engine to find out what it wants
    */

   // how many tensors do we have, yolov8n is 2
    int n = engine -> getNbIOTensors();

    // vector for input meory
    std::vector<void*> buffers;

    void*  d_input = nullptr;
    void*  d_output = nullptr;
    size_t input_bytes = 0;
    size_t output_count = 0;


    for (int i = 0; i < n; ++i){
        // nullptr for cudamalloc
        void* ptr = nullptr;
        size_t count = 1;
        // for (i) we get ack a string like "images" or "output0"
        const char* name  = engine ->getIOTensorName(i);
        // what shape is it, returns Dims struct with two fields:
        // nbDims(how many dimensions)
        // d[](an array of the sizes)
        auto shape = engine -> getTensorShape(name);
        // "is this an input or output" returns enum value
        auto mode = engine->getTensorIOMode(name);

        // the ? : is a ternary, a compact if/else that produces a value
        // reads as: if mode equals kINPUT, use the string "input" otherwise "ouput"
        printf(" %s (%s)" , name, 
            mode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output");

        //prints each dimension, unkown ammount so we loop nbDims times
        for (int d = 0; d < shape.nbDims; ++d){
            // multiply dims
            count *= shape.d[d];
            printf("%ld  ", shape.d[d]);
        }
        // float 32
        size_t bytes = count * 4;
        
        cudaMalloc(&ptr, bytes);
        // tracking variables, since they die after for loop
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            d_input = ptr;
            input_bytes = bytes;
        } else {
            d_output = ptr;
            output_count = count;
        }
        bool ok = context -> setTensorAddress(name, ptr);
        buffers.push_back(ptr);

        printf("\n");
        printf("count=%zu bytes=%zu bind=%s\n", count, bytes, ok ? "ok" : "FAILED");
    }
    
    // we grab a frame here
    // open the camera, cap(0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()){
        printf("failed to open camera\n");
        return 1;
    }
    // proper dimensions
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()){
        printf("empty frame\n");
        return 1;
    }
    printf("frame: %dx%d\n", frame.cols, frame.rows);

    float scale;
    int pad_x, pad_y;
    cv:: Mat canvas = letterbox(frame, 640, 640, scale, pad_x, pad_y);
    printf("scale=%.3f pad_x=%d pad_y=%d canvas=%dx%d",
    scale, pad_x, pad_y, canvas.cols, canvas.rows);
    cv:imwrite ("letterbox.png", canvas);



    // create a stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // set mem to 0
    cudaMemset(d_input, 0, input_bytes);
    // host buffer for output
    std::vector<float> h_output(output_count);

    // run inference
    bool ran = context -> enqueueV3(stream);
    if (!ran){
        printf("enqueueV3 failed\n");
    }
    // Arguments: destination, source, size, direction, stream
    // must use .data() since its a vector, we extract the raw pointer from it
    // we do output_cout * 4 since we transform the elements into bytes, 1 element = 4 bytes
    cudaMemcpyAsync(h_output.data(), d_output, output_count * 4, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for ( int i = 0; i < 10; ++i){
        printf("%.4f ", h_output[i]);
    }







    cudaStreamDestroy(stream);
    // clean it all up at the end
    for (void* p: buffers){
        cudaFree(p);
    }
    delete context;
    delete engine;
    delete runtime;



    return 0;
}