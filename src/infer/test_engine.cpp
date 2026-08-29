#include <cstdio>
#include <NvInfer.h>
#include <vector>
#include <fstream>

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

Logger logger;


int main(){
    std:: ifstream file("models/yolov8n_fp16.engine", std::ios::binary);
    if (!file){
        printf("failed to open engine file\n");
        return 1;
    }
    // there exist a cursor that exists at byte 0, so we seek/move that cursor to the end
    // seek to the end
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    // seek back to the beginning
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    printf("read %zu bytes\n", size);
    return 0;
}