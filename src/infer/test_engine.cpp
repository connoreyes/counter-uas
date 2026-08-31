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

Logger gLogger;


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

    for (int i = 0; i < n; ++i){
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
            printf("%ld  ", shape.d[d]);
        }
        printf("\n");
    }


    // clean it all up at the end
    delete context;
    delete engine;
    delete runtime;



    return 0;
}