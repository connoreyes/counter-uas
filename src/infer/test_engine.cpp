#include <cstdio>
#include <NvInfer.h>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

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








    // clean it all up at the end
    for (void* p: buffers){
        cudaFree(p);
    }
    delete context;
    delete engine;
    delete runtime;



    return 0;
}