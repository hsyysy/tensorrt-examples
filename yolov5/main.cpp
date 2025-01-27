#include <iostream>
#include <fstream>
#include <vector>

#include "NvInfer.h"

//extern "C"{
#include "utils.h"
#include "yolov5.h"
//}

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR) {
            std::cerr << "ERROR: " << msg << std::endl;
        } else if (severity == Severity::kWARNING) {
            std::cerr << "WARNING: " << msg << std::endl;
        } else {
            std::cout << "INFO:" << msg << std::endl;
        }
    }
} logger;

int main(int argc, char** argv){
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);

    // 打开 engine 文件
    std::string engineFile = "yolov5s_v6.1_noTF32_1b.engine";
    //std::string engineFile = "yolov5s_v6.1_int8_1b.engine";
    //std::string engineFile = "yolov5s_v6.1_fp16_1b.engine";
    std::ifstream engineFileStream(engineFile, std::ios::binary);
    if (!engineFileStream.is_open()) {
	    std::cerr << "Failed to open engine file!" << std::endl;
	    return -1;
    }

    // 获取文件大小
    engineFileStream.seekg(0, std::ios::end);
    size_t engineSize = engineFileStream.tellg();
    engineFileStream.seekg(0, std::ios::beg);

    // 读取文件内容到内存
    std::vector<char> modelData(engineSize);
    engineFileStream.read(modelData.data(), engineSize);
    engineFileStream.close();

    nvinfer1::ICudaEngine* engine = 
            runtime->deserializeCudaEngine(modelData.data(), modelData.size());

    int numInputs = engine->getNbIOTensors();

    std::vector<std::string> tensor_names;

    //std::vector<int> is_io;
    // 遍历所有的输入输出
    for (int i = 0; i < numInputs; ++i) {
        const char* bindingName = engine->getIOTensorName(i);
	/*
        if (engine->getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kINPUT) {
            is_io.push_back(0);
        } else if (engine->getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kOUTPUT) {
            is_io.push_back(1);
        } else {
            is_io.push_back(2);
        }
	*/
        tensor_names.push_back(bindingName);
    }
    std::vector<size_t> tensor_count(numInputs);

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    for(int i=0;i<numInputs;i++){
        nvinfer1::Dims dims = engine->getTensorShape(tensor_names[i].c_str());
        size_t input_size = 1;
        for(int j=0;j<dims.nbDims;j++)
            input_size *= dims.d[j];
        tensor_count[i] = input_size;
    }

    for(int i=0;i<80;i++) std::cout << "-";
    std::cout << std::endl;

    // get img path
    const char* img_path;
    if (argc > 1){
        img_path = argv[1];
    } else {
        img_path = "dog.jpg";
    }

    // read image
    int width, height, channels;
    unsigned char *img = stbi_load(img_path, &width, &height, &channels, 0);
    if (img == NULL) {
        std::cerr << "Error in loading the image" << std::endl;
        exit(1);
    }
    std::cout << "img: " << img_path << ", width = " << width << ", height = " << height << ", channels = " << channels << std::endl;

    nvinfer1::Dims dims = engine->getTensorShape(tensor_names[0].c_str());

    struct resize_info r_info;
    r_info.ori_w = width;
    r_info.ori_h = height;
    r_info.net_w = dims.d[3];
    r_info.net_h = dims.d[2];
    r_info.ratio_x = (float)r_info.net_w/r_info.ori_w;
    r_info.ratio_y = (float)r_info.net_h/r_info.ori_h;
    r_info.start_x = 0;
    r_info.start_y = 0;
    r_info.keep_aspect = true;

    std::vector<float> input_data(tensor_count[0],0.0f);
    pre_process(img, input_data.data(), &r_info);

    void* gpu_buffer[4];
    for(int i=0;i<numInputs;i++){
        cudaMalloc(&gpu_buffer[i],tensor_count[i]*sizeof(float));
        context->setTensorAddress(tensor_names[i].c_str(), gpu_buffer[i]);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(gpu_buffer[0], input_data.data(),
            tensor_count[0]*sizeof(float), cudaMemcpyHostToDevice, stream);

    context->enqueueV3(stream);

    int output_num = numInputs - 1;
    std::vector<std::vector<float>> output(output_num);
    for(int i=0;i<output_num;i++){
        output[i].resize(tensor_count[i+1]);
        cudaMemcpyAsync(output[i].data(),gpu_buffer[i+1],
                tensor_count[i+1]*sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    float* ptr[3];
    for(int i=0;i<output_num;i++)
        ptr[i] = output[i].data();
    post_process(ptr, img_path, img, &r_info);

    cudaStreamDestroy(stream);
    for(int i=0;i<numInputs;i++)
        cudaFree(gpu_buffer[i]);

    stbi_image_free(img);
    delete context;
    delete engine;
    delete runtime;
    return 0;
}
