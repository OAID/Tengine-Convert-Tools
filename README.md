# Introduction

Tengine Convert Tool supports converting multi framworks' models into tmfile that suitable for Tengine and Tengine-Lite AI framework.
Since this tool relys on protobuf to resolve proto file of Caffe, ONNX, TensorFlow, TFLite and so on, it can only run under x86 Linux system.

## Install dependent libraries
For loading caffe model or TensorFlow model
``` 
    sudo apt install libprotobuf-dev protobuf-compiler
```

* if use the Fedora/CentOS ,use follow command instead.
```
    sudo dnf install protobuf-devel
    sudo dnf install boost-devel glog-devel
```

## Build Convert Tool
```
    mkdir build
    cmake ..
    make -j(cpu number) && make install
```

## Exection File

The exection should be under build/install/bin/ named as convert_tool

## Run Convert Tool

To run the convert tool, running as following command:
Note: The command examples are based on mobilenet model

### Caffe
`./install/bin/convert_tool -f caffe -p mobilenet_deploy.prototxt -m mobilenet.caffemodel -o mobilenet.tmfile`

### MxNet
`./install/bin/convert_tool -f mxnet -p mobilenet1_0-symbol.json -m mobilene1_0-0000.params -o mobileent.tmfile`

### ONNX
`./install/bin/convert_tool -f onnx -m mobilenet.onnx -o mobilenet.tmfile`

### TensorFlow
`./install/bin/convert_tool -f tensorflow -m mobielenet_v1_1.0_224_frozen.pb -o mobilenet.tmfile`

### TFLITE
`./install/bin/convert_tool -f tflite -m mobielenet.tflite -o mobilenet.tmfile`

### DarkNet: darknet only support for yolov3 model
`./install/bin/convert_tool -f darknet -p yolov3.cfg -m yolov3.weights -o yolov3.tmfile`

### NCNN
`./install/bin/convert_tool -f ncnn -p mobilenet.params -m mobilenet.bin -o mobilenet.tmfile`


# How to add self define operator

1. Adding operator's name defined file under operator/include directory that likes xxx.hpp and xxx_param.hpp (including operator's params)
2. Adding operator's memory allocation (calculate the memory) under operator/operator directory
3. Register operator in project operators' registery under operator/operator/plugin/init.cpp file
4. After adding operator definition, you need to add operator into model serializers, these files are under tools directory. There are multiply framework model serializers, finding file name and .cpp file under that corresponding framwork folder. Following the other operator's definition in that .cpp file