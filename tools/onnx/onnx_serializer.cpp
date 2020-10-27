/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <algorithm>
#include <vector>

#include "tengine_c_api.h"
#include "exec_attr.hpp"
#include "data_type.hpp"
#include "tengine_errno.hpp"
#include "operator_manager.hpp"
#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/flatten_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/gemm_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/reshape_param.hpp"
#include "operator/permute_param.hpp"
#include "operator/clip_param.hpp"
#include "operator/hardswish_param.hpp"
#include "operator/elu_param.hpp"
#include "operator/interp_param.hpp"
#include "operator/transpose_param.hpp"
#include "operator/slice_param.hpp"
#include "operator/split_param.hpp"
#include "operator/unsqueeze_param.hpp"
#include "operator/squeeze_param.hpp"
#include "operator/reducel2_param.hpp"
#include "operator/gather_param.hpp"
#include "operator/comparison_param.hpp"
#include "operator/unary_param.hpp"
#include "operator/logical_param.hpp"
#include "operator/lrn_param.hpp"
#include "operator/reduction_param.hpp"
#include "operator/pad_param.hpp"
#include "operator/expand_param.hpp"
#include "operator/argmax_param.hpp"
#include "operator/argmin_param.hpp"
#include "operator/log_softmax_param.hpp"
#include "operator/deconv_param.hpp"
#include "operator/scatter_param.hpp"
#include "operator/selu_param.hpp"
#include "operator/hardsigmoid_param.hpp"
#include "operator/tile_param.hpp"
#include "operator/cast_param.hpp"
#include "operator/depthtospace_param.hpp"
#include "operator/lstm_param.hpp"

#include "type_name.hpp"
#include "compiler.hpp"

#include "onnx_serializer.hpp"

namespace TEngine {

using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const onnx::NodeProto&)>;

bool OnnxSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if (file_list.size() != GetFileNum())
        return false;

    onnx::ModelProto model;

    if (!LoadModelFile(file_list[0].c_str(), model))
        return false;

    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "onnx");
    SetGraphConstTensorFile(graph, file_list[0]);
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_ONNX);

    return LoadGraph(model, graph);
}

bool OnnxSerializer::LoadModelFile(const char* fname, onnx::ModelProto& model)
{
    std::ifstream is(fname, std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        LOG_ERROR() << "cannot open file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);

    coded_input.SetTotalBytesLimit(1024 << 20, 512 << 20);

    bool ret = model.ParseFromCodedStream(&coded_input);

    is.close();

    if (!ret)
    {
        LOG_ERROR() << "onnx serializer: parse file: " << fname << " failed\n";
        set_tengine_errno(EINVAL);
        return false;
    }

    return ret;
}
static onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node, const char* key)
{
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key)
        {
            return attr.t();
        }
    }

    return onnx::TensorProto();
}

void OnnxSerializer::LoadConstNode(const onnx::GraphProto& onnx_graph, StaticGraph* graph)
{
    std::map<std::string, onnx::TensorProto> node_tensor;

    int node_count = onnx_graph.node_size();

    for (int i = 0; i < node_count; i++)
    {
        const onnx::NodeProto& node = onnx_graph.node(i);
        const std::string& op = node.op_type();
        if (op == "Constant")
        {
            onnx::TensorProto node_attr = get_node_attr_tensor(node, "value");
            node_tensor.insert(std::pair<std::string, onnx::TensorProto>(node.output(0), node_attr));
        }
    }
    if (node_tensor.size() == 0)
    {
        return;
    }
    for (int i = 0; i < node_count; i++)
    {
        const onnx::NodeProto& node = onnx_graph.node(i);

        const std::string& op = node.op_type();

        if (op == "Reshape" || op == "Gather")
        {
            const onnx::TensorProto& shape_tensor = node_tensor[node.input(1)];
            StaticTensor* tensor = CreateStaticConstTensor(graph, node.input(1));
            std::vector<int> dims;
            int dim_size = shape_tensor.dims_size();
            int tensor_size = 1;
            for (int l = 0; l < dim_size; l++)
            {
                tensor_size *= shape_tensor.dims(l);
            }
            if (shape_tensor.has_raw_data())
            {
                SetTensorDataType(tensor, DataType::GetTypeID("int"));
                tensor_size = tensor_size * sizeof(int64_t);
                SetTensorSize(tensor, tensor_size);
                int64_t* raw_data = ( int64_t* )shape_tensor.raw_data().data();
                int64_t* mem_buf = ( int64_t* )std::malloc(tensor_size);
                for (int i = 0; i < tensor_size / ( int )sizeof(int64_t); i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                dims.push_back(tensor_size / ( int )sizeof(int64_t));
                SetTensorDim(tensor, dims);
                SetConstTensorBuffer(tensor, mem_buf);
            }
            SetConstTensorFileLocation(tensor, -1, 0);
            StaticOp* op = CreateStaticOp(graph, "Const");
            StaticNode* node_create = CreateStaticNode(graph, GetTensorName(tensor));
            SetNodeOp(node_create, op);
            AddNodeOutputTensor(node_create, tensor);
        }
        if (op == "Div")
        {
            const onnx::TensorProto& shape_tensor = node_tensor[node.input(1)];
            StaticTensor* tensor = CreateStaticConstTensor(graph, node.input(1));
            std::vector<int> dims;
            int dim_size = shape_tensor.dims_size();
            int tensor_size = 1;
            for (int l = 0; l < dim_size; l++)
            {
                tensor_size *= shape_tensor.dims(l);
            }
            if (shape_tensor.has_raw_data())
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = tensor_size * sizeof(int64_t);
                SetTensorSize(tensor, tensor_size);
                int64_t* raw_data = ( int64_t* )shape_tensor.raw_data().data();
                int64_t* mem_buf = ( int64_t* )std::malloc(tensor_size);
                for (int i = 0; i < tensor_size / ( int )sizeof(int64_t); i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                dims.push_back(tensor_size / ( int )sizeof(int64_t));
                SetTensorDim(tensor, dims);
                SetConstTensorBuffer(tensor, mem_buf);
            }
            SetConstTensorFileLocation(tensor, -1, 0);
            StaticOp* op = CreateStaticOp(graph, "Const");
            StaticNode* node_create = CreateStaticNode(graph, GetTensorName(tensor));
            SetNodeOp(node_create, op);
            AddNodeOutputTensor(node_create, tensor);
        }
    }
}

bool OnnxSerializer::LoadConstTensor(StaticGraph* graph, const onnx::GraphProto& onnx_graph)
{
    int const_tensor_number = onnx_graph.initializer_size();

    LoadConstNode(onnx_graph, graph);

    for (int i = 0; i < const_tensor_number; i++)
    {
        const onnx::TensorProto& onnx_tensor = onnx_graph.initializer(i);

        StaticTensor* tensor = CreateStaticConstTensor(graph, onnx_tensor.name());
        std::vector<int> dims;

        int dim_size = onnx_tensor.dims_size();
        int tensor_size = 1;

        for (int l = 0; l < dim_size; l++)
        {
            tensor_size *= onnx_tensor.dims(l);
            dims.push_back(onnx_tensor.dims(l));
        }

        if (dims.empty())
        {
            dims.push_back(1);
        }

        SetTensorDim(tensor, dims);

        // Note: the const tensor layout will be set in operator load function

        if (onnx_tensor.has_raw_data())
        {
            if (onnx_tensor.data_type() == 7)
            {
                SetTensorDataType(tensor, DataType::GetTypeID("int"));
                tensor_size = sizeof(int64_t) * tensor_size;
                SetTensorSize(tensor, tensor_size);

                int64_t* mem_buf = ( int64_t* )std::malloc(tensor_size);
                int64_t* raw_data = ( int64_t* )onnx_tensor.raw_data().data();

                for (unsigned int i = 0; i < tensor_size / sizeof(int64_t); i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                SetConstTensorBuffer(tensor, mem_buf);
            }
            else
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = 4 * tensor_size;
                SetTensorSize(tensor, tensor_size);

                uint8_t* mem_buf = ( uint8_t* )std::malloc(tensor_size);
                uint8_t* raw_data = ( uint8_t* )onnx_tensor.raw_data().c_str();

                for (int i = 0; i < tensor_size; i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                SetConstTensorBuffer(tensor, mem_buf);
            }
        }
        else
        {
            if (onnx_tensor.data_type() == 1)
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                SetTensorSize(tensor, tensor_size * sizeof(float));

                float* mem_buf = ( float* )std::malloc(tensor_size * sizeof(float));
                const float* float_data = onnx_tensor.float_data().data();
                for (int i = 0; i < tensor_size; i++)
                    mem_buf[i] = float_data[i];

                SetConstTensorBuffer(tensor, mem_buf);
            }
            else if (onnx_tensor.data_type() == 7)
            {
                SetTensorDataType(tensor, DataType::GetTypeID("float32"));
                tensor_size = tensor_size * sizeof(int64_t);
                SetTensorSize(tensor, tensor_size);
                int64_t* raw_data = ( int64_t* )onnx_tensor.int64_data().data();
                int64_t* mem_buf = ( int64_t* )std::malloc(tensor_size);
                for (int i = 0; i < tensor_size / ( int )sizeof(int64_t); i++)
                {
                    mem_buf[i] = raw_data[i];
                }
                SetConstTensorBuffer(tensor, mem_buf);
            }
        }

        SetConstTensorFileLocation(tensor, -1, 0);

        /* Now, create the node .... */

        StaticOp* op = CreateStaticOp(graph, "Const");
        StaticNode* node = CreateStaticNode(graph, GetTensorName(tensor));
        SetNodeOp(node, op);

        AddNodeOutputTensor(node, tensor);
    }

    return true;
}

void OnnxSerializer::CreateInputNode(StaticGraph* graph, const onnx::GraphProto& onnx_graph)
{
    int input_number = onnx_graph.input_size();
    for (int i = 0; i < input_number; i++)
    {
        const onnx::ValueInfoProto& val = onnx_graph.input(i);

        if (FindConstTensor(graph, val.name()) != nullptr)
        {
            continue;
        }

        const onnx::TypeProto& type = val.type();

        const onnx::TypeProto::Tensor& tensor_type = type.tensor_type();

        const onnx::TensorShapeProto& shape = tensor_type.shape();

        int has_shape = 1;

        std::vector<int> dims;

        for (int i = 0; i < shape.dim_size(); i++)
        {
            const onnx::TensorShapeProto::Dimension& dim = shape.dim(i);

            if (dim.has_dim_param())
            {
                has_shape = 0;
                break;
            }

            dims.push_back(dim.dim_value());
        }

        StaticTensor* tensor = CreateStaticTensor(graph, val.name());

        SetTensorDataType(tensor, DataType::GetTypeID("float32"));

        if (has_shape)
            SetTensorDim(tensor, dims);

        StaticNode* node = CreateStaticNode(graph, val.name());
        StaticOp* op = CreateStaticOp(graph, "InputOp");

        SetNodeOp(node, op);

        AddNodeOutputTensor(node, tensor);

        /*add this node into graph input node list */

        AddGraphInputNode(graph, node);
    }
}

static bool onnx_skip_output_for_test(const std::string& op_type, int idx)
{
    if (op_type == "Dropout" && idx > 0)
        return true;
    return false;
}

bool OnnxSerializer::LoadNode(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
   
    for(int i = 0; i < onnx_node.input_size(); i++)
    {
        const std::string& input_name = onnx_node.input(i);

        StaticTensor* tensor = FindTensor(graph, input_name);
        StaticTensor* new_tensor = nullptr;
        if(node_name[tensor->name] != 0){
            if(tensor->dims.size() == 0){
                AddNodeInputTensor(node, tensor);
                continue;
            }
            std::string new_tensor_name  = tensor->name + "_" + std::to_string(node_name[tensor->name]);
            new_tensor = CreateStaticConstTensor(graph, new_tensor_name);
            std::vector<int> dims = tensor->dims;
            int dim_size = tensor->dims.size();
            int tensor_size = 1;
            for(int t = 0; t < dim_size; t++){
                tensor_size *= dims[t];
            }
            SetTensorDim(new_tensor, dims);
            SetTensorDataType(new_tensor, DataType::GetTypeID("float32"));
            tensor_size = 4*tensor_size;
            SetTensorSize(tensor, tensor_size);
            uint8_t* mem_buf = ( uint8_t* )std::malloc(tensor_size);
            uint8_t* raw_data = ( uint8_t* )GetConstTensorBuffer(tensor);
            for(int i = 0; i < tensor_size; i++)
            {
                mem_buf[i] = raw_data[i];
            }
            SetConstTensorBuffer(new_tensor, mem_buf);
            SetConstTensorFileLocation(new_tensor, -1, 0);
            StaticOp* op = CreateStaticOp(graph, "Const");
            StaticNode* new_node = CreateStaticNode(graph, GetTensorName(new_tensor));
            SetNodeOp(new_node, op);
            AddNodeOutputTensor(new_node, new_tensor);
            AddNodeInputTensor(node, new_tensor);
            node_name[tensor->name] = node_name[tensor->name] + 1; 
        } else {
            AddNodeInputTensor(node, tensor);
            node_name[tensor->name] = node_name[tensor->name] + 1;
        }
    }

    for(int i = 0; i < onnx_node.output_size(); i++)
    {
        const std::string& onnx_op_name = onnx_node.op_type();

        if(onnx_skip_output_for_test(onnx_op_name, i))
            continue;

        const std::string& output_name = onnx_node.output(i);

        StaticTensor* tensor = CreateStaticTensor(graph, output_name);

        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        AddNodeOutputTensor(node, tensor);
    }

    return true;
}
bool OnnxSerializer::LoadGraph(onnx::ModelProto& model, StaticGraph* graph)
{
    const onnx::GraphProto& onnx_graph = model.graph();

    SetGraphIdentity(graph, model.domain(), onnx_graph.name(), std::to_string(( int )model.model_version()));

    LoadConstTensor(graph, onnx_graph);
    CreateInputNode(graph, onnx_graph);

    int i;
    std::vector<std::string> no_supported_op;
    for (i = 0; i < onnx_graph.node_size(); i++)
    {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();

        if (!FindOpLoadMethod(onnx_op_name))
        {
            auto it = find(no_supported_op.begin(), no_supported_op.end(), onnx_op_name);
            if (it == no_supported_op.end())
            {
                if (onnx_op_name == "Constant")
                    continue;
                no_supported_op.push_back(onnx_op_name);
            }
            //       LOG_ERROR() << "cannot find load function for operator: " << onnx_op_name << "\n";
            //       continue;
        }
    }
    if (no_supported_op.size())
    {
        LOG_ERROR() << "These " << no_supported_op.size() << "op are not supported\n";
        LOG_ERROR() << "{";
        for (int j = 0; j < ( int )no_supported_op.size(); j++)
        {
            LOG_ERROR() << no_supported_op[j] << ",";
        }
        LOG_ERROR() << "}\n";

        return false;
    }
    for(int i = 0; i < onnx_graph.node_size(); i++){
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();
        if(onnx_op_name == "null" || onnx_op_name == "_zeros" || onnx_op_name == "constant")
            continue; 

        std::vector<std::string>::iterator iter=std::find(support_op.begin(), support_op.end(), onnx_op_name);
        if(iter==support_op.end()){
            std::vector<std::string>::iterator uniter=std::find(unsupport_op.begin(), unsupport_op.end(), onnx_op_name);
            if(uniter==unsupport_op.end()){
                unsupport_op.push_back(onnx_op_name);
            } else {
                continue;
            }
        } else {
            continue;
        }
    }
    if(unsupport_op.size() != 0){
        printf("These ops are not in onnx serializer: \n");
        for(int i = 0; i < (int)unsupport_op.size(); i++){
            printf("[ %s ]\n", unsupport_op[i].c_str());
        }
        printf("\n");
        printf("You may need use onnx simplifier first\n");
        return false;
    }
    for (i = 0; i < onnx_graph.node_size(); i++)
    {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();

        if (onnx_op_name == "Constant")
            continue;
        StaticNode* node = CreateStaticNode(graph, onnx_node.output(0));

        if (!LoadNode(graph, node, onnx_node))
            break;

        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(onnx_op_name));

        if (!op_func(graph, node, onnx_node))
            break;
    }

    if (i < onnx_graph.node_size())
        return false;

    return true;
}

/* Global functions to load indiviual operator */

static bool LoadOnnxConvolutionOp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "kernel_shape")
        {
            param.kernel_h = attr.ints(0);
            param.kernel_w = attr.ints(1);
        }
        else if (attr.name() == "strides")
        {
            param.stride_h = attr.ints(0);
            param.stride_w = attr.ints(1);
        }
        else if (attr.name() == "pads")
        {
            param.pad_h0 = attr.ints(0);
            param.pad_h1 = attr.ints(0);
            param.pad_w0 = attr.ints(1);
            param.pad_w1 = attr.ints(1);
        }
        else if (attr.name() == "group")
        {
            param.group = attr.i();
        }
        else if (attr.name() == "dilations")
        {
            param.dilation_h = attr.ints(0);
            param.dilation_w = attr.ints(0);
        }
        else
            LOG_ERROR() << node->name << " attr.name:"<< attr.name() << "\n";
    }

    /* update the input tensor data layout */

    for (int k = 0; k < onnx_node.input_size(); k++)
    {
        const std::string& input_name = onnx_node.input(k);
        StaticTensor* tensor = FindTensor(graph, input_name);
        if (k == 1)    // weight
        {
            const std::vector<int>& dim = GetTensorDim(tensor);
            /* onnx hide the output channel in weight ..*/
            param.output_channel = dim[0];
        }
    }
    StaticOp* op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxBN(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

    // get espilon

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "epsilon")
            param.eps = attr.f();
    }

    StaticOp* op = CreateStaticOp(graph, "BatchNormalization");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxRelu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
    param.negative_slope = 0.f;

    StaticOp* op = CreateStaticOp(graph, "ReLu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxTanh(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Tanh");
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxPooling(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));

    const std::string& onnx_op = onnx_node.op_type();

    if (onnx_op == "GlobalAveragePool")
    {
        param.global = 1;
        param.alg = kPoolAvg;
    }
    else if (onnx_op == "MaxPool" || onnx_op == "AveragePool")
    {
        param.global = 0;

        if (onnx_op == "AveragePool")
            param.alg = kPoolAvg;
        else
            param.alg = kPoolMax;

        for (int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);

            if (attr.name() == "kernel_shape")
            {
                param.kernel_h = attr.ints(0);
                param.kernel_w = attr.ints(1);
            }
            else if (attr.name() == "strides")
            {
                param.stride_h = attr.ints(0);
                param.stride_w = attr.ints(1);
            }
            else if (attr.name() == "pads")
            {
                param.pad_h0 = attr.ints(0);
                param.pad_h1 = attr.ints(0);
                param.pad_w0 = attr.ints(1);
                param.pad_w1 = attr.ints(1);
            }
        }
    }
    else
    {
        LOG_ERROR() << "UKNOWN POOLING: " << onnx_op << "\n";
        return false;
    }

    StaticOp* op = CreateStaticOp(graph, "Pooling");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxFlatten(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam("Flatten"));
    param.axis = 1;

    if (1 == onnx_node.attribute_size())
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(0);
        param.axis = attr.i();
    }

    StaticOp* op = CreateStaticOp(graph, "Flatten");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxGemm(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    GemmParam param = any_cast<GemmParam>(OpManager::GetOpDefParam("Gemm"));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "alpha")
            param.alpha = attr.f();
        else if (attr.name() == "beta")
            param.beta = attr.f();
        else if (attr.name() == "transA")
            param.transA = attr.i();
        else if (attr.name() == "transB")
            param.transB = attr.i();
    }

    StaticTensor* weight_tensor = FindTensor(graph, onnx_node.input(1));

    StaticTensor* bias_tensor = FindTensor(graph, onnx_node.input(2));

    if (param.transA)
    {
        StaticOp* op = CreateStaticOp(graph, "Gemm");

        SetOperatorParam(op, param);

        SetNodeOp(node, op);

        return true;
    }

    // create fc instead
    if (!param.transB)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];

        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        float* tmp = ( float* )malloc(k * n * sizeof(float));
        float* data = ( float* )GetConstTensorBuffer(weight_tensor);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }

        memcpy(data, tmp, n * k * sizeof(float));

        free(tmp);
    }

    if (param.alpha != 1)
    {
        float* data = ( float* )GetConstTensorBuffer(weight_tensor);
        int tensor_size = weight_tensor->dims[0] * weight_tensor->dims[1];

        for (int i = 0; i < tensor_size; i++)
            data[i] *= param.alpha;
    }

    if (param.beta != 1)
    {
        float* data = ( float* )GetConstTensorBuffer(bias_tensor);
        int tensor_size = weight_tensor->dims[0];

        for (int i = 0; i < tensor_size; i++)
            data[i] *= param.beta;
    }

    StaticOp* op = CreateStaticOp(graph, "FullyConnected");

    FCParam fc_param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));

    fc_param.num_output = weight_tensor->dims[0];

    SetOperatorParam(op, fc_param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxConcat(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            param.axis = attr.i();
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Concat");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxDropout(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Dropout");

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxAdd(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_SUM;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSoftmax(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Softmax");
    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            param.axis = attr.i();
        }
    }

    // param.axis = 1;

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxHardSwish(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Hardswish");

    HardswishParam param = any_cast<HardswishParam>(OpManager::GetOpDefParam("Hardswish"));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "alpha")
            param.alpha = attr.f();
        else if (attr.name() == "beta")
            param.beta = attr.f();
    }

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxElu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Elu");

    EluParam param = any_cast<EluParam>(OpManager::GetOpDefParam("Elu"));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "alpha")
            param.alpha = attr.f();
    }

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxPRelu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "PReLU");

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxInterp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Interp");

    InterpParam param = any_cast<InterpParam>(OpManager::GetOpDefParam("Interp"));

    if (onnx_node.input_size() == 1)
    {
        for (int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);
            if (attr.name() == "scales")
            {
                param.height_scale = attr.f();
                param.width_scale = attr.f();
            }
        }
    }
    else
    {
        const std::string& input_name = onnx_node.input(1);
        // std::cout<<"tensor name:"<<input_name<<"\n";
        StaticTensor* tensor = FindTensor(graph, input_name);
        float* data = ( float* )GetConstTensorBuffer(tensor);

        // int scales_size = tensor->dims[0];
        // printf("scale size:%d\n", scales_size);
        // printf("scale data:%f %f\n",data[0], data[1]);
        param.height_scale = data[2];
        param.width_scale = data[3];
    }

    std::string mode = "nearest";
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "mode")
        {
            mode = attr.s();
        }
    }

    if (mode == "nearest")
    {
        param.resize_type = 1;
    }
    else if (mode == "bilinear" || mode == "linear")
    {
        param.resize_type = 2;
    }

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxClip(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ClipParam param = any_cast<ClipParam>(OpManager::GetOpDefParam("Clip"));

    int size = onnx_node.attribute_size();

    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "max")
        {
            param.max = attr.f();
            // std::cout<<"max:"<<param.max<<std::endl;
        }
        else if (attr.name() == "min")
        {
            param.min = attr.f();
            // std::cout<<"min:"<<param.min<<std::endl;
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Clip");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxMul(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_PROD;

    for (int i = 0; i < onnx_node.input().size(); ++i)
    {
        StaticTensor* tensor = FindTensor(graph, onnx_node.input(i));
        std::vector<int> dims = tensor->dims;
        if (dims.size() == 0)
        {
            std::vector<int> new_dims;
            new_dims.push_back(1);
            SetTensorDim(tensor, new_dims);
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxDiv(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_DIV;

    for (int i = 0; i < onnx_node.input().size(); ++i)
    {
        StaticTensor* tensor = FindTensor(graph, onnx_node.input(i));
        std::vector<int> dims = tensor->dims;
        if (dims.size() == 0)
        {
            std::vector<int> new_dims;
            new_dims.push_back(1);
            SetTensorDim(tensor, new_dims);
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxFloor(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_FLOOR;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxTranspose(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    TransposeParam param = any_cast<TransposeParam>(OpManager::GetOpDefParam("Transpose"));
    const onnx::AttributeProto& attr = onnx_node.attribute(0);

    int size = attr.ints_size();
    for (int i = 0; i < size; i++)
    {
        param.tr_shape.push_back(attr.ints(i));
    }

    StaticOp* op = CreateStaticOp(graph, "Transpose");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReshape(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));

    StaticTensor* shape_tensor = FindTensor(graph, onnx_node.input(1));

    param.is_onnx = true;
    int size = shape_tensor->dims[0];
    int64_t* data = ( int64_t* )GetConstTensorBuffer(shape_tensor);
    for (int i = 0; i < size; i++)
    {
        param.re_shape.push_back(data[i]);
    }

    StaticOp* op = CreateStaticOp(graph, "Reshape");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxLeakyReLu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
    const onnx::AttributeProto& attr = onnx_node.attribute(0);
    param.negative_slope = attr.f();

    StaticOp* op = CreateStaticOp(graph, "ReLu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxSlice(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    SliceParam param = any_cast<SliceParam>(OpManager::GetOpDefParam("Slice"));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            param.axis = attr.ints(0);
        }
        else if (attr.name() == "ends")
        {
            long long end = attr.ints(0);
            if (end > INT_MAX)
                end = INT_MAX;
            param.end = ( int )end;
        }
        else if (attr.name() == "starts")
        {
            param.begin = attr.ints(0);
        }
    }
    param.iscaffe = false;
    param.ismxnet = false;
    param.isonnx = true;
    StaticOp* op = CreateStaticOp(graph, "Slice");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSigmod(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    // printf("Load Sigmod\n");
    StaticOp* op = CreateStaticOp(graph, "Sigmoid");
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSplit(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    SplitParam param = any_cast<SplitParam>(OpManager::GetOpDefParam("Split"));
    param.is_onnx = true;
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            param.axis = attr.i();
        }
        else if (attr.name() == "split")
        {
            int size = attr.ints_size();
            param.split_dim = size;
            for (int i = 0; i < size; i++)
            {
                param.split_sizes_.push_back(attr.ints(i));
            }
        }
    }

    param.is_caffe = false;

    StaticOp* op = CreateStaticOp(graph, "Split");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxExp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_EXP;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSub(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_SUB;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxMax(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Maximum");
    SetNodeOp(node, op);

    return true;
}
// to do support gather
static bool LoadOnnxGather(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    GatherParam param = any_cast<GatherParam>(OpManager::GetOpDefParam("Gather"));
    StaticTensor* indices_tensor = FindTensor(graph, onnx_node.input(1));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            param.axis = attr.i();
        }
    }
    int64_t* data = ( int64_t* )GetConstTensorBuffer(indices_tensor);
    param.indices_num = *data;
    param.is_onnx = true;
    // printf("Gather data: %d %d \n", param.axis, param.indices_num);

    StaticOp* op = CreateStaticOp(graph, "Gather");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

// To do support dims > 4 Onwer : ZP
static bool LoadOnnxSqueeze(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    SqueezeParam param = any_cast<SqueezeParam>(OpManager::GetOpDefParam("Squeeze"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            for (int i = 0; i < attr.ints_size(); i++)
            {
                if (0 == attr.ints(i))
                {
                    param.dim_0 = 1;
                }
                else if (1 == attr.ints(i))
                {
                    param.dim_1 = 1;
                }
                else if (2 == attr.ints(i))
                {
                    param.dim_2 = 1;
                }
                else if (3 == attr.ints(i))
                {
                    param.dim_3 = 1;
                }
            }
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Squeeze");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxUnsqueeze(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnsqueezeParam param = any_cast<UnsqueezeParam>(OpManager::GetOpDefParam("Unsqueeze"));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            for (int i = 0; i < attr.ints_size(); i++)
            {
                param.axises.push_back(attr.ints(i));
            }
        }
    }
    sort(param.axises.begin(), param.axises.end());
    StaticOp* op = CreateStaticOp(graph, "Unsqueeze");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxReduceL2(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReduceL2Param param = any_cast<ReduceL2Param>(OpManager::GetOpDefParam("ReduceL2"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            param.axis = attr.ints(0);    // TODO:Support muti axis
        }
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
    }
    StaticOp* op = CreateStaticOp(graph, "ReduceL2");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxMatMul(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticTensor* input_tensor = FindTensor(graph, onnx_node.input(0));
    StaticTensor* weight_tensor = FindTensor(graph, onnx_node.input(1));

    if (2 == input_tensor->dims.size() && weight_tensor->type == kConstTensor)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];

        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        float* tmp = ( float* )malloc(k * n * sizeof(float));
        float* data = ( float* )GetConstTensorBuffer(weight_tensor);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }
        }
        memcpy(data, tmp, n * k * sizeof(float));

        free(tmp);

        StaticOp* op = CreateStaticOp(graph, "FullyConnected");

        FCParam fc_param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));

        fc_param.num_output = weight_tensor->dims[0];

        SetOperatorParam(op, fc_param);

        SetNodeOp(node, op);

        return true;
    }

    StaticOp* op = CreateStaticOp(graph, "MatMul");
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxGreater(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ComparisonParam param = any_cast<ComparisonParam>(OpManager::GetOpDefParam("Comparison"));

    param.type = COMP_GREATER;
    StaticOp* op = CreateStaticOp(graph, "Comparison");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxEqual(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ComparisonParam param = any_cast<ComparisonParam>(OpManager::GetOpDefParam("Comparison"));

    param.type = COMP_EQUAL;
    StaticOp* op = CreateStaticOp(graph, "Comparison");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxLRN(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    LRNParam param = any_cast<LRNParam>(OpManager::GetOpDefParam("LRN"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
        {
            param.alpha = attr.f();    // TODO:Support muti axis
        }
        if (attr.name() == "beta")
        {
            param.beta = attr.f();
        }
        if (attr.name() == "bias")
        {
            param.k = attr.f();
        }
        if (attr.name() == "size")
        {
            param.local_size = attr.i();
        }
    }
    param.is_onnx = true;
    StaticOp* op = CreateStaticOp(graph, "LRN");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxNeg(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 1;

    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxMean(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Mean");

    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxMin(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Minimum");
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxOr(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    LogicalParam param = any_cast<LogicalParam>(OpManager::GetOpDefParam("Logical"));
    param.type = 1;

    StaticOp* op = CreateStaticOp(graph, "Logical");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxPad(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    PadParam param = any_cast<PadParam>(OpManager::GetOpDefParam("Pad"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "mode")
        {
            if (attr.s() == "constant")
            {
                param.mode = 0;
            }
            else if (attr.s() == "reflect")
            {
                param.mode = 1;
            }
            else
            {
                param.mode = 2;
            }
        }
        if (attr.name() == "pads")
        {
            param.pad_0_h = attr.ints(0);
            param.pad_0_w = attr.ints(4);
            param.pad_1_h = attr.ints(1);
            param.pad_1_w = attr.ints(5);
            param.pad_2_h = attr.ints(2);
            param.pad_2_w = attr.ints(6);
            param.pad_3_h = attr.ints(3);
            param.pad_3_w = attr.ints(7);
        }
        if (attr.name() == "value")
        {
            param.value = attr.f();
        }
    }
    StaticOp* op = CreateStaticOp(graph, "Pad");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxPow(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type = ELT_POW;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);

    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxExpand(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ExpandParam param = any_cast<ExpandParam>(OpManager::GetOpDefParam("Expand"));

    StaticTensor* shape_tensor = FindTensor(graph, onnx_node.input(1));
    int size = shape_tensor->dims[0];
    int64_t* data = ( int64_t* )GetConstTensorBuffer(shape_tensor);
    for (int i = 0; i < size; i++)
    {
        param.shape.push_back(data[i]);
    }

    StaticOp* op = CreateStaticOp(graph, "Expand");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxReduceMean(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));
    param.type = 1;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            int dim_size = attr.ints().size();
            if (dim_size == 1)
            {
                param.dim_0 = attr.ints(0);
            }
            else if (dim_size == 2)
            {
                param.dim_0 = attr.ints(0);
                param.dim_1 = attr.ints(1);
            }
            else if (dim_size == 3)
            {
                param.dim_0 = attr.ints(0);
                param.dim_1 = attr.ints(1);
                param.dim_2 = attr.ints(2);
            }
            else if (dim_size == 4)
            {
                param.dim_0 = attr.ints(0);
                param.dim_1 = attr.ints(1);
                param.dim_2 = attr.ints(2);
                param.dim_3 = attr.ints(3);
            }
        }
        else if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Reduction");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxReduceLogSum(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    param.type = 9;
    param.dim_0 = -2;
    param.dim_1 = -2;
    param.dim_2 = -2;
    param.dim_3 = -2;
    param.keepdim = 1;
    StaticOp* op = CreateStaticOp(graph, "Reduction");
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                param.dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
                param.dim_3 = attr_3;
            }
        }
    }
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReduceLogSumExp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    param.type = 10;
    param.dim_0 = -2;
    param.dim_1 = -2;
    param.dim_2 = -2;
    param.dim_3 = -2;
    param.keepdim = 1;
    StaticOp* op = CreateStaticOp(graph, "Reduction");
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                param.dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
                param.dim_3 = attr_3;
            }
        }
    }
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReduceMax(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    param.type = 4;
    param.dim_0 = -2;
    param.dim_1 = -2;
    param.dim_2 = -2;
    param.dim_3 = -2;
    param.keepdim = 1;
    StaticOp* op = CreateStaticOp(graph, "Reduction");
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                param.dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
                param.dim_3 = attr_3;
            }
        }
    }
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReduceMin(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    param.type = 5;
    param.dim_0 = -2;
    param.dim_1 = -2;
    param.dim_2 = -2;
    param.dim_3 = -2;
    param.keepdim = 1;
    StaticOp* op = CreateStaticOp(graph, "Reduction");
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                param.dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
                param.dim_3 = attr_3;
            }
        }
    }
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReduceProd(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    param.type = 6;
    param.dim_0 = -2;
    param.dim_1 = -2;
    param.dim_2 = -2;
    param.dim_3 = -2;
    param.keepdim = 1;
    StaticOp* op = CreateStaticOp(graph, "Reduction");
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                param.dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
                param.dim_3 = attr_3;
            }
        }
    }
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReduceSum(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    param.type = 0;
    param.dim_0 = -2;
    param.dim_1 = -2;
    param.dim_2 = -2;
    param.dim_3 = -2;
    param.keepdim = 1;
    StaticOp* op = CreateStaticOp(graph, "Reduction");
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                param.dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
                param.dim_3 = attr_3;
            }
        }
    }
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxReduceSumSquare(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam("Reduction"));

    param.type = 3;
    param.dim_0 = -2;
    param.dim_1 = -2;
    param.dim_2 = -2;
    param.dim_3 = -2;
    param.keepdim = 1;
    StaticOp* op = CreateStaticOp(graph, "Reduction");
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            param.keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                param.dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                param.dim_0 = attr_0;
                param.dim_1 = attr_1;
                param.dim_2 = attr_2;
                param.dim_3 = attr_3;
            }
        }
    }
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxAbs(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 0;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxAsin(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 12;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxAcos(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 13;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxAtan(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 14;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxAnd(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    LogicalParam param = any_cast<LogicalParam>(OpManager::GetOpDefParam("Logical"));
    StaticOp* op = CreateStaticOp(graph, "Logical");

    param.type = LOGICAL_AND;
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxArgmax(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "ArgMax");
    ArgMaxParam param = any_cast<ArgMaxParam>(OpManager::GetOpDefParam("ArgMax"));
    int size = onnx_node.attribute_size();
    param.axis = 0;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            param.axis = attr.i();
        if (attr.name() == "keepdims")
            param.keepdims = attr.i();
    }
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxArgmin(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "ArgMin");
    ArgMinParam param = any_cast<ArgMinParam>(OpManager::GetOpDefParam("ArgMin"));
    int size = onnx_node.attribute_size();
    param.axis = 0;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            param.axis = attr.i();
        if (attr.name() == "keepdims")
            param.keepdims = attr.i();
    }
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxCos(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 10;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxCeil(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 3;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxLog(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = 8;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxLess(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Comparison");
    ComparisonParam param = any_cast<ComparisonParam>(OpManager::GetOpDefParam("Comparison"));
    param.type = COMP_LESS;
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxLogSoftmax(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "LogSoftmax");
    LogSoftmaxParam param = any_cast<LogSoftmaxParam>(OpManager::GetOpDefParam("LogSoftmax"));
    int size = onnx_node.attribute_size();
    param.axis = 1;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            param.axis = attr.i();
    }
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxScatter(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Scatter");
    ScatterParam param = any_cast<ScatterParam>(OpManager::GetOpDefParam("Scatter"));
    int size = onnx_node.attribute_size();
    param.axis = 0;
    param.is_onnx = true;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            param.axis = attr.i();
    }
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxDeConvOp(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    DeconvParam param = any_cast<DeconvParam>(OpManager::GetOpDefParam("Deconvolution"));

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "kernel_shape")
        {
            param.kernel_h = attr.ints(0);
            param.kernel_w = attr.ints(1);
        }
        else if (attr.name() == "strides")
        {
            param.stride_h = attr.ints(0);
            param.stride_w = attr.ints(1);
        }
        else if (attr.name() == "output_padding")
        {
            param.output_pad_h0 = attr.ints(0);
            param.output_pad_w0 = attr.ints(1);
        }
        else if (attr.name() == "pads")
        {
            param.pad_h0 = attr.ints(0);
            param.pad_h1 = attr.ints(0);
            param.pad_w0 = attr.ints(1);
            param.pad_w1 = attr.ints(1);
        }
        else if (attr.name() == "group")
        {
            param.group = attr.i();
        }
        else if (attr.name() == "dilations")
        {
            param.dilation_h = attr.ints(0);
            param.dilation_w = attr.ints(0);
        }
        else
            LOG_ERROR() << "attr.name:" << attr.name() << "\n";
    }

    /* update the input tensor data layout */

    for (int k = 0; k < onnx_node.input_size(); k++)
    {
        const std::string& input_name = onnx_node.input(k);
        StaticTensor* tensor = FindTensor(graph, input_name);
        if (k == 1)    // weight
        {
            const std::vector<int>& dim = GetTensorDim(tensor);
            /* onnx hide the output channel in weight ..*/
            param.num_output = dim[1];
            param.kernel_h = dim[2];
            param.kernel_w = dim[3];
        }
    }
    StaticOp* op = CreateStaticOp(graph, "Deconvolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxShape(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Shape");
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxWhere(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    StaticOp* op = CreateStaticOp(graph, "Where");
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxSelu(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    SeluParam param = any_cast<SeluParam>(OpManager::GetOpDefParam("Selu"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
            param.alpha = attr.f();
        else if (attr.name() == "gamma")
            param.gamma = attr.f();
    }
    StaticOp* op = CreateStaticOp(graph, "Selu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxHardsigmoid(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    HardsigmoidParam param = any_cast<HardsigmoidParam>(OpManager::GetOpDefParam("Hardsigmoid"));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
            param.alpha = attr.f();
        else if (attr.name() == "beta")
            param.beta = attr.f();
    }
    StaticOp* op = CreateStaticOp(graph, "Hardsigmoid");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadOnnxTile(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    TileParam param = any_cast<TileParam>(OpManager::GetOpDefParam("Tile"));
    param.frame_flag = 1;

    /*
        for(int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);
            if(attr.name() == "frame_flag")
            {
                param.frame_flag = 1;
            }
            else if(attr.name() == "reps")
            {
                for(int i = 0; i < attr.ints_size();i++)
                {
                    param.reps.push_back(attr.ints(i));
                }
            }
        }
    */

    // sort(param.reps.begin(), param.reps.end());
    StaticOp* op = CreateStaticOp(graph, "Tile");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxCast(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    CastParam param = any_cast<CastParam>(OpManager::GetOpDefParam("Cast"));
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "to")
            param.type_to = attr.i();
    }
    param.type_from = 1;
    StaticOp* op = CreateStaticOp(graph, "Cast");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadOnnxDepthToSpace(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    DepthToSpaceParam param = any_cast<DepthToSpaceParam>(OpManager::GetOpDefParam("DepthToSpace"));
    for(int k = 0; k < onnx_node.attribute_size(); k++){
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "block_size"){
            param.block_size = attr.i();
        }
    }

    StaticOp* op = CreateStaticOp(graph, "DepthToSpace");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadOnnxLstm(StaticGraph* graph, StaticNode* node, const onnx::NodeProto& onnx_node)
{
    LSTMParam param = any_cast<LSTMParam>(OpManager::GetOpDefParam("LSTM"));
    int s_size;
    std::string lstm_type;
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "hidden_size")
            s_size = attr.i();
        if(attr.name() == "direction")
            lstm_type = attr.s();

    }
    if(lstm_type == "bidirectional"){
        param.algorithm = 0;
    } else {
        param.algorithm = 0;
    }
    param.mxnet_flag = 0;
    param.hidden_size = s_size;
    param.cell_size = s_size;

    StaticOp* op = CreateStaticOp(graph, "LSTM");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
// To register all op loader...
bool OnnxSerializerRegisterOpLoader(void)
{
    // first get the onnx_serializer object

    SerializerPtr serializer;

    if (!SerializerManager::SafeGet("onnx", serializer))
        return false;

    OnnxSerializer* p_onnx = dynamic_cast<OnnxSerializer*>(serializer.get());

    p_onnx->RegisterOpLoadMethod("Conv", op_load_t(LoadOnnxConvolutionOp));
    p_onnx->RegisterOpLoadMethod("Relu", op_load_t(LoadOnnxRelu));
    p_onnx->RegisterOpLoadMethod("MaxPool", op_load_t(LoadOnnxPooling));
    p_onnx->RegisterOpLoadMethod("GlobalAveragePool", op_load_t(LoadOnnxPooling));
    p_onnx->RegisterOpLoadMethod("AveragePool", op_load_t(LoadOnnxPooling));
    p_onnx->RegisterOpLoadMethod("Concat", op_load_t(LoadOnnxConcat));
    p_onnx->RegisterOpLoadMethod("Dropout", op_load_t(LoadOnnxDropout));
    p_onnx->RegisterOpLoadMethod("Softmax", op_load_t(LoadOnnxSoftmax));
    p_onnx->RegisterOpLoadMethod("BatchNormalization", op_load_t(LoadOnnxBN));
    p_onnx->RegisterOpLoadMethod("Add", op_load_t(LoadOnnxAdd));
    p_onnx->RegisterOpLoadMethod("Flatten", op_load_t(LoadOnnxFlatten));
    p_onnx->RegisterOpLoadMethod("Gemm", op_load_t(LoadOnnxGemm));
    p_onnx->RegisterOpLoadMethod("HardSwish", op_load_t(LoadOnnxHardSwish));
    p_onnx->RegisterOpLoadMethod("Elu", op_load_t(LoadOnnxElu));
    p_onnx->RegisterOpLoadMethod("Tanh", op_load_t(LoadOnnxTanh));
    p_onnx->RegisterOpLoadMethod("PRelu", op_load_t(LoadOnnxPRelu));
    p_onnx->RegisterOpLoadMethod("Upsample", op_load_t(LoadOnnxInterp));
    p_onnx->RegisterOpLoadMethod("Clip", op_load_t(LoadOnnxClip));
    p_onnx->RegisterOpLoadMethod("Mul", op_load_t(LoadOnnxMul));
    p_onnx->RegisterOpLoadMethod("Div", op_load_t(LoadOnnxDiv));
    p_onnx->RegisterOpLoadMethod("Floor", op_load_t(LoadOnnxFloor));
    p_onnx->RegisterOpLoadMethod("Transpose", op_load_t(LoadOnnxTranspose));
    p_onnx->RegisterOpLoadMethod("Reshape", op_load_t(LoadOnnxReshape));
    p_onnx->RegisterOpLoadMethod("LeakyRelu", op_load_t(LoadOnnxLeakyReLu));
    p_onnx->RegisterOpLoadMethod("Transpose", op_load_t(LoadOnnxTranspose));
    p_onnx->RegisterOpLoadMethod("Slice", op_load_t(LoadOnnxSlice));
    p_onnx->RegisterOpLoadMethod("Sigmoid", op_load_t(LoadOnnxSigmod));
    p_onnx->RegisterOpLoadMethod("Split", op_load_t(LoadOnnxSplit));
    p_onnx->RegisterOpLoadMethod("Exp", op_load_t(LoadOnnxExp));
    p_onnx->RegisterOpLoadMethod("Sub", op_load_t(LoadOnnxSub));
    p_onnx->RegisterOpLoadMethod("Unsqueeze", op_load_t(LoadOnnxUnsqueeze));
    p_onnx->RegisterOpLoadMethod("Squeeze", op_load_t(LoadOnnxSqueeze));
    p_onnx->RegisterOpLoadMethod("MatMul", op_load_t(LoadOnnxMatMul));
    p_onnx->RegisterOpLoadMethod("ReduceL2", op_load_t(LoadOnnxReduceL2));
    p_onnx->RegisterOpLoadMethod("Max", op_load_t(LoadOnnxMax));
    p_onnx->RegisterOpLoadMethod("Gather", op_load_t(LoadOnnxGather));
    p_onnx->RegisterOpLoadMethod("Greater", op_load_t(LoadOnnxGreater));
    p_onnx->RegisterOpLoadMethod("LRN", op_load_t(LoadOnnxLRN));
    p_onnx->RegisterOpLoadMethod("Neg", op_load_t(LoadOnnxNeg));
    p_onnx->RegisterOpLoadMethod("Mean", op_load_t(LoadOnnxMean));
    p_onnx->RegisterOpLoadMethod("Min", op_load_t(LoadOnnxMin));
    p_onnx->RegisterOpLoadMethod("Mul", op_load_t(LoadOnnxMul));
    p_onnx->RegisterOpLoadMethod("Or", op_load_t(LoadOnnxOr));
    p_onnx->RegisterOpLoadMethod("Pad", op_load_t(LoadOnnxPad));
    p_onnx->RegisterOpLoadMethod("Pow", op_load_t(LoadOnnxPow));
    p_onnx->RegisterOpLoadMethod("Equal", op_load_t(LoadOnnxEqual));
    p_onnx->RegisterOpLoadMethod("Expand", op_load_t(LoadOnnxExpand));
    p_onnx->RegisterOpLoadMethod("ReduceMean", op_load_t(LoadOnnxReduceMean));
    p_onnx->RegisterOpLoadMethod("ReduceLogSumExp", op_load_t(LoadOnnxReduceLogSumExp));
    p_onnx->RegisterOpLoadMethod("ReduceLogSum", op_load_t(LoadOnnxReduceLogSum));
    p_onnx->RegisterOpLoadMethod("ReduceMax", op_load_t(LoadOnnxReduceMax));
    p_onnx->RegisterOpLoadMethod("ReduceMin", op_load_t(LoadOnnxReduceMin));
    p_onnx->RegisterOpLoadMethod("ReduceProd", op_load_t(LoadOnnxReduceProd));
    p_onnx->RegisterOpLoadMethod("ReduceSum", op_load_t(LoadOnnxReduceSum));
    p_onnx->RegisterOpLoadMethod("ReduceSumSquare", op_load_t(LoadOnnxReduceSumSquare));
    p_onnx->RegisterOpLoadMethod("Abs", op_load_t(LoadOnnxAbs));
    p_onnx->RegisterOpLoadMethod("Acos", op_load_t(LoadOnnxAcos));
    p_onnx->RegisterOpLoadMethod("And", op_load_t(LoadOnnxAnd));
    p_onnx->RegisterOpLoadMethod("ArgMax", op_load_t(LoadOnnxArgmax));
    p_onnx->RegisterOpLoadMethod("ArgMin", op_load_t(LoadOnnxArgmin));
    p_onnx->RegisterOpLoadMethod("Asin", op_load_t(LoadOnnxAsin));
    p_onnx->RegisterOpLoadMethod("Atan", op_load_t(LoadOnnxAtan));
    p_onnx->RegisterOpLoadMethod("Ceil", op_load_t(LoadOnnxCeil));
    p_onnx->RegisterOpLoadMethod("Cos", op_load_t(LoadOnnxCos));
    p_onnx->RegisterOpLoadMethod("Less", op_load_t(LoadOnnxLess));
    p_onnx->RegisterOpLoadMethod("Log", op_load_t(LoadOnnxLog));
    p_onnx->RegisterOpLoadMethod("LogSoftmax", op_load_t(LoadOnnxLogSoftmax));
    p_onnx->RegisterOpLoadMethod("ConvTranspose", op_load_t(LoadOnnxDeConvOp));
    p_onnx->RegisterOpLoadMethod("Scatter", op_load_t(LoadOnnxScatter));
    p_onnx->RegisterOpLoadMethod("Shape", op_load_t(LoadOnnxShape));
    p_onnx->RegisterOpLoadMethod("Where", op_load_t(LoadOnnxWhere));
    p_onnx->RegisterOpLoadMethod("Selu", op_load_t(LoadOnnxSelu));
    p_onnx->RegisterOpLoadMethod("HardSigmoid", op_load_t(LoadOnnxHardsigmoid));
    p_onnx->RegisterOpLoadMethod("Tile", op_load_t(LoadOnnxTile));
    p_onnx->RegisterOpLoadMethod("Cast", op_load_t(LoadOnnxCast));
    p_onnx->RegisterOpLoadMethod("DepthToSpace", op_load_t(LoadOnnxDepthToSpace));
    p_onnx->RegisterOpLoadMethod("LSTM", op_load_t(LoadOnnxLstm));
    // p_onnx->RegisterOpLoadMethod("Constant", op_load_t(LoadOnnxConstant));
    return true;
}

}    // namespace TEngine
