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
 * Copyright (c) 2019, Open AI Lab
 * Author: bzhang@openailab.com
 *         cxl@openailab.com
 */

#include "paddle_serializer.hpp"
#include <set>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include "tengine_c_api.h"
#include "exec_attr.hpp"
#include "type_name.hpp"
#include "data_type.hpp"
#include "tengine_errno.hpp"
#include "static_graph.hpp"
#include "operator_manager.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "operator/conv_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/flatten_param.hpp"
#include "operator/scale_param.hpp"
#include "operator/gemm_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/interp_param.hpp"
#include "operator/resize_param.hpp"

namespace TEngine {

using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const PaddleNode& paddle_node)>;

void dump_pp_nodelist(std::vector<PaddleNode>& pp_nodelist)
{
    for (int i = 0; i < pp_nodelist.size(); i++)
    {
        PaddleNode node = pp_nodelist[i];
        fprintf(stderr, "idx: %d, op: %s \n", i, node.op.c_str());
        fprintf(stderr, "input tensor: \n");
        for (int j = 0; j < node.inputs.size(); j++)
        {
            fprintf(stderr, "   param: %s, \t name: %s \n", node.inputs[j].first.c_str(), node.inputs[j].second.c_str());
        }
        fprintf(stderr, "output tensor: %s \n", node.name.c_str());
    }
}

void dump_paramlist(std::vector<PaddleParam> paramlist)
{
    fprintf(stderr, "=======================dump paramlist==========================\n");
    for (int i = 0; i < paramlist.size(); i++)
    {
        PaddleParam param = paramlist[i];
        fprintf(stderr, "tensor name: %s \n", param.name.c_str());
        fprintf(stderr, "dims[ ");
        for (int j = 0; j < param.dims.size(); j++)
        {
            fprintf(stderr, "%d ", param.dims[j]);
        }
        fprintf(stderr, "]\n");
        
        float* data = (float*)param.raw_data;
        for (int j = 0; j < 10; j++)
        {
            fprintf(stderr, "%f ", data[j]);
        }
        fprintf(stderr, "\n\n");
    }
    fprintf(stderr, "=======================dump paramlist end=====================\n");
    
}

bool PaddleSerializer::LoadBinaryFile(const char* fname, std::vector<PaddleParam>& paramlist, paddle::framework::proto::ProgramDesc pp_net)
{
    if (pp_net.blocks_size() != 1)
    {
        LOG_ERROR() << "supported 1 block only.\n";
        return false;
    }
    // get vars seq
    paddle::framework::proto::BlockDesc block = pp_net.blocks(0);
    std::vector<std::string> vars;
    for (int i = 0; i < block.vars_size(); i++)
    {
        paddle::framework::proto::VarDesc var = block.vars(i);
        paddle::framework::proto::VarType var_type = var.type();
        int type = var_type.type();
        if (var.persistable() == 0 || type == 17 || type == 8 || var.name() == "feed" || var.name() == "fetch")
            continue;
        vars.push_back(var.name());
    }
    std::sort(vars.begin(), vars.end());
    
    // get param list
    std::ifstream fin(fname, std::ios::binary);
    std::istream& is = fin;
    for (int i = 0; i < vars.size(); i++)
    {
        // some useless read
        uint32_t version;
        is.read(reinterpret_cast<char *>(&version), sizeof(version));
        uint64_t lod_level;
        is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
        for (uint64_t j = 0; j < lod_level; j++)
        {
            uint64_t size;
            is.read(reinterpret_cast<char *>(&size), sizeof(size));
            std::vector<size_t> tmp(size / sizeof(size_t));
            is.read(reinterpret_cast<char *>(tmp.data()),
                    static_cast<std::streamsize>(size));
        }
        uint32_t version1;
        is.read(reinterpret_cast<char *>(&version1), sizeof(version1));

        // proto buffer
        int32_t size;
        is.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::unique_ptr<char[]> buff(new char[size]);
        is.read(reinterpret_cast<char*>(buff.get()), size);
        paddle::framework::proto::VarType::TensorDesc desc;
        desc.ParseFromArray(buff.get(), size);

        PaddleParam param;
        param.name = vars[i];
        int elem_num = 1;
        for (int j = 0; j < desc.dims_size(); j++)
        {
            elem_num *= desc.dims(j);
            param.dims.push_back(desc.dims(j));
        }
        param.dim_size = elem_num;
        // read tensor
        void* buf;
        size_t buf_size = 0;
        switch (desc.data_type())
        {
        case 2:
        case 5:
            buf_size = elem_num * sizeof(float);
            break;
        
        default:
            LOG_ERROR() << "data type is not fp32、int32 !!!! \n";
            return false;
            break;
        }
        param.data_len = buf_size;
        buf = new char[buf_size];
        is.read(static_cast<char*>(buf), buf_size);
        param.raw_data = buf;
        paramlist.push_back(param);
    }
    
    return true;
}

bool PaddleSerializer::LoadTextFile(const char* fname, paddle::framework::proto::ProgramDesc& pp_net)
{
    // load binary
    std::ifstream is(fname, std::ios::in | std::ios::binary);
    if (!is.is_open())
    {
        LOG_ERROR() << "cannot open file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }
    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);
    coded_input.SetTotalBytesLimit(512 << 20, 64 << 20);
    bool ret = pp_net.ParseFromCodedStream(&coded_input);
    is.close();

    if (!ret)
    {
        LOG_ERROR() << "paddle serializer: parse file: " << fname << " failed\n";
        set_tengine_errno(EINVAL);
        return false;
    }

    return ret;
}


bool PaddleSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if (file_list.size() != GetFileNum())
        return false;

    paddle::framework::proto::ProgramDesc pp_net;
    if (!LoadTextFile(file_list[0].c_str(), pp_net))
    {
        LOG_ERROR() << "Parse text file " << file_list[0].c_str() << " failed\n";
        return false;
    }

    std::vector<PaddleParam> paramlist;
    if (!LoadBinaryFile(file_list[1].c_str(), paramlist, pp_net))
    {
        LOG_ERROR() << "Parse binary file " << file_list[1].c_str() << " failed\n";
        return false;
    }

    SetGraphSource(graph, file_list[1]);
    SetGraphSourceFormat(graph, "paddle");
    SetGraphConstTensorFile(graph, file_list[1]);
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_PADDLE);

    return LoadGraph(pp_net, paramlist, graph);
}

bool PaddleSerializer::ConstructGraph(paddle::framework::proto::ProgramDesc& pp_net, std::vector<PaddleNode>& pp_nodelist)
{
    int block_size = pp_net.blocks_size();
    std::vector<std::string> fake_nodes;
    for (int i = 0; i < block_size; i++)
    {
        paddle::framework::proto::BlockDesc block = pp_net.blocks(i);
        for (int j = 0; j < block.ops_size(); j++)
        {
            PaddleNode node;
            paddle::framework::proto::OpDesc op = block.ops(j);

            node.op_desc = op;
            node.op = op.type();
            
            bool fake_node = false;
            for (int k = 0; k < op.inputs_size(); k++)
            {
                paddle::framework::proto::OpDesc::Var var = op.inputs(k);
                if (var.arguments_size() == 0)
                    continue;
                std::pair<std::string, std::string> input;
                input.first = var.parameter();
                for (int idx = 0; idx < var.arguments_size(); idx++)
                {
                    input.second = var.arguments(idx);
                    node.inputs.push_back(input);
                }

                auto in_fake_nodes = find(fake_nodes.begin(), fake_nodes.end(), input.second);
                if (in_fake_nodes != fake_nodes.end())
                {
                    fake_node = true;
                    for (int k = 0; k < op.outputs_size(); k++)
                    {
                        paddle::framework::proto::OpDesc::Var var = op.outputs(k);
                        std::string name = var.parameter();
                        if (name == "Y" || name == "Output" || name == "Out")
                            fake_nodes.push_back(var.arguments(0));
                    }
                }    
            }
            if (fake_node)
                continue;

            // fake node
            if (node.inputs.empty())
            {
                for (int k = 0; k < op.outputs_size(); k++)
                {
                    paddle::framework::proto::OpDesc::Var var = op.outputs(k);
                    std::string name = var.parameter();
                    if (name == "Y" || name == "Output" || name == "Out")
                        fake_nodes.push_back(var.arguments(0));
                }
                continue;
            }

            node.name = "";
            for (int k = 0; k < op.outputs_size(); k++)
            {
                paddle::framework::proto::OpDesc::Var var = op.outputs(k);
                std::string name = var.parameter();
                if (name == "Y" || name == "Output" || name == "Out")
                    node.name = var.arguments(0);
            }
            if (node.name == "") 
            {
                LOG_ERROR() << "Parse op output name failed! Check and try again! \n";
                return false;
            }
            pp_nodelist.push_back(node);
        }
        
    }
    return true;
}

bool PaddleSerializer::LoadConstTensor(const std::vector<PaddleParam>& paramlist, StaticGraph* graph)
{
    int const_tensor_num = paramlist.size();
    for (int i = 0; i < const_tensor_num; i++)
    {
        const PaddleParam& pp_tensor = paramlist.at(i);

        // create tensor
        StaticTensor* tensor = CreateStaticConstTensor(graph, pp_tensor.name);
        std::vector<int> dims = pp_tensor.dims;
        SetTensorDim(tensor, dims);
        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        SetTensorSize(tensor, pp_tensor.data_len);
        uint8_t* mem_buf = (uint8_t*)std::malloc(pp_tensor.data_len);
        uint8_t* raw_data = (uint8_t*)pp_tensor.raw_data;
        memcpy(mem_buf, raw_data, pp_tensor.data_len);
        SetConstTensorBuffer(tensor, mem_buf);
        SetConstTensorFileLocation(tensor, -1, 0);

        // create node
        StaticOp* op = CreateStaticOp(graph, "Const");
        StaticNode* node = CreateStaticNode(graph, GetTensorName(tensor));
        SetNodeOp(node, op);
        AddNodeOutputTensor(node, tensor);

        delete raw_data;
    }
    return true;
}

static bool GetAllTensorDims(paddle::framework::proto::ProgramDesc& pp_net, std::map<std::string, std::vector<int>>& all_tensor_dims)
{
    // get all tensor dims
    paddle::framework::proto::BlockDesc block = pp_net.blocks(0);
    for (unsigned int i = 0; i < block.vars_size(); i++)
    {
        std::vector<int> dims;
        paddle::framework::proto::VarDesc var = block.vars(i);
        paddle::framework::proto::VarType var_type = var.type();
        if (var_type.has_lod_tensor())
        {
            paddle::framework::proto::VarType::LoDTensorDesc lod_tensor = var_type.lod_tensor();
            paddle::framework::proto::VarType::TensorDesc tensor = lod_tensor.tensor();
            for (int j = 0; j < tensor.dims_size(); j++)
            {
                dims.push_back(tensor.dims(j) == -1 ? 1 : tensor.dims(j));
            }
        }
        all_tensor_dims[var.name()] = dims;
    }

    return true;
}

bool PaddleSerializer::CreateInputNode(std::map<std::string, std::vector<int>>& all_tensor_dims, std::vector<PaddleNode>& nodelist, StaticGraph* graph)
{
    // get feed op
    if (nodelist.size() < 0 || nodelist[0].op != "feed")
    {
        LOG_ERROR() << "CreateInputNode: pp nodelist is null or the first op is not feed! \n";
        return false;
    }
    PaddleNode feed_node = nodelist[0];
    std::string tensor_name = feed_node.name;

    // get input tensor dims
    std::vector<int> dims;
    dims = all_tensor_dims[tensor_name];

    // create input node
    StaticTensor* tensor = CreateStaticTensor(graph, tensor_name);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    SetTensorDim(tensor, dims);

    StaticNode* node = CreateStaticNode(graph, tensor_name);
    StaticOp* op = CreateStaticOp(graph, "InputOp");
    SetNodeOp(node, op);
    AddNodeOutputTensor(node, tensor);

    /*add this node into graph input node list */
    AddGraphInputNode(graph, node);
    
    return true;
}

bool PaddleSerializer::LoadNode(StaticGraph* graph, StaticNode* node, PaddleNode& pp_node, std::map<std::string, std::vector<int>>& all_tensor_dims)
{
    int input_num = pp_node.inputs.size();
    if (input_num <= 0)
        return false;

    // add var input
    auto input_ir = pp_node.inputs.begin();
    while (input_ir != pp_node.inputs.end())
    {
        if ((*input_ir).first == "Input" || (*input_ir).first == "X")
            break;
        input_ir++;
    }
    if (input_ir == pp_node.inputs.end())
    {
        LOG_ERROR() << "Load node: " << node->name << " ,input tensor not find!\n";
        return false;
    }
    
    std::string tensor_name = (*input_ir).second;
    StaticTensor* in_tensor = FindTensor(graph, tensor_name);
    if (in_tensor == nullptr)
        return true;
    AddNodeInputTensor(node, in_tensor);
    pp_node.inputs.erase(input_ir);
    
    // add const input
    // [sc, bias, mean, var] --> [2, 0, 1, 3]
    input_num = pp_node.inputs.size();
    if (pp_node.op == "batch_norm")
    {
        std::vector<int> add_seq{2, 0, 1, 3};
        for (int i = 0; i < input_num; i++)
        {
            std::string tensor_name = pp_node.inputs[add_seq[i]].second;
            StaticTensor* tensor = FindTensor(graph, tensor_name);
            AddNodeInputTensor(node, tensor);
        }
    }
    else
    {
        for (int i = 0; i < input_num; i++)
        {
            std::string tensor_name = pp_node.inputs[i].second;
            StaticTensor* tensor = FindTensor(graph, tensor_name);
            AddNodeInputTensor(node, tensor);
        }
    }
    
    const std::string& output_name = pp_node.name;
    StaticTensor* out_tensor = CreateStaticTensor(graph, output_name);
    std::vector<int> dims = all_tensor_dims[output_name];
    if (!dims.empty())
        SetTensorDim(out_tensor, dims);
    SetTensorDataType(out_tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, out_tensor);

    return true;
}

bool PaddleSerializer::LoadGraph(paddle::framework::proto::ProgramDesc& pp_net, const std::vector<PaddleParam>& paramlist, StaticGraph* graph)
{
    SetGraphIdentity(graph, "paddle", "paddle-paddle", "0");
    
    std::vector<PaddleNode> pp_nodelist;

    // construct pp nodelist
    if(!ConstructGraph(pp_net, pp_nodelist))
        return false;

    if(!LoadConstTensor(paramlist, graph))
        return false;

    std::map<std::string, std::vector<int>> all_tensor_dims;
    if(!GetAllTensorDims(pp_net, all_tensor_dims))
        return false;
    
    if(!CreateInputNode(all_tensor_dims, pp_nodelist, graph))
        return false;

    unsigned int i;
    std::vector<std::string> no_supported_op;
    for(i = 0; i < pp_nodelist.size(); i++)
    {
        PaddleNode node = pp_nodelist.at(i);
        if(node.op == "feed" || node.op == "fetch")
            continue;
        if(!FindOpLoadMethod(node.op))
        {
            auto it = find(no_supported_op.begin(),no_supported_op.end(),node.op);
            if(it == no_supported_op.end())
                no_supported_op.push_back(node.op);
        }
    }
    if(no_supported_op.size())
    {
        LOG_ERROR() << "These " <<no_supported_op.size() <<" ops are not supported \n";
        LOG_ERROR() << "{";
        for(int j = 0; j < (int)no_supported_op.size(); j++)
        {
            LOG_ERROR() << no_supported_op[j] <<",";
        }
        LOG_ERROR() << "}\n";
        return false;
    }

    for ( i = 0; i < pp_nodelist.size(); i++)
    {
        PaddleNode pp_node = pp_nodelist.at(i);
        if (pp_node.op == "feed" || pp_node.op == "fetch")
            continue;
        
        StaticNode* node = CreateStaticNode(graph, pp_node.name);
        if(!LoadNode(graph, node, pp_node, all_tensor_dims))
            return false;

        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(pp_node.op));

        if(!op_func(graph, node, pp_node))
            break;
    }

    if(i < pp_nodelist.size())
        return false;
    
    return true;
}

static bool LoadPaddleConvolution(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "dilations")
        {
            param.dilation_h = attr.ints(0);
            param.dilation_w = attr.ints(1);
        }
        if (name == "paddings")
        {
            param.pad_h0 = attr.ints(0);
            param.pad_h1 = attr.ints(0);
            param.pad_w0 = attr.ints(1);
            param.pad_w1 = attr.ints(1);
        }
        if (name == "strides")
        {
            param.stride_h = attr.ints(0);
            param.stride_w = attr.ints(1);
        }
    }
    
    StaticTensor* input = GetNodeInputTensor(graph, node, 0);
    param.input_channel = input->dims[1];

    StaticTensor* kernel = GetNodeInputTensor(graph, node, 1);
    param.kernel_h = kernel->dims[2];
    param.kernel_w = kernel->dims[3];

    StaticTensor* output = GetNodeOutputTensor(graph, node, 0);
    param.output_channel = output->dims[1];

    param.group = 1;
    if(pp_node.op == "depthwise_conv2d")
    {
        param.group = param.input_channel;
        param.input_channel = 1;
    }

    StaticOp* op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleBatchNorm(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "epsilon")
        {
            param.eps = attr.f();
        }
    }
    
    param.caffe_flavor = 0;

    StaticOp* op = CreateStaticOp(graph, "BatchNormalization");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadOpPaddleElemwiseAdd(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    param.type = ELT_SUM;
    param.caffe_flavor = 0;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddlePooling(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    bool adaptive_pool = false;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "pooling_type")
        {
            if (attr.s() == "avg")
                param.alg = kPoolAvg;
            if (attr.s() == "max")
                param.alg = kPoolMax;
        }
        if (name == "paddings")
        {
            param.pad_h0 = attr.ints(0);
            param.pad_h1 = attr.ints(0);
            param.pad_w0 = attr.ints(1);
            param.pad_w1 = attr.ints(1);
        }
        if (name == "strides")
        {
            param.stride_h = attr.ints(0);
            param.stride_w = attr.ints(1);
        }
        if (name == "ksize")
        {
            param.kernel_h = attr.ints(0);
            param.kernel_w = attr.ints(1);
        }
        if (name == "global_pooling")
        {
            param.global = attr.b();
        }
        if (name == "adaptive")
        {
            adaptive_pool = attr.b();
        }
    }

    // if adaptive pool, ksize equal to output size
    if (adaptive_pool)
    {
        if (param.kernel_h == 1 && param.kernel_w == 1)
            param.global = 1;
        
        // stride = floor(in_size/out_size)
        // ksize = in_size - (out_size - 1) * stride
        else
        {
            StaticTensor* input = GetNodeInputTensor(graph, node, 0);
            std::vector<int> dims = input->dims;
            int in_h = dims[2];
            int in_w = dims[3];
            param.kernel_h = in_h - (param.kernel_h - 1) * param.stride_h;
            param.kernel_w = in_w - (param.kernel_w - 1) * param.stride_w;
        }
    }
    

    StaticOp* op = CreateStaticOp(graph, "Pooling");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleFlatten(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam("Flatten"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "start_axis")
        {
            param.axis = attr.i();
        }
        if (name == "stop_axis")
        {
            param.end_axis = attr.i();
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Flatten");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleDropout(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    StaticOp* op = CreateStaticOp(graph, "Dropout");
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleFullyconnected(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    FCParam param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));

    StaticTensor* weight = GetNodeInputTensor(graph, node, 1);
    if (weight->dims.size() == 2)
    {
        // transpose
        float* data = (float*)GetConstTensorBuffer(weight);
        int rows = weight->dims[0];
        int cols = weight->dims[1];
        float* new_data = (float*)std::malloc(rows * cols * sizeof(float));
        for (int col = 0; col < cols; col++)
        {
            for (int row = 0; row < rows; row++)
            {
                new_data[col * rows + row] = data[row * cols + col];
            }
        }
        free(data);
        SetConstTensorBuffer(weight, new_data);
        weight->dims[0] = cols;
        weight->dims[1] = rows;
    }
    
    param.num_output = weight->dims[1];

    StaticOp* op = CreateStaticOp(graph, "FullyConnected");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleScale(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    ScaleParam param = any_cast<ScaleParam>(OpManager::GetOpDefParam("Scale"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    float bias = 0.f, scale = 1.f;
    bool bias_after_scale = true;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "bias")
        {
            bias = attr.f();
        }
        if (name == "scale")
        {
            scale = attr.f();
        }
        if (name == "bias_after_scale")
        {
            bias_after_scale = attr.b();
        }
    }

    StaticTensor* input = GetNodeInputTensor(graph, node, 0);
    std::vector<int> dims = input->dims;
    int elem_num = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        elem_num *= dims[i];
    }

    // create and add gamma tensor
    // create tensor
    StaticTensor* gamma = CreateStaticConstTensor(graph, input->name + "_gamma");
    SetTensorDim(gamma, dims);
    SetTensorDataType(gamma, DataType::GetTypeID("float32"));
    SetTensorSize(gamma, elem_num * sizeof(float));
    uint8_t* mem_buf = (uint8_t*)std::malloc(elem_num * sizeof(float));
    float* data = (float*) mem_buf;
    for (int i = 0; i < elem_num; i++)
        data[i] = bias_after_scale ? scale : scale + bias;
    SetConstTensorBuffer(gamma, mem_buf);
    SetConstTensorFileLocation(gamma, -1, 0);

    // create node
    StaticOp* op_gamma = CreateStaticOp(graph, "Const");
    StaticNode* node_gamma = CreateStaticNode(graph, GetTensorName(gamma));
    SetNodeOp(node_gamma, op_gamma);
    AddNodeOutputTensor(node_gamma, gamma);
    AddNodeInputTensor(node, gamma);

    // create and add beta tensor
    if (bias_after_scale)
    {
        // create tensor
        StaticTensor* beta = CreateStaticConstTensor(graph, input->name + "_beta");
        SetTensorDim(beta, dims);
        SetTensorDataType(beta, DataType::GetTypeID("float32"));
        SetTensorSize(beta, elem_num * sizeof(float));
        uint8_t* mem_buf = (uint8_t*)std::malloc(elem_num * sizeof(float));
        float* data = (float*) mem_buf;
        for (int i = 0; i < elem_num; i++)
            data[i] = bias;
        SetConstTensorBuffer(beta, mem_buf);
        SetConstTensorFileLocation(beta, -1, 0);

        // create node
        StaticOp* op_beta = CreateStaticOp(graph, "Const");
        StaticNode* node_beta = CreateStaticNode(graph, GetTensorName(beta));
        SetNodeOp(node_beta, op_beta);
        AddNodeOutputTensor(node_beta, beta);
        AddNodeInputTensor(node, beta);
    }
    

    StaticOp* op = CreateStaticOp(graph, "Scale");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleReLu(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));

    if (pp_node.op == "relu")
        param.negative_slope = 0.f;
    if (pp_node.op == "leaky_relu")
    {
        paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
        for (int i = 0; i < op_desc.attrs_size(); i++)
        {
            paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
            std::string name = attr.name();
            if (name == "alpha")
            {
                param.negative_slope = attr.f();
            }
        }
    }

    StaticOp* op = CreateStaticOp(graph, "ReLu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleReLu6(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    StaticOp* op = CreateStaticOp(graph, "ReLu6");
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleSoftmax(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "axis")
        {
            param.axis = attr.i();
        }
    }
    if (param.axis == -1)
    {
        StaticTensor* input = GetNodeInputTensor(graph, node, 0);
        int dim_num = input->dims.size();
        param.axis = dim_num - 1;
    }

    StaticOp* op = CreateStaticOp(graph, "Softmax");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleInterp(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    InterpParam param = any_cast<InterpParam>(OpManager::GetOpDefParam("Interp"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "scale")
        {
            param.height_scale = attr.floats(0);
            param.width_scale = attr.floats(1);
        }
    }
    if (pp_node.op == "nearest_interp_v2")
        param.resize_type = 1;

    StaticOp* op = CreateStaticOp(graph, "Interp");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleResize(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    ResizeParam param = any_cast<ResizeParam>(OpManager::GetOpDefParam("Resize"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "scale")
        {
            param.scale_h = attr.floats(0);
            param.scale_w = attr.floats(1);
        }
    }
    if (pp_node.op == "nearest_interp_v2")
        param.type = 0;

    StaticOp* op = CreateStaticOp(graph, "Resize");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadPaddleConcat(StaticGraph* graph, StaticNode* node, const PaddleNode& pp_node)
{
    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));
    paddle::framework::proto::OpDesc op_desc = pp_node.op_desc;
    for (int i = 0; i < op_desc.attrs_size(); i++)
    {
        paddle::framework::proto::OpDesc::Attr attr = op_desc.attrs(i);
        std::string name = attr.name();
        if (name == "axis")
        {
            param.axis = attr.i();
        }
    }
    if (param.axis == -1)
    {
        StaticTensor* input = GetNodeInputTensor(graph, node, 0);
        int dim_num = input->dims.size();
        param.axis = dim_num - 1;
    }

    StaticOp* op = CreateStaticOp(graph, "Concat");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool PaddleSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if (!SerializerManager::SafeGet("paddle", serializer))
        return false;

    PaddleSerializer* p_paddle = dynamic_cast<PaddleSerializer*>(serializer.get());

    p_paddle->RegisterOpLoadMethod("conv2d", op_load_t(LoadPaddleConvolution));
    p_paddle->RegisterOpLoadMethod("depthwise_conv2d", op_load_t(LoadPaddleConvolution));
    p_paddle->RegisterOpLoadMethod("batch_norm", op_load_t(LoadPaddleBatchNorm));
    p_paddle->RegisterOpLoadMethod("elementwise_add", op_load_t(LoadOpPaddleElemwiseAdd));
    p_paddle->RegisterOpLoadMethod("pool2d", op_load_t(LoadPaddlePooling));
    p_paddle->RegisterOpLoadMethod("flatten_contiguous_range", op_load_t(LoadPaddleFlatten));
    p_paddle->RegisterOpLoadMethod("dropout", op_load_t(LoadPaddleDropout));
    p_paddle->RegisterOpLoadMethod("matmul", op_load_t(LoadPaddleFullyconnected));
    p_paddle->RegisterOpLoadMethod("scale", op_load_t(LoadPaddleScale));
    p_paddle->RegisterOpLoadMethod("relu", op_load_t(LoadPaddleReLu));
    p_paddle->RegisterOpLoadMethod("leaky_relu", op_load_t(LoadPaddleReLu));
    p_paddle->RegisterOpLoadMethod("relu6", op_load_t(LoadPaddleReLu6));
    p_paddle->RegisterOpLoadMethod("softmax", op_load_t(LoadPaddleSoftmax));
    p_paddle->RegisterOpLoadMethod("nearest_interp_v2", op_load_t(LoadPaddleResize));
    p_paddle->RegisterOpLoadMethod("concat", op_load_t(LoadPaddleConcat));

    return true;
}
}