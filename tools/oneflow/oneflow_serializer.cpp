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
 * Copyright (c) 2020, OneFlow Inc.
 * Author: daquexian566@gmail.com
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

#include "oneflow_serializer.hpp"

namespace TEngine {

std::map<std::string, OneFlowOpConverter*> method_map;
OneFlowOpConverter::OneFlowOpConverter(std::string op_name)
{
    method_map[op_name] = this;
}

#define XSTR(token) #token
#define STR(token) XSTR(token)

#define DEFINE_ONEFLOW_CONVERTER(arg_op_name) DEFINE_ONEFLOW_CONVERTER_WITH_INPUT_NAME_LIST(arg_op_name, InputNameList())

#define DEFINE_ONEFLOW_CONVERTER_WITH_INPUT_NAME_LIST(arg_op_name, arg_input_name_list) \
    DEFINE_ONEFLOW_CONVERTER_FULL(arg_op_name, arg_input_name_list, false)

#define DEFINE_ONEFLOW_CUSTOM_CONVERTER(arg_op_name)                \
    DEFINE_ONEFLOW_CONVERTER_FULL(arg_op_name, InputNameList(), true)

#define DEFINE_ONEFLOW_CONVERTER_FULL(arg_op_name, arg_input_name_list, arg_custom)                \
    class arg_op_name##Converter : public OneFlowOpConverter                                                               \
    {                                                                                                             \
    public:                                                                                                       \
        arg_op_name##Converter(std::string op_name) : OneFlowOpConverter(op_name) {}                                       \
        InputNameList input_name_list() const override;                                                                     \
        bool custom() const override;                                                                             \
        bool convert(StaticGraph* graph, StaticNode* node, const std::string& checkpoint_dir,                     \
                     const oneflow::OperatorConf& oneflow_node) const override;                                   \
    };                                                                                                            \
    static arg_op_name##Converter arg_op_name##_converter(STR(arg_op_name));                                      \
    bool arg_op_name##Converter::custom() const                                                                   \
    {                                                                                                             \
        return arg_custom;                                                                                        \
    }                                                                                                             \
    InputNameList arg_op_name##Converter::input_name_list() const                                                           \
    {                                                                                                             \
        return arg_input_name_list;                                                                                     \
    }                                                                                                             \
    bool arg_op_name##Converter::convert(StaticGraph* graph, StaticNode* node, const std::string& checkpoint_dir, \
                                         const oneflow::OperatorConf& oneflow_node) const

bool OneFlowSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if (file_list.size() != GetFileNum())
        return false;

    oneflow::SavedModel model;

    const auto prototxt_path = file_list[0];
    const auto weight_path = file_list[1];

    if (!LoadModelFile(prototxt_path.c_str(), model))
        return false;

    SetGraphSource(graph, prototxt_path);
    SetGraphSourceFormat(graph, "onnx");
    SetGraphConstTensorFile(graph, file_list[0]);
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_ONEFLOW);

    const auto& flow_graph = model.graphs().at(model.default_graph_name());
    return LoadGraph(flow_graph, weight_path, graph);
}

bool OneFlowSerializer::LoadModelFile(const char* fname, oneflow::SavedModel& model)
{
    std::ifstream is(fname, std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        LOG_ERROR() << "cannot open file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    bool ret = google::protobuf::TextFormat::Parse(&input_stream, &model);

    is.close();

    if (!ret)
    {
        LOG_ERROR() << "onnx serializer: parse file: " << fname << " failed\n";
        set_tengine_errno(EINVAL);
        return false;
    }

    return ret;
}

void OneFlowSerializer::CreateInputNode(StaticGraph* graph, const oneflow::GraphDef& oneflow_graph)
{
    const auto& default_signature = oneflow_graph.signatures().at(oneflow_graph.default_signature_name());
    int input_number = default_signature.inputs_size();
    for (const auto& input : default_signature.inputs())
    {
        const auto& input_name = input.second.lbi().op_name() + "/" + input.second.lbi().blob_name();
        std::vector<int> dims;
        for (const auto& dim : input.second.blob_conf().shape().dim())
        {
            dims.push_back(dim);
        }
        StaticTensor* tensor = CreateStaticTensor(graph, input_name);
        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        SetTensorDim(tensor, dims);

        StaticNode* node = CreateStaticNode(graph, input_name);
        StaticOp* op = CreateStaticOp(graph, "InputOp");

        SetNodeOp(node, op);
        AddNodeOutputTensor(node, tensor);

        /*add this node into graph input node list */
        AddGraphInputNode(graph, node);
    }
}

std::string GetOpTypeName(const oneflow::OperatorConf& op_conf)
{
    if (op_conf.has_user_conf())
    {
        return op_conf.user_conf().op_type_name();
    }
    if (op_conf.has_variable_conf())
    {
        return "variable";
    }
    throw std::invalid_argument("Unknown op type, op_conf is " + op_conf.DebugString());
}

bool OneFlowSerializer::LoadNode(StaticGraph* graph, StaticNode** node, const oneflow::OperatorConf& flow_node)
{
    auto* converter = GetConverterForOpConf(flow_node);
    if (converter->custom())
    {
        return true;
    }
    *node = CreateStaticNode(graph, flow_node.name());
    if (!flow_node.has_user_conf())
    {
        return true;
    }
    const auto op_type_name = GetOpTypeName(flow_node);
    auto input_list = converter->input_name_list();
    if (flow_node.user_conf().input_size() > 1 && input_list.empty())
    {
        throw std::runtime_error("Please define linput_list for " + op_type_name);
    }

    const auto& inputs = flow_node.user_conf().input();
    if (inputs.size() > 1)
    {
        for (const auto& input_name : input_list)
        {
            if (inputs.find(input_name) == inputs.end())
            {
                break;
            }
            const auto& input = inputs.at(input_name);
            for (const auto s : input.s())
            {
                StaticTensor* tensor = FindTensor(graph, s);
                AddNodeInputTensor(*node, tensor);
            }
        }
    }
    else
    {
        // TODO:
        const auto& input = inputs.begin();
        for (const auto s : input->second.s())
        {
            StaticTensor* tensor = FindTensor(graph, s);
            AddNodeInputTensor(*node, tensor);
        }
    }

    for (const auto& output : flow_node.user_conf().output())
    {
        assert(output.second.s_size() == 1);
        const std::string& output_name = output.second.s(0);
        const std::string& tensor_name = output_name;

        StaticTensor* tensor = CreateStaticTensor(graph, tensor_name);
        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        AddNodeOutputTensor(*node, tensor);
    }
    return true;
}

OneFlowOpConverter* OneFlowSerializer::GetConverterForOpConf(const oneflow::OperatorConf& op_conf)
{
    const auto& op_type_name = GetOpTypeName(op_conf);
    const auto maybe_converter = GetOpLoadMethod(op_type_name);
    if (maybe_converter.empty())
    {
        throw std::runtime_error("Unknown op type: " + op_type_name);
    }
    auto* converter = any_cast<OneFlowOpConverter*>(maybe_converter);
    return converter;
}

bool OneFlowSerializer::LoadGraph(const oneflow::GraphDef& model, const std::string& checkpoint_dir, StaticGraph* graph)
{
    SetGraphIdentity(graph, "oneflow", "daquexian", "0");

    CreateInputNode(graph, model);

    for (const auto& op : model.op_list())
    {
        if (op.has_input_conf() || op.has_return_conf())
        {
            continue;
        }
        StaticNode* node;
        if (!LoadNode(graph, &node, op))
        {
            break;
        }

        auto* converter = GetConverterForOpConf(op);
        if (!converter->convert(graph, node, checkpoint_dir, op))
        {
            break;
        }
    }
    return true;
}

template <typename T> T GetAttr(const oneflow::OperatorConf& op_conf, const std::string& attr_name);

#define DEFINE_GET_ATTR(type, flow_suffix)                                                                       \
    template <> type GetAttr<type>(const oneflow::OperatorConf& op_conf, const std::string& attr_name)           \
    {                                                                                                            \
        for (const auto& attr_pair : op_conf.user_conf().attr())                                                 \
        {                                                                                                        \
            if (attr_name == attr_pair.first)                                                                    \
            {                                                                                                    \
                return op_conf.user_conf().attr().at(attr_name).at_##flow_suffix();                              \
            }                                                                                                    \
        }                                                                                                        \
        throw std::invalid_argument("unknown attr_name " + attr_name + " for op_conf " + op_conf.DebugString()); \
    }

DEFINE_GET_ATTR(int32_t, int32)
DEFINE_GET_ATTR(int64_t, int64)
DEFINE_GET_ATTR(float, float)
DEFINE_GET_ATTR(double, double)
DEFINE_GET_ATTR(bool, bool)
DEFINE_GET_ATTR(std::string, string)

#define DEFINE_GET_ATTR_LIST(type, flow_suffix)                                                                      \
    template <>                                                                                                      \
    std::vector<type> GetAttr<std::vector<type>>(const oneflow::OperatorConf& op_conf, const std::string& attr_name) \
    {                                                                                                                \
        for (const auto& attr_pair : op_conf.user_conf().attr())                                                     \
        {                                                                                                            \
            if (attr_name == attr_pair.first)                                                                        \
            {                                                                                                        \
                std::vector<type> res;                                                                               \
                for (const auto& val : op_conf.user_conf().attr().at(attr_name).at_list_##flow_suffix().val())       \
                {                                                                                                    \
                    res.push_back(val);                                                                              \
                }                                                                                                    \
                return res;                                                                                          \
            }                                                                                                        \
        }                                                                                                            \
        throw std::invalid_argument("unknown attr_name " + attr_name + " for op_conf " + op_conf.DebugString());     \
    }

DEFINE_GET_ATTR_LIST(int32_t, int32)
DEFINE_GET_ATTR_LIST(int64_t, int64)
DEFINE_GET_ATTR_LIST(float, float)

#undef DEFINE_GET_ATTR_LIST
#undef DEFINE_GET_ATTR

std::vector<int64_t> GetShapeAttr(const oneflow::OperatorConf& op_conf, const std::string& attr_name)
{
    for (const auto& attr_pair : op_conf.user_conf().attr())
    {
        if (attr_name == attr_pair.first)
        {
            std::vector<int64_t> res;
            for (const auto& val : op_conf.user_conf().attr().at(attr_name).at_shape().dim())
            {
                res.push_back(val);
            }
            return res;
        }
    }
    throw std::invalid_argument("unknown attr_name " + attr_name + " for op_conf " + op_conf.DebugString());
}

/* functions to load indiviual operator */

DEFINE_ONEFLOW_CONVERTER(variable)
{
    assert(oneflow_node.has_variable_conf());
    size_t size = 1;
    std::vector<int> dims;
    for (const auto& dim : oneflow_node.variable_conf().shape().dim())
    {
        dims.push_back(dim);
        size *= dim;
    }

    const auto node_name = oneflow_node.name();
    StaticTensor* tensor = CreateStaticConstTensor(graph, node_name + "/out");
    SetTensorDim(tensor, dims);

    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    const size_t mem_size = sizeof(float) * size;
    SetTensorSize(tensor, mem_size);

    char* mem_buf = static_cast<char*>(std::malloc(mem_size));
    {
        const std::string file_path = checkpoint_dir + "/" + node_name + "/out";
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        std::streamsize file_len = file.tellg();
        assert(file_len == mem_size);
        file.seekg(0, std::ios::beg);
        file.read(mem_buf, file_len);
    }

    SetConstTensorBuffer(tensor, mem_buf);
    SetConstTensorFileLocation(tensor, -1, 0);
    StaticOp* op = CreateStaticOp(graph, "Const");
    SetNodeOp(node, op);

    AddNodeOutputTensor(node, tensor);
    return true;
}

DEFINE_ONEFLOW_CONVERTER(relu)
{
    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
    param.negative_slope = 0.f;

    StaticOp* op = CreateStaticOp(graph, "ReLu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER(sigmoid)
{
    StaticOp* op = CreateStaticOp(graph, "Sigmoid");
    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER_WITH_INPUT_NAME_LIST(broadcast_add, InputNameList({"x" ,"y"}))
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    param.type = ELT_SUM;
    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

DEFINE_ONEFLOW_CONVERTER(add_n)
{
    assert(oneflow_node.user_conf().input().begin()->second.s_size() == 2);

    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    param.type = ELT_SUM;
    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool ConvertPooling(StaticGraph* graph, StaticNode* node, const oneflow::OperatorConf& oneflow_node)
{
    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));

    const std::string& op_type = GetOpTypeName(oneflow_node);

    assert(op_type == "max_pool_2d" || op_type == "avg_pool_2d");
    param.global = 0;
    param.alg = op_type == "avg_pool_2d" ? kPoolAvg : kPoolMax;

    param.kernel_h = GetAttr<std::vector<int32_t>>(oneflow_node, "pool_size")[0];
    param.kernel_w = GetAttr<std::vector<int32_t>>(oneflow_node, "pool_size")[1];
    param.stride_h = GetAttr<std::vector<int32_t>>(oneflow_node, "strides")[0];
    param.stride_w = GetAttr<std::vector<int32_t>>(oneflow_node, "strides")[1];
    const auto padding_str = GetAttr<std::string>(oneflow_node, "padding");
    if (padding_str == "customized")
    {
        const auto padding_before = GetAttr<std::vector<int32_t>>(oneflow_node, "padding_before");
        const auto padding_after = GetAttr<std::vector<int32_t>>(oneflow_node, "padding_after");
        param.pad_h0 = padding_before[0];
        param.pad_w0 = padding_before[1];
        param.pad_h1 = padding_after[0];
        param.pad_w1 = padding_after[1];
    }
    else if (padding_str == "valid")
    {
        param.pad_h0 = 0;
        param.pad_w0 = 0;
        param.pad_h1 = 0;
        param.pad_w1 = 0;
    }
    else
    {
        throw std::invalid_argument("Unsupported padding str " + padding_str);
    }
    // TODO: ceil mode

    StaticOp* op = CreateStaticOp(graph, "Pooling");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER(max_pool_2d)
{
    return ConvertPooling(graph, node, oneflow_node);
}

DEFINE_ONEFLOW_CONVERTER(avg_pool_2d)
{
    return ConvertPooling(graph, node, oneflow_node);
}

DEFINE_ONEFLOW_CONVERTER_WITH_INPUT_NAME_LIST(conv2d, (InputNameList{"in", "weight", "bias"}))
{
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));
    param.kernel_h = GetAttr<std::vector<int32_t>>(oneflow_node, "kernel_size")[0];
    param.kernel_w = GetAttr<std::vector<int32_t>>(oneflow_node, "kernel_size")[1];
    param.stride_h = GetAttr<std::vector<int32_t>>(oneflow_node, "strides")[0];
    param.stride_w = GetAttr<std::vector<int32_t>>(oneflow_node, "strides")[1];
    const auto padding_before = GetAttr<std::vector<int32_t>>(oneflow_node, "padding_before");
    param.pad_h0 = padding_before[0];
    param.pad_w0 = padding_before[1];
    param.pad_h1 = padding_before[0];
    param.pad_w1 = padding_before[1];
    const auto dilations = GetAttr<std::vector<int32_t>>(oneflow_node, "dilation_rate");
    param.dilation_h = dilations[0];
    param.dilation_w = dilations[1];
    param.group = GetAttr<int32_t>(oneflow_node, "groups");
    param.output_channel = GetAttr<int32_t>(oneflow_node, "filters");

    StaticOp* op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER_WITH_INPUT_NAME_LIST(normalization, (InputNameList{"x", "gamma", "beta", "moving_mean",
                                                                  "moving_variance", "_add_to_output"}))
{
    assert(oneflow_node.user_conf().input_size() < 6);
    StaticOp* op = CreateStaticOp(graph, "BatchNormalization");

    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));
    param.eps = GetAttr<float>(oneflow_node, "epsilon");
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CUSTOM_CONVERTER(bias_add)
{
    const auto& inputs = oneflow_node.user_conf().input();
    assert(inputs.at("b").s_size() == 1);
    StaticTensor* unsqueezed_bias_tensor;
    {
        const auto bias_tensor_name = inputs.at("b").s(0);
        StaticTensor* bias_tensor = FindTensor(graph, bias_tensor_name);
        StaticNode* unsqueeze_node = CreateStaticNode(graph, oneflow_node.name() + "_unsqueeze_tengine");
        AddNodeInputTensor(unsqueeze_node, bias_tensor);
        const std::string& unsqueeze_output_name = oneflow_node.name() + "_unsqueeze_tengine_out";

        unsqueezed_bias_tensor = CreateStaticTensor(graph, unsqueeze_output_name);
        SetTensorDataType(unsqueezed_bias_tensor, DataType::GetTypeID("float32"));
        AddNodeOutputTensor(unsqueeze_node, unsqueezed_bias_tensor);

        UnsqueezeParam param = any_cast<UnsqueezeParam>(OpManager::GetOpDefParam("Unsqueeze"));
        param.axises.push_back(0);
        param.axises.push_back(2);
        param.axises.push_back(3);
        StaticOp* op = CreateStaticOp(graph, "Unsqueeze");
        SetOperatorParam(op, param);
        SetNodeOp(unsqueeze_node, op);
    }

    StaticNode* add_node = CreateStaticNode(graph, oneflow_node.name());
    AddNodeInputTensor(add_node, FindTensor(graph, inputs.at("a").s(0)));
    AddNodeInputTensor(add_node, unsqueezed_bias_tensor);
    const auto& output_name = oneflow_node.user_conf().output().begin()->second.s(0);
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(add_node, tensor);

    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    param.type = ELT_SUM;
    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(add_node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER(clip_by_scalar)
{
    // TODO: integer_max/min
    StaticOp* op = CreateStaticOp(graph, "Clip");

    ClipParam param = any_cast<ClipParam>(OpManager::GetOpDefParam("Clip"));
    param.max = GetAttr<double>(oneflow_node, "floating_max");
    param.min = GetAttr<double>(oneflow_node, "floating_min");
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER(flatten)
{
    StaticOp* op = CreateStaticOp(graph, "Flatten");

    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam("Flatten"));
    param.axis = 1;
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER(reshape)
{
    StaticOp* op = CreateStaticOp(graph, "Reshape");

    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));
    for (const auto& dim : GetShapeAttr(oneflow_node, "shape"))
    {
        param.re_shape.push_back(dim);
    }
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

DEFINE_ONEFLOW_CONVERTER_WITH_INPUT_NAME_LIST(matmul, (InputNameList{"a", "b"}))
{
    const auto& op_conf = oneflow_node.user_conf();
    assert(op_conf.input_size() == 2);
    assert(!op_conf.attr().at("transpose_a").at_bool());
    assert(op_conf.attr().at("transpose_b").at_bool());

    StaticOp* op = CreateStaticOp(graph, "FullyConnected");
    FCParam fc_param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));
    StaticTensor* weight_tensor = FindTensor(graph, op_conf.input().at("b").s(0));
    fc_param.num_output = weight_tensor->dims[0];

    SetOperatorParam(op, fc_param);
    SetNodeOp(node, op);

    return true;
}

// To register all op loader...
bool OneFlowSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if (!SerializerManager::SafeGet("oneflow", serializer))
    {
        return false;
    }

    OneFlowSerializer* flow_serializer = dynamic_cast<OneFlowSerializer*>(serializer.get());

    for (const auto& x : method_map)
    {
        flow_serializer->RegisterOpLoadMethod(x.first, x.second);
    }

    return true;
}

}    // namespace TEngine
