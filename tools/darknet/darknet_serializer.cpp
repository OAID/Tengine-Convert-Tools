#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "tengine_c_api.h"
#include "exec_attr.hpp"
#include "darknet_serializer.hpp"
//#include "darknet/te_darknet.hpp"
#include "logger.hpp"
#include "data_type.hpp"

#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/reshape_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/upsample_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/reorg_param.hpp"
#include "operator/region_param.hpp"
#include "operator/slice_param.hpp"

namespace TEngine {

using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map,
                                     list* options, int index, FILE* fp)>;
bool DarkNetSerializer::ConstructGraph(StaticGraph* graph, const char* weight_file, list* sections)
{
    FILE* fp = fopen(weight_file, "rb");
    int major;
    int minor;
    int revision;
    int seen;
    std::vector<std::string> tensor_name_map;
    if (0 == fread(&major, sizeof(int), 1, fp))
    {
        printf("read major failed\n");
        return false;
    }
    if (0 == fread(&minor, sizeof(int), 1, fp))
    {
        printf("read minor failed\n");
        return false;
    }
    if (0 == fread(&revision, sizeof(int), 1, fp))
    {
        printf("read revision failed\n");
        return false;
    }
    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000)
    {
        double iseen = 0;
        if (0 == fread(&iseen, sizeof(double), 1, fp))
        {
            printf("read iseen failed\n");
            return false;
        }
        seen = ( int )iseen;
    }
    else
    {
        if (0 == fread(&seen, sizeof(int), 1, fp))
        {
            printf("read seen failed\n");
            return false;
        }
    }
    int transpose = (major > 1000) || (minor > 1000);

    printf("major: %d, minor: %d, revision: %d, seen: %d, transpose: %d \n", major, minor, revision, seen, transpose);

    node* n = sections->front;
    section* s = ( section* )n->val;
    list* options = s->options;
    // Creat the input node
    StaticNode* input_node = CreateStaticNode(graph, "input");
    StaticTensor* input_tensor = CreateStaticTensor(graph, "input_0");
    SetTensorDataType(input_tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(input_node, input_tensor);
    std::vector<int> dim;

    int input_h = option_find_int_quiet(options, ( char* )"height", 0);
    int input_w = option_find_int_quiet(options, ( char* )"width", 0);
    int input_c = option_find_int_quiet(options, ( char* )"channels", 0);
    int batch_num = option_find_int(options, ( char* )"batch", 1);

    dim.push_back(batch_num);
    dim.push_back(input_c);
    dim.push_back(input_h);
    dim.push_back(input_w);
    SetTensorDim(input_tensor, dim);
    tensor_name_map.push_back("input_0");

    StaticOp* op = CreateStaticOp(graph, "InputOp");
    SetNodeOp(input_node, op);
    AddGraphInputNode(graph, input_node);
    free_section(s);
    n = n->next;
    int count = 1;
    while (n)
    {
        s = ( section* )n->val;
        options = s->options;
        std::string node_name = s->type + std::to_string(count);
        std::string a = "[";
        std::string b = "]";
        std::string c = "_";

        remove_str(node_name, a);
        replace_str(node_name, b, c);
        StaticNode* node = CreateStaticNode(graph, node_name);
        std::string tensor_name = node_name + "_0";
        StaticTensor* tensor = CreateStaticTensor(graph, tensor_name);
        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        AddNodeOutputTensor(node, tensor);
        tensor_name_map.push_back(tensor_name);

        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(s->type));
        if (!op_func(graph, node, tensor_name_map, options, count, fp))
            break;
        free_section(s);
        count++;
        n = n->next;
    }
    fclose(fp);

    return true;
}

bool DarkNetSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if (file_list.size() != GetFileNum())
        return false;

    const char* cfg_file = file_list[0].c_str();
    list* sections = read_cfg(cfg_file);
    // Read the weights get
    const char* weight_file = file_list[1].c_str();
    if (nullptr == weight_file)
        return false;
    // Construct the Graph
    ConstructGraph(graph, weight_file, sections);

    free_list(sections);
    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "darknet");
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_DARKNET);

    return true;
}
static bool LoadConvBlob(StaticGraph* graph, StaticNode* node, std::vector<int>& weight_dims, int batch_norm, FILE* fp)
{
    if (fp == NULL)
        return false;
    // Add the weight tensor
    std::string weight_tensor_name = GetNodeName(node) + "_1";
    StaticTensor* weight_tensor = CreateStaticConstTensor(graph, weight_tensor_name);
    SetTensorDim(weight_tensor, weight_dims);
    SetTensorDataType(weight_tensor, DataType::GetTypeID("float32"));
    StaticNode* weight_node = CreateStaticNode(graph, weight_tensor_name);
    StaticOp* weight_const_op = CreateStaticOp(graph, "Const");
    SetNodeOp(weight_node, weight_const_op);
    AddNodeOutputTensor(weight_node, weight_tensor);
    AddNodeInputTensor(node, weight_tensor);
    // Add the bias tensor
    std::vector<int> bias_dims;
    bias_dims.push_back(1);
    bias_dims.push_back(1);
    bias_dims.push_back(1);
    bias_dims.push_back(weight_dims[0]);
    std::string bias_tensor_name = GetNodeName(node) + "_2";
    StaticTensor* bias_tensor = CreateStaticConstTensor(graph, bias_tensor_name);
    SetTensorDim(bias_tensor, bias_dims);
    SetTensorDataType(bias_tensor, DataType::GetTypeID("float32"));
    StaticNode* bias_node = CreateStaticNode(graph, bias_tensor_name);
    StaticOp* bias_const_op = CreateStaticOp(graph, "Const");
    SetNodeOp(bias_node, bias_const_op);
    AddNodeOutputTensor(bias_node, bias_tensor);
    AddNodeInputTensor(node, bias_tensor);

    int out_channel = weight_dims[0];
    float* bias_data = ( float* )malloc(out_channel * sizeof(float));
    if (0 == fread(bias_data, sizeof(float), out_channel, fp))
        printf("Read bias data failed\n");
    float* scales = NULL;
    float* means = NULL;
    float* variances = NULL;
    if (batch_norm)
    {
        scales = ( float* )malloc(sizeof(float) * out_channel);
        means = ( float* )malloc(sizeof(float) * out_channel);
        variances = ( float* )malloc(sizeof(float) * out_channel);
        if (0 == fread(scales, sizeof(float), out_channel, fp))
            printf("Read scales failed\n");
        if (0 == fread(means, sizeof(float), out_channel, fp))
            printf("Read means failed\n");
        if (0 == fread(variances, sizeof(float), out_channel, fp))
            printf("Read variances failed\n");
    }
    int weight_size = weight_dims[0] * weight_dims[1] * weight_dims[2] * weight_dims[3];
    float* weight_data = ( float* )malloc(weight_size * sizeof(float) + 128);
    if (0 == fread(weight_data, sizeof(float), weight_size, fp))
        printf("Read weight data failed\n");

    // fuse the batchnorm
    if (batch_norm)
    {
        int kernel_size = weight_dims[1] * weight_dims[2] * weight_dims[3];
        for (int i = 0; i < out_channel; ++i)
        {
            float scale = scales[i] / sqrt(variances[i] + .00001);
            for (int j = 0; j < kernel_size; ++j)
            {
                weight_data[i * kernel_size + j] *= scale;
            }
            bias_data[i] -= means[i] * scale;
        }
    }
    SetTensorSize(weight_tensor, weight_size * sizeof(float));
    SetTensorSize(bias_tensor, out_channel * sizeof(float));
    SetConstTensorBuffer(weight_tensor, ( void* )weight_data);
    SetConstTensorFileLocation(weight_tensor, -1, 0);
    SetConstTensorBuffer(bias_tensor, ( void* )bias_data);
    SetConstTensorFileLocation(bias_tensor, -1, 0);

    return true;
}

static bool LoadConv2D(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map, list* options,
                       int index, FILE* fp)
{
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[index - 1]);
    if (tensor == NULL)
        return false;
    AddNodeInputTensor(node, tensor);
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));
    int n = option_find_int(options, ( char* )"filters", 1);
    int size = option_find_int(options, ( char* )"size", 1);
    int stride = option_find_int(options, ( char* )"stride", 1);
    int pad = option_find_int_quiet(options, ( char* )"pad", 0);
    int padding = option_find_int_quiet(options, ( char* )"padding", 0);
    int groups = option_find_int_quiet(options, ( char* )"groups", 1);
    int batch_normalize = option_find_int_quiet(options, ( char* )"batch_normalize", 0);
    char* activation_s = option_find_str(options, ( char* )"activation", ( char* )"logistic");
    // ACTIVATION activation = get_activation(activation_s);
    if (pad)
        padding = size / 2;

    param.kernel_h = size;
    param.kernel_w = size;
    param.stride_h = stride;
    param.stride_w = stride;
    param.pad_h0 = padding;
    param.pad_h1 = padding;
    param.pad_w0 = padding;
    param.pad_w1 = padding;
    param.group = groups;
    param.output_channel = n;

    std::vector<int> input_dims = GetTensorDim(tensor);
    int batch = input_dims[0];
    int in_c = input_dims[1];
    int in_h = input_dims[2];
    int in_w = input_dims[3];
    // Create the weight tensor
    std::vector<int> weight_dims;
    weight_dims.push_back(n);
    weight_dims.push_back(in_c / groups);
    weight_dims.push_back(size);
    weight_dims.push_back(size);
    LoadConvBlob(graph, node, weight_dims, batch_normalize, fp);
    // Set the Ouput Tensor Dim
    std::vector<int> out_dims;
    int out_c = n;
    int out_h = (in_h + 2 * padding - size) / stride + 1;
    int out_w = (in_w + 2 * padding - size) / stride + 1;
    out_dims.push_back(batch);
    out_dims.push_back(out_c);
    out_dims.push_back(out_h);
    out_dims.push_back(out_w);

    StaticTensor* out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(out_tensor, out_dims);

    StaticOp* op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    if (strcmp(activation_s, "leaky") == 0)
    {
        std::string relu_name = "leaky_" + std::to_string(index);
        StaticNode* relu_node = CreateStaticNode(graph, relu_name);
        ReLuParam relu_param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
        relu_param.negative_slope = 0.1f;
        StaticOp* relu_op = CreateStaticOp(graph, "ReLu");
        SetOperatorParam(relu_op, relu_param);
        SetNodeOp(relu_node, relu_op);
        AddNodeInputTensor(relu_node, out_tensor);
        std::string relu_tensor_name = relu_name + "_0";
        StaticTensor* relu_out_tensor = CreateStaticTensor(graph, relu_tensor_name);
        SetTensorDataType(relu_out_tensor, DataType::GetTypeID("float32"));
        SetTensorDim(relu_out_tensor, out_dims);
        AddNodeOutputTensor(relu_node, relu_out_tensor);

        // update the tensor name map;
        tensor_name_map[index] = relu_tensor_name;
    }
    if (strcmp(activation_s, "mish") == 0)
    {
        std::string mish_name = "mish_" + std::to_string(index);
        StaticNode* mish_node = CreateStaticNode(graph, mish_name);
        StaticOp* mish_op = CreateStaticOp(graph, "Mish");
        SetNodeOp(mish_node, mish_op);
        AddNodeInputTensor(mish_node, out_tensor);
        std::string mish_tensor_name = mish_name + "_0";
        StaticTensor* mish_out_tensor = CreateStaticTensor(graph, mish_tensor_name);
        SetTensorDataType(mish_out_tensor, DataType::GetTypeID("float32"));
        SetTensorDim(mish_out_tensor, out_dims);
        AddNodeOutputTensor(mish_node, mish_out_tensor);

        // update the tensor name map;
        tensor_name_map[index] = mish_tensor_name;
    }
    return true;
}


static bool LoadShortCut(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map, list* options,
                         int index, FILE* fp)
{
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[index - 1]);
    std::vector<int> input_dims = GetTensorDim(tensor);
    if (tensor == NULL)
    {
        printf("tensor is null \n");
        return false;
    }
    AddNodeInputTensor(node, tensor);
    char* l = option_find(options, ( char* )"from");
    int from_index = atoi(l);
    if (from_index < 0)
        from_index = index + from_index;

    tensor = FindTensor(graph, tensor_name_map[from_index]);
    if (tensor == NULL)
        return false;
    AddNodeInputTensor(node, tensor);

    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    param.type = ELT_SUM;

    std::vector<int> out_dims;
    out_dims.push_back(input_dims[0]);
    out_dims.push_back(input_dims[1]);
    out_dims.push_back(input_dims[2]);
    out_dims.push_back(input_dims[3]);

    StaticTensor* out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(out_tensor, out_dims);

    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMaxPooling(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map,
                           list* options, int index, FILE* fp)
{
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[index - 1]);
    if (tensor == NULL)
    {
        printf("tensor is null \n");
        return false;
    }
    std::vector<int> input_dims = GetTensorDim(tensor);
    AddNodeInputTensor(node, tensor);
    int stride = option_find_int(options, ( char* )"stride", 1);
    int size = option_find_int(options, ( char* )"size", stride);
    int padding = option_find_int_quiet(options, ( char* )"padding", size - 1);

    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));
    param.kernel_h = size;
    param.kernel_w = size;

    param.pad_h0 = padding;
    param.pad_h1 = padding;
    param.pad_w0 = padding;
    param.pad_w1 = padding;
    param.stride_h = stride;
    param.stride_w = stride;
    param.caffe_flavor = 2;
    param.alg = kPoolMax;
    std::vector<int> out_dims;
    out_dims.push_back(input_dims[0]);
    out_dims.push_back(input_dims[1]);
    int in_h = input_dims[2];
    int in_w = input_dims[3];
    int out_h = (in_h + padding - size) / stride + 1;
    int out_w = (in_w + padding - size) / stride + 1;
    out_dims.push_back(out_h);
    out_dims.push_back(out_w);

    StaticTensor* out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(out_tensor, out_dims);

    StaticOp* op = CreateStaticOp(graph, "Pooling");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadYolo(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map, list* options,
                     int index, FILE* fp)
{
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[index - 1]);
    if (tensor == NULL)
        return false;
    AddNodeInputTensor(node, tensor);
    std::vector<int> input_dims = GetTensorDim(tensor);
    std::vector<int> out_dims;
    out_dims.push_back(input_dims[0]);
    out_dims.push_back(input_dims[1]);
    out_dims.push_back(input_dims[2]);
    out_dims.push_back(input_dims[3]);

    StaticTensor* out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(out_tensor, out_dims);
    StaticOp* op = CreateStaticOp(graph, "Dropout");
    SetNodeOp(node, op);

    return true;
}

static bool LoadRoute(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map, list* options,
                      int index, FILE* fp)
{
    //check layers option
    char* layers = option_find(options, ( char* )("layers"));
    int layers_len = strlen(layers);
    int n_layers = 1;
    std::vector<int> layers_arr;
    for (int i = 0; i < layers_len; ++i)
    {
        if (layers[i] == ',')
            ++n_layers;
    }
    for (int i = 0; i < n_layers; ++i)
    {
        int from_index = atoi(layers);
        layers = strchr(layers, ',') + 1;
        if (from_index < 0)
            from_index = index + from_index;
        else
            from_index = from_index + 1;
        layers_arr.push_back(from_index);
    }
    //check groups option
    char* groups = option_find(options, ( char* )("groups"));
    int groups_len = (groups != nullptr) ? strlen(groups) : 0;
    int n_groups = 0;
    std::vector<int> groups_arr;
    if (groups_len > 0)
    {
        n_groups = 1;
        for (int i = 0; i < groups_len; ++i)
        {
            if (groups[i] == ',')
                ++n_groups;
        }
        for (int i = 0; i < n_groups; ++i)
        {
            int group_count = atoi(groups);
            groups = strchr(groups, ',') + 1;
            groups_arr.push_back(group_count);
        }
    }

    //check group_id option
    char* group_id = option_find(options, ( char* )("group_id"));
    int group_id_len = (group_id != nullptr) ? strlen(group_id) : 0;
    int n_group_id = 0;
    std::vector<int> group_id_arr;
    if(group_id_len > 0)
    {
        n_group_id = 1;
        for (int i = 0; i < group_id_len; ++i)
        {
            if (group_id[i] == ',')
                ++n_group_id;
        }
        for (int i = 0; i < n_group_id; ++i)
        {
            int group_index = atoi(group_id);
            group_id = strchr(group_id, ',') + 1;
            group_id_arr.push_back(group_index);
        }
    }
    if (groups_arr.size() == 0)
    {
        for(int i = 0; i < layers_arr.size(); i++)
            groups_arr.push_back(1);
    }
    if (group_id_arr.size() == 0)
    {
        for(int i = 0; i < layers_arr.size(); i++)
            group_id_arr.push_back(0);
    }
    //debug print
    for (int i = 0; i < layers_arr.size(); i++)
    {
        printf("layers %d: %d\n", i, layers_arr[i]);
        printf("groups_arr %d: %d\n", i, groups_arr[i]);
        printf("group_id_arr %d: %d\n", i, group_id_arr[i]);
    }
    //split if need
    std::vector<bool> slice_flag;
    std::vector<StaticNode*> slice_node_arr;
    std::vector<StaticTensor*> slice_out_tensor_arr;
    for (int i = 0; i < layers_arr.size(); i++)
    {
        std::string slice_name = "route_slice_" + std::to_string(index) + std::to_string(i);
        int from_index = layers_arr[i];
        StaticTensor* input_tensor = FindTensor(graph, tensor_name_map[from_index]);
        std::vector<int> input_dims = GetTensorDim(input_tensor);
        if (groups_arr[i] == 1)
        {
            slice_flag.push_back(false);
            slice_node_arr.push_back((StaticNode*)nullptr);
        }
        else
        {
            slice_flag.push_back(true);
            std::vector<int> out_dims = input_dims;
            SliceParam param = any_cast<SliceParam>(OpManager::GetOpDefParam("Slice"));
            param.iscaffe = true;
            param.slice_point_.clear();
            int step = input_dims[1] / groups_arr[i];
            out_dims[1] = step;
            for (int j = 0; j < groups_arr[i] - 1; j++)
            {
                param.slice_point_.push_back(step*(j + 1));
            }
            StaticNode* slice_node = CreateStaticNode(graph, slice_name);
            StaticOp* slice_op = CreateStaticOp(graph, "Slice");
            SetOperatorParam(slice_op, param);
            SetNodeOp(slice_node, slice_op);
            AddNodeInputTensor(slice_node, input_tensor);
            for (int k = 0; k < groups_arr[i]; k++)
            {
                std::string slice_tensor_name = slice_name + "_" + std::to_string(k);
                StaticTensor* slice_out_tensor = CreateStaticTensor(graph, slice_tensor_name);
                SetTensorDataType(slice_out_tensor, DataType::GetTypeID("float32"));
                SetTensorDim(slice_out_tensor, out_dims);
                AddNodeOutputTensor(slice_node, slice_out_tensor);
            }
            slice_node_arr.push_back(slice_node);
            //debug print
            for(int it = 0; it < param.slice_point_.size(); it++)
                printf("param slice_point %d : %d\n", it, param.slice_point_[it]);
        }
    }
    //concat
    int output_c = 0;
    for (int i = 0; i < layers_arr.size(); i++)
    {
        if(slice_flag[i] == true)
        {
            StaticNode* slice_node = slice_node_arr[i];
            StaticTensor* slice_out_tensor = GetNodeOutputTensor(graph, slice_node, group_id_arr[i]);
            std::vector<int> slice_out_dims = GetTensorDim(slice_out_tensor);
            output_c += slice_out_dims[1];
            AddNodeInputTensor(node, slice_out_tensor);
        }
        else
        {
            int from_index = layers_arr[i];
            StaticTensor* tensor = FindTensor(graph, tensor_name_map[from_index]);
            std::vector<int> input_dims = GetTensorDim(tensor);
            output_c += input_dims[1];
            AddNodeInputTensor(node, tensor);
        }
    }
    std::vector<int> input_dims;
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[layers_arr[0]]);
    input_dims = GetTensorDim(tensor);
    std::string concat_name = "route_concat" + std::to_string(index);
    std::string concat_tensor_name = concat_name + "_0";
    std::vector<int> concat_out_dims;
    concat_out_dims.push_back(input_dims[0]);
    concat_out_dims.push_back(output_c);
    concat_out_dims.push_back(input_dims[2]);
    concat_out_dims.push_back(input_dims[3]);
    StaticTensor* concat_out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(concat_out_tensor, concat_out_dims);
    StaticOp* concat_op = CreateStaticOp(graph, "Concat");
    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));
    SetOperatorParam(concat_op, param);
    SetNodeOp(node, concat_op);

    return true;
}
static bool LoadUpsample(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map, list* options,
                         int index, FILE* fp)
{
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[index - 1]);
    AddNodeInputTensor(node, tensor);
    UpsampleParam param = any_cast<UpsampleParam>(OpManager::GetOpDefParam("Upsample"));
    int scale = option_find_int(options, ( char* )"stride", 2);
    param.scale = scale;

    std::vector<int> input_dims = GetTensorDim(tensor);
    std::vector<int> out_dims;
    out_dims.push_back(input_dims[0]);
    out_dims.push_back(input_dims[1]);
    out_dims.push_back(input_dims[2] * scale);
    out_dims.push_back(input_dims[3] * scale);

    StaticTensor* out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(out_tensor, out_dims);
    StaticOp* op = CreateStaticOp(graph, "Upsample");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadReorg(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map, list* options,
                      int index, FILE* fp)
{
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[index - 1]);
    AddNodeInputTensor(node, tensor);
    ReorgParam param = any_cast<ReorgParam>(OpManager::GetOpDefParam("Reorg"));
    int stride = option_find_int(options, ( char* )"stride", 1);
    param.stride = stride;

    std::vector<int> input_dims = GetTensorDim(tensor);
    std::vector<int> out_dims;
    out_dims.push_back(input_dims[0]);
    out_dims.push_back(input_dims[1] * stride * stride);
    out_dims.push_back(input_dims[2] / stride);
    out_dims.push_back(input_dims[3] / stride);

    StaticTensor* out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(out_tensor, out_dims);
    StaticOp* op = CreateStaticOp(graph, "Reorg");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadRegion(StaticGraph* graph, StaticNode* node, std::vector<std::string>& tensor_name_map, list* options,
                       int index, FILE* fp)
{
    StaticTensor* tensor = FindTensor(graph, tensor_name_map[index - 1]);
    AddNodeInputTensor(node, tensor);
    RegionParam param = any_cast<RegionParam>(OpManager::GetOpDefParam("Region"));

    int coords = option_find_int(options, ( char* )"coords", 4);
    int classes = option_find_int(options, ( char* )"classes", 20);
    int num = option_find_int(options, ( char* )"num", 1);
    char* a = option_find_str(options, ( char* )"anchors", 0);
    float thresh = option_find_float(options, ( char* )"thresh", .5);

    param.num_classes = classes;
    param.num_box = num;
    param.coords = coords;
    param.nms_threshold = thresh;

    // get the bias;
    if (a)
    {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i)
        {
            if (a[i] == ',')
                ++n;
        }
        for (i = 0; i < n; ++i)
        {
            float bias = atof(a);
            param.biases.push_back(bias);
            a = strchr(a, ',') + 1;
        }
    }
    std::vector<int> input_dims = GetTensorDim(tensor);
    std::vector<int> out_dims;
    out_dims.push_back(input_dims[0]);
    out_dims.push_back(input_dims[1]);
    out_dims.push_back(input_dims[2]);
    out_dims.push_back(input_dims[3]);

    StaticTensor* out_tensor = GetNodeOutputTensor(graph, node, 0);
    SetTensorDim(out_tensor, out_dims);
    StaticOp* op = CreateStaticOp(graph, "Region");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
bool DarkNetSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if (!SerializerManager::SafeGet("darknet", serializer))
        return false;

    DarkNetSerializer* darknet = dynamic_cast<DarkNetSerializer*>(serializer.get());
    darknet->RegisterOpLoadMethod("[convolutional]", op_load_t(LoadConv2D));
    darknet->RegisterOpLoadMethod("[shortcut]", op_load_t(LoadShortCut));
    darknet->RegisterOpLoadMethod("[yolo]", op_load_t(LoadYolo));
    darknet->RegisterOpLoadMethod("[route]", op_load_t(LoadRoute));
    darknet->RegisterOpLoadMethod("[upsample]", op_load_t(LoadUpsample));
    darknet->RegisterOpLoadMethod("[maxpool]", op_load_t(LoadMaxPooling));
    darknet->RegisterOpLoadMethod("[reorg]", op_load_t(LoadReorg));
    darknet->RegisterOpLoadMethod("[region]", op_load_t(LoadRegion));

    return true;
}

}    // namespace TEngine
