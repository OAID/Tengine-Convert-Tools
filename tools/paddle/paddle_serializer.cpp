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

namespace TEngine {

using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const PaddleNode& paddle_node)>;


bool PaddleSerializer::LoadBinaryFile(const char* fname)
{
    std::ifstream fin(fname, std::ios::in | std::ios::binary);
    // printf("%s \n", fname);
    std::istream *is = &fin;
    uint64_t lod_level;
    is->read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    // printf("top level data: %d \n", lod_level);
    for (uint64_t i = 0; i < 40; ++i) {
        uint64_t size;
        is->read(reinterpret_cast<char *>(&size), sizeof(size));
        // printf("sub data : %d \n", size);
        // lod[i] = tmp;
    }
    return true;
}

bool PaddleSerializer::LoadTextFile(const char* fname, paddle::framework::proto::ProgramDesc& model_net)
{
    std::ifstream is(fname, std::ios::in);

    if (!is.is_open())
    {
        LOG_ERROR() << "cannot open file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);
    coded_input.SetTotalBytesLimit(512 << 20);
    bool ret = model_net.ParseFromCodedStream(&coded_input);
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

    paddle::framework::proto::ProgramDesc model;

    if (!LoadTextFile(file_list[1].c_str(), model))
        return false;

    
    if (!LoadBinaryFile(file_list[0].c_str()))
        return false;


    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "paddle");
    SetGraphConstTensorFile(graph, file_list[0]);
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_PADDLE);

    return LoadGraph(model, graph);
}


bool PaddleSerializer::LoadGraph(paddle::framework::proto::ProgramDesc& paddle_net, StaticGraph* graph)
{
    SetGraphIdentity(graph, "paddle", "paddle-paddle", "0");
    // name_map_t tensor_name_map;
    int graph_num = paddle_net.blocks_size();
    for(int g = 0;  g < graph_num; g++){
        paddle::framework::proto::BlockDesc graph_net = paddle_net.blocks(g);
        int layer_num = graph_net.ops_size();
        printf("layer_num : %d \n", layer_num);
        int tensor_num = graph_net.vars_size();
        // Tensor Name
        #if 0
        for(int v = 0; v < tensor_num; v++){
            paddle::framework::proto::VarDesc paddle_tensor = graph_net.vars(v);
            std::string tensor_name = paddle_tensor.name();
            printf("%s \n", tensor_name.c_str());
        }
        #endif


        // Op Name
        #if 1
        for(int l = 0; l < layer_num; l++){
            paddle::framework::proto::OpDesc op_info = graph_net.ops(l);
            // paddle::framework::proto::OpProto op_info = op_desc.;
            // paddle::framework::proto::OpDesc_Attr op_attr = op_info.attrs(4);
            std::string op_name = op_info.type();

            // paddle::framework::proto::
            int input_size = op_info.inputs_size();
            int output_size = op_info.outputs_size();
            printf("op_name: %s ; inputs: %d ; outputs: %d ; \n",op_name.c_str(), input_size, output_size);
            for(int in = 0; in < input_size; in++){
                // std::string in_name = op_info.inputs(in);

                paddle::framework::proto::OpDesc_Var in_tensors = op_info.inputs(in);
                std::string in_name = in_tensors.parameter();
                google::protobuf::RepeatedPtrField<std::string > args_name = in_tensors.arguments();
                // std::string at_name = args_name.Get(0);
                int args_size = in_tensors.arguments_size();
                if(args_size){
                    std::string at_name = args_name.Get(0);
                    printf("%s %s \n", in_name.c_str(), at_name.c_str());
                }
                // printf("args size: %d \n", args_size);
                // printf("%s %s \n", in_name.c_str(), at_name.c_str());
            }
            for(int in = 0; in < output_size; in++){
                // std::string in_name = op_info.inputs(in);

                paddle::framework::proto::OpDesc_Var in_tensors = op_info.outputs(in);
                std::string in_name = in_tensors.parameter();
                // printf("%s ", in_name.c_str());
            }
            google::protobuf::RepeatedPtrField<paddle::framework::proto::OpDesc_Attr> op_attr = op_info.attrs();
            // const std::string attr_name = paddle::framework::proto::OpDesc_Attr::name() ;
            // paddle::framework::proto::OpDesc *op_var = op_info.attrs();
            std::string attr_name = op_attr.Get(0).name();
            // printf("%s \n",  attr_name.c_str());
            printf("\n");
        }
        #endif
    }
}

bool PaddleSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if (!SerializerManager::SafeGet("paddle", serializer))
        return false;

    // PaddleSerializer* p_paddle = dynamic_cast<PaddleSerializer*>(serializer.get());

    return true;
}


}