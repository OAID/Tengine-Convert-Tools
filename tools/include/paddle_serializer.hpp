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
#ifndef __PADDLE_SERIALIZER_HPP__
#define __PADDLE_SERIALIZER_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <map>
#include <vector>
#include <string>

#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"

#include "framework.pb.h"

namespace TEngine {

struct PaddleNode
{
    std::string op;
    std::string name;
    std::vector<std::pair<std::string, std::string>> inputs;
    paddle::framework::proto::OpDesc op_desc;
};

struct PaddleParam
{
    int dim_size;
    int data_len;
    std::string name;
    std::vector<int> dims;
    void* raw_data;
};

class PaddleSerializer : public Serializer
{
public:
    PaddleSerializer()
    {
        name_ = "paddle_loader";
        version_ = "0.1";
        format_name_ = "paddle";
    }
    virtual ~PaddleSerializer(){}

    unsigned int GetFileNum(void) override
    {
        return 2;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }
protected:
    bool LoadBinaryFile(const char* fname, std::vector<PaddleParam>& paramlist, paddle::framework::proto::ProgramDesc pp_net);
    bool LoadTextFile(const char* fname, paddle::framework::proto::ProgramDesc& pp_net);

    bool LoadGraph(paddle::framework::proto::ProgramDesc& pp_net, const std::vector<PaddleParam>& paramlist, StaticGraph* graph);
    bool LoadConstTensor(const std::vector<PaddleParam>& paramlist, StaticGraph* graph);
    bool CreateInputNode(std::map<std::string, std::vector<int>>& all_tensor_dims, std::vector<PaddleNode>& nodelist, StaticGraph* graph);
    bool LoadNode(StaticGraph* graph, StaticNode* node, PaddleNode& pp_node, std::map<std::string, std::vector<int>>& all_tensor_dims);
    bool ConstructGraph(paddle::framework::proto::ProgramDesc& pp_net, std::vector<PaddleNode>& pp_nodelist);    
};

}

#endif