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

#ifndef __ONEFLOW_SERIALIER_HPP__
#define __ONEFLOW_SERIALIER_HPP__

#include <iostream>
#include <fstream>
#include <functional>
#include <unordered_map>

#include "serializer.hpp"
#include "static_graph_interface.hpp"

#include "logger.hpp"
#include "oneflow/core/serving/saved_model.pb.h"

namespace TEngine {

using InputNameList = std::vector<std::string>;

class OneFlowOpConverter
{
public:
    virtual bool convert(StaticGraph* graph, StaticNode* node, const std::string& checkpoint_dir,
                         const oneflow::OperatorConf& onnx_node) const = 0;
    virtual InputNameList input_name_list() const
    {
        return InputNameList();
    }
    virtual bool custom() const {
        return false;
    }
    OneFlowOpConverter(std::string op_name);
};

class OneFlowSerializer : public Serializer
{
public:
    OneFlowSerializer()
    {
        name_ = "oneflow_loader";
        format_name_ = "oneflow";
        version_ = "0.1";
    }

    virtual ~OneFlowSerializer(){};

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
    bool LoadModelFile(const char* fname, oneflow::SavedModel& model);
    void LoadConstNode(const oneflow::GraphDef& onnx_graph, StaticGraph* graph);
    bool LoadGraph(const oneflow::GraphDef& model, const std::string &checkpoint_dir, StaticGraph* graph);
    bool LoadConstTensor(StaticGraph* graph, const oneflow::GraphDef& onnx_graph);
    void CreateInputNode(StaticGraph* graph, const oneflow::GraphDef& onnx_graph);
    bool LoadNode(StaticGraph* graph, StaticNode**, const oneflow::OperatorConf&);
    OneFlowOpConverter* GetConverterForOpConf(const oneflow::OperatorConf& op_conf);
};

}    // namespace TEngine

#endif

