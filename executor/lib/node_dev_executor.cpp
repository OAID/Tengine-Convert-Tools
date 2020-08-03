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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */

#include "node_dev_driver.hpp"
#include "node_dev_executor.hpp"

namespace TEngine {

void NodeExecutor::DevGetWorkload(DevWorkload& load)
{
    backend_dev_->GetWorkload(load);
}

bool NodeExecutor::DevGetPerf(Subgraph* graph, int policy, GraphPerf& perf)
{
    return backend_dev_->GetPerf(graph, policy, perf);
}

float NodeExecutor::DevGetFops(Subgraph* graph, int policy)
{
    return backend_dev_->GetFops(graph, policy);
}

int NodeExecutor::DevGetPolicyPriority(int policy)
{
    return backend_dev_->GetPolicyPriority(policy);
}

bool NodeExecutor::DevSetGraphAttr(void* graph_handle, const char* name, const void* val, int size)
{
    return backend_dev_->SetGraphAttr(graph_handle, name, val, size);
}

bool NodeExecutor::DevGetGraphAttr(void* graph_handle, const char* name, void* val, int size)
{
    return backend_dev_->GetGraphAttr(graph_handle, name, val, size);
}

bool NodeExecutor::DevGetProposal(Graph* graph, int policy, bool static_assign)
{
    return backend_dev_->GetProposal(graph, policy, static_assign);
}

void* NodeExecutor::DevCreateGraphHandle(Subgraph* graph)
{
    void* handle = backend_dev_->CreateGraphHandle();

    if (handle == nullptr)
        return nullptr;

    NodeContext* context = new NodeContext();

    context->dev_context = handle;
    context->sub_graph = graph;
    context->optimized_graph = nullptr;

    return context;
}

bool NodeExecutor::DevOptimizeGraph(void* graph_handle)
{
    NodeContext* context = reinterpret_cast<NodeContext*>(graph_handle);
    context->optimized_graph = context->sub_graph;

    return true;
}

Subgraph* NodeExecutor::DevGetOptimizedGraph(void* graph_handle)
{
    NodeContext* context = reinterpret_cast<NodeContext*>(graph_handle);
    return context->optimized_graph;
}

bool NodeExecutor::DevPrerun(void* graph_handle)
{
    NodeContext* context = reinterpret_cast<NodeContext*>(graph_handle);
    Subgraph* graph = context->optimized_graph;

    for (unsigned int i = 0; i < graph->seq_nodes.size(); i++)
    {
        Node* node = graph->seq_nodes[i];
        Operator* op = node->GetOp();

        if (op->GetName() == "Const" || op->GetName() == "Input")
            continue;

        node->SetAttr("DEV_RUN", true);

        if (!backend_dev_->Prerun(context->dev_context, node))
            return false;
    }

    return true;
}

void NodeExecutor::ProcessTask(const NodeTask& task)
{
    // backend_dev_->Run(task.dev_context, task.node);
}

bool NodeExecutor::DevPostrun(void* graph_handle)
{
    NodeContext* context = reinterpret_cast<NodeContext*>(graph_handle);

    Subgraph* graph = context->optimized_graph;

    for (unsigned int i = 0; i < graph->seq_nodes.size(); i++)
    {
        Node* node = graph->seq_nodes[i];

        if (!node->ExistAttr("DEV_RUN"))
            continue;

        if (!backend_dev_->Postrun(context->dev_context, node))
            return false;
    }

    return true;
}

bool NodeExecutor::DevReleaseGraphHandle(void* graph_handle)
{
    NodeContext* context = reinterpret_cast<NodeContext*>(graph_handle);

    backend_dev_->ReleaseGraphHandle(context->dev_context);

    delete context;

    return true;
}

const dev_id_t& NodeExecutor::DevGetID(void)
{
    return backend_dev_->GetDeviceID();
}

const dev_type_t& NodeExecutor::DevGetType(void)
{
    return backend_dev_->GetDeviceType();
}

dev_status_t NodeExecutor::DevGetStatus(void)
{
    return backend_dev_->GetDeviceStatus();
}

bool NodeExecutor::Init(void)
{
    if (SupportNonblockRun() && create_worker_)
    {
        auto f = std::bind(&NodeExecutor::ProcessTask, this, std::placeholders::_1);
        worker_ = new WorkerThread<NodeTask>(f);

        worker_->SetQueue(&task_queue_, &worker_lock_, &worker_cv_);
        worker_->LaunchWorker();
        worker_->Activate(-1);
    }

    return true;
}

bool NodeExecutor::Release(void)
{
    if (SupportNonblockRun() && create_worker_)
    {
        if (worker_)
        {
            worker_->Deactivate();
            delete worker_;
        }
    }

    return true;
}

void NodeExecutor::UnbindDevice(void)
{
    backend_dev_ = nullptr;
}

void NodeExecutor::BindDevice(Device* device)
{
    NodeDevice* dev = dynamic_cast<NodeDevice*>(device);
    backend_dev_ = dev;
}

bool NodeExecutor::DevStart(void)
{
    return backend_dev_->Start();
}

bool NodeExecutor::DevStop(void)
{
    return backend_dev_->Stop();
}

}    // namespace TEngine
