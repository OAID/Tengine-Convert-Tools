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
#include "tengine_errno.hpp"
#include "exec_context.hpp"
#include "dev_executor.hpp"

namespace TEngine {

void ExecContext::InitContext(int empty_context)
{
    if (empty_context)
    {
        // printf("empty_context: %d \n", empty_context);
        return;
    }

    DevExecutorManager::Get();

    DevExecutorManager::StartSeqAccess();

    int n = DevExecutorManager::GetNum();
    for (int i = 0; i < n; i++)
    {
        DevExecutor* dev = DevExecutorManager::GetSeqObj();
        dev_list_.push_back(dev);
    }

    DevExecutorManager::Put();
}

ExecContext* ExecContext::GetDefaultContext(void)
{
    static ExecContext default_context("default", 0);

    return &default_context;
}

bool ExecContext::SetAttr(const char* attr_name, const void* val, int val_size)
{
    set_tengine_errno(ENOTSUP);
    return false;
}

bool ExecContext::GetAttr(const char* attr_name, void* val, int val_size)
{
    set_tengine_errno(ENOTSUP);
    return false;
}

}    // namespace TEngine
