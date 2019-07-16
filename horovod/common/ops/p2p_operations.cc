// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Microsoft Corp.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "p2p_operations.h"

namespace horovod {
namespace common {

PointToPointOp::PointToPointOp(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), mpi_context_(mpi_context) {}

template<class T>
void PointToPointOp::PointToPointSend(T* input_data_buffer,
                                     int64_t num_elements,
                                     int dest_rank,
                                     int tag,
                                     Communicator communicator) {
  int status;                            
  if (!global_state_->msg_chunk_enabled) {
      status = MPI_Send(input_data_buffer,
                        (int)num_elements * sizeof(T),
                        MPI_CHAR,
                        dest_rank,
                        tag,
                        mpi_context_->GetMPICommunicator(communicator));
  }
  else {
        const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
        for (int buf_index = 0; buf_index < num_elements; buf_index += chunk_size) {
          status = MPI_Send((uint8_t *)input_data_buffer + buf_index,
                            std::min((int)num_elements - buf_index, chunk_size) * sizeof(T),
                            MPI_CHAR,
                            dest_rank,
                            tag,
                            mpi_context_->GetMPICommunicator(communicator));
          status &= status;
        }
  }

  if (status != MPI_SUCCESS) {
    throw std::logic_error("MPI_Send failed, see MPI output for details.");
  }
}

template<class T>
void PointToPointOp::PointToPointRecv(T* output_data_buffer,
                                     int64_t num_elements,
                                     int src_rank,
                                     int tag,
                                     Communicator communicator) {
  int status;                            
  if (!global_state_->msg_chunk_enabled) {
      status = MPI_Recv(output_data_buffer,
                        (int)num_elements * sizeof(T),
                        MPI_CHAR,
                        src_rank,
                        tag,
                        mpi_context_->GetMPICommunicator(communicator),
                        MPI_STATUS_IGNORE);
  }
  else {
        const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
        for (int buf_index = 0; buf_index < num_elements; buf_index += chunk_size) {
          status = MPI_Recv((uint8_t *)output_data_buffer + buf_index,
                            std::min((int)num_elements - buf_index, chunk_size) * sizeof(T),
                            MPI_CHAR,
                            src_rank,
                            tag,
                            mpi_context_->GetMPICommunicator(communicator),
                            MPI_STATUS_IGNORE);
          status &= status;
        }
  }

  if (status != MPI_SUCCESS) {
    throw std::logic_error("MPI_Recv failed, see MPI output for details.");
  }
}

} // namespace common
} // namespace horovod