"""
Pipeline Scheduler for Communication-Computation Overlap

Implements double buffering and pipelining to hide All-to-All
communication latency behind expert computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.distributed as dist


@dataclass
class PipelineConfig:
    """Configuration for pipeline scheduling."""
    
    num_micro_batches: int = 2
    """Number of micro-batches for pipelining."""
    
    overlap_comm_compute: bool = True
    """Whether to overlap communication with computation."""
    
    num_streams: int = 2
    """Number of CUDA streams for overlap."""


class PipelineScheduler:
    """
    Pipeline scheduler for Expert Parallelism with All-to-All overlap.
    
    The key insight is that we can split tokens into chunks and overlap:
    1. All-to-All send/recv for chunk i+1
    2. Expert computation for chunk i
    
    This hides most of the communication latency.
    
    Example:
        >>> scheduler = PipelineScheduler(config)
        >>> output = scheduler.run_moe_layer(hidden_states, moe_block)
    """
    
    def __init__(self, config: PipelineConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Create CUDA streams for overlap
        self.streams = [
            torch.cuda.Stream(device=device)
            for _ in range(config.num_streams)
        ]
        
        # Create events for synchronization
        self.events = [
            torch.cuda.Event(enable_timing=False)
            for _ in range(config.num_streams)
        ]
        
        # Distributed info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    def run_moe_layer(
        self,
        hidden_states: torch.Tensor,
        moe_block,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run MoE layer with pipelined All-to-All.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            moe_block: The MoE block module
            routing_weights: Routing weights [batch, seq_len, top_k]
            expert_indices: Expert indices [batch, seq_len, top_k]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        if self.world_size == 1:
            # Single GPU - no communication needed
            output, _ = moe_block(hidden_states)
            return output
        
        if not self.config.overlap_comm_compute:
            # Simple non-overlapped version
            return self._run_simple(hidden_states, moe_block, routing_weights, expert_indices)
        
        # Pipelined version with overlap
        return self._run_pipelined(hidden_states, moe_block, routing_weights, expert_indices)
    
    def _run_simple(
        self,
        hidden_states: torch.Tensor,
        moe_block,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Simple non-overlapped execution."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. All-to-All dispatch (send tokens to owning experts)
        recv_tokens = self._all_to_all_dispatch(hidden_states, expert_indices)
        
        # 2. Process local experts
        local_output = self._process_local_experts(recv_tokens, moe_block)
        
        # 3. All-to-All combine (gather results back)
        output = self._all_to_all_combine(local_output, routing_weights, expert_indices)
        
        return output
    
    def _run_pipelined(
        self,
        hidden_states: torch.Tensor,
        moe_block,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Pipelined execution with communication overlap."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_chunks = self.config.num_micro_batches
        
        # Split into chunks
        chunk_size = (seq_len + num_chunks - 1) // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)
            if start < seq_len:
                chunks.append((
                    hidden_states[:, start:end],
                    routing_weights[:, start:end],
                    expert_indices[:, start:end],
                ))
        
        # Double buffering
        recv_buffers = [None, None]
        send_buffers = [None, None]
        outputs = []
        
        for i, chunk in enumerate(chunks):
            buf_idx = i % 2
            stream = self.streams[buf_idx]
            
            with torch.cuda.stream(stream):
                # Start All-to-All for this chunk
                chunk_hidden, chunk_weights, chunk_indices = chunk
                recv_buffers[buf_idx] = self._all_to_all_dispatch_async(
                    chunk_hidden, chunk_indices, stream
                )
                self.events[buf_idx].record(stream)
            
            # Process previous chunk while waiting
            if i > 0:
                prev_buf_idx = (i - 1) % 2
                self.events[prev_buf_idx].synchronize()
                
                # Process on default stream
                local_output = self._process_local_experts(
                    recv_buffers[prev_buf_idx], moe_block
                )
                
                # Combine results
                prev_chunk = chunks[i - 1]
                output = self._all_to_all_combine(
                    local_output, prev_chunk[1], prev_chunk[2]
                )
                outputs.append(output)
        
        # Process final chunk
        final_idx = (len(chunks) - 1) % 2
        self.events[final_idx].synchronize()
        local_output = self._process_local_experts(recv_buffers[final_idx], moe_block)
        final_chunk = chunks[-1]
        output = self._all_to_all_combine(local_output, final_chunk[1], final_chunk[2])
        outputs.append(output)
        
        # Concatenate all chunks
        return torch.cat(outputs, dim=1)
    
    def _all_to_all_dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch tokens to their assigned experts via All-to-All.
        
        Each GPU sends tokens to the GPU that owns the expert.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        top_k = expert_indices.size(-1)
        
        # Flatten
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_indices = expert_indices.view(-1)
        
        # Determine which GPU owns each expert
        experts_per_gpu = self._get_experts_per_gpu()
        
        # Sort tokens by destination GPU
        dest_gpu = flat_indices // experts_per_gpu
        sorted_idx = torch.argsort(dest_gpu)
        
        # Count tokens per GPU
        send_counts = []
        for gpu in range(self.world_size):
            count = (dest_gpu == gpu).sum().item()
            send_counts.append(count * hidden_size)
        
        # All-to-All exchange of counts
        recv_counts = [0] * self.world_size
        dist.all_to_all_single(
            torch.tensor(recv_counts, device=self.device),
            torch.tensor(send_counts, device=self.device),
        )
        recv_counts = [c.item() for c in recv_counts]
        
        # Prepare send buffer
        send_buffer = flat_hidden[sorted_idx]
        
        # All-to-All exchange of tokens
        total_recv = sum(recv_counts)
        recv_buffer = torch.empty(total_recv // hidden_size, hidden_size, 
                                  dtype=hidden_states.dtype, device=self.device)
        
        dist.all_to_all_single(recv_buffer.view(-1), send_buffer.view(-1))
        
        return recv_buffer
    
    def _all_to_all_dispatch_async(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Async version of dispatch for pipelining."""
        # For simplicity, just call sync version in stream
        # Production would use async NCCL operations
        with torch.cuda.stream(stream):
            return self._all_to_all_dispatch(hidden_states, expert_indices)
    
    def _process_local_experts(
        self,
        tokens: torch.Tensor,
        moe_block,
    ) -> torch.Tensor:
        """Process tokens through local experts."""
        # Just forward through the MoE block
        # The block handles expert assignment internally
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # Add batch dim
        
        output, _ = moe_block(tokens)
        return output.squeeze(0) if output.size(0) == 1 else output
    
    def _all_to_all_combine(
        self,
        expert_output: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine expert outputs via All-to-All and weighted sum.
        """
        # Reverse of dispatch - gather results back to original GPU
        # Then weight and sum
        
        batch_size, seq_len, _ = routing_weights.shape
        hidden_size = expert_output.size(-1)
        
        # TODO: Implement proper reverse All-to-All
        # For now, just return reshaped output
        return expert_output.view(batch_size, seq_len, hidden_size)
    
    def _get_experts_per_gpu(self) -> int:
        """Get number of experts assigned to each GPU."""
        # Assuming uniform distribution
        return 8 // self.world_size  # Default 8 experts


def create_pipeline_scheduler(
    num_micro_batches: int = 2,
    overlap: bool = True,
    device: str = "cuda",
) -> PipelineScheduler:
    """Create a pipeline scheduler with default configuration."""
    config = PipelineConfig(
        num_micro_batches=num_micro_batches,
        overlap_comm_compute=overlap,
    )
    return PipelineScheduler(config, device)
