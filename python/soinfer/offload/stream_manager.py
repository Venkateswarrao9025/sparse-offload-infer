"""M6 task 2: N CUDA streams, double (then triple) buffering, cudaMemcpyAsync
+ events, to prefetch layer i+1's weights while layer i's kernels are still
running on the default (compute) stream.
"""
from __future__ import annotations

import torch


class StreamManager:
    """Manages `num_buffers` (2 = double buffering, 3 = triple) copy
    streams, each paired with a CUDA event marking "this buffer's transfer
    is done." Compute is assumed to run on the default stream; `wait`
    makes the default stream block on a buffer's transfer before a kernel
    reads it, via `cudaStreamWaitEvent` -- a GPU-side ordering constraint,
    not a CPU sync, so the CPU keeps issuing work while the wait is
    pending.
    """

    def __init__(self, num_buffers: int = 2):
        if num_buffers < 1:
            raise ValueError("num_buffers must be >= 1")
        self.num_buffers = num_buffers
        self.streams = [torch.cuda.Stream() for _ in range(num_buffers)]
        self.events = [torch.cuda.Event() for _ in range(num_buffers)]

    def prefetch(self, buf_idx: int, gpu_buffer: torch.Tensor, host_data: torch.Tensor) -> None:
        """Asynchronously copies host_data (must be pinned CPU memory) into
        gpu_buffer on buf_idx's own stream, then records an event on that
        stream marking completion. Returns immediately -- the copy runs
        concurrently with whatever the default stream is doing."""
        if not host_data.is_pinned():
            raise ValueError("prefetch: host_data must be pinned (see weight_store.PinnedWeightStore) for a real async copy")
        if gpu_buffer.shape != host_data.shape:
            raise ValueError(f"prefetch: gpu_buffer shape {gpu_buffer.shape} != host_data shape {host_data.shape}")
        stream = self.streams[buf_idx]
        with torch.cuda.stream(stream):
            gpu_buffer.copy_(host_data, non_blocking=True)
        self.events[buf_idx].record(stream)

    def wait(self, buf_idx: int) -> None:
        """Makes the current stream (the default/compute stream, if called
        outside any `torch.cuda.stream` block) wait for buf_idx's transfer
        to finish. Call this immediately before launching a kernel that
        reads that buffer -- not earlier (it would serialize prefetch and
        compute) and not later (the kernel could read a partially-copied
        buffer)."""
        torch.cuda.current_stream().wait_event(self.events[buf_idx])

    def synchronize(self, buf_idx: int) -> None:
        """CPU-side blocking wait on buf_idx's transfer. Only for timing
        and tests -- the actual decode loop should use `wait`, which never
        blocks the CPU."""
        self.events[buf_idx].synchronize()
