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
        # "this buffer's previous reader has finished" events -- see
        # prefetch()'s docstring for the write-after-read hazard these
        # close. Never recorded until the first mark_read_done(buf_idx)
        # call; torch.cuda.Event.wait() on a never-recorded event is a
        # no-op (nothing queued on it yet), which is exactly right for a
        # buffer's very first prefetch, before anything has ever read it.
        self.read_done_events = [torch.cuda.Event() for _ in range(num_buffers)]

    def prefetch(self, buf_idx: int, gpu_buffer: torch.Tensor, host_data: torch.Tensor) -> None:
        """Asynchronously copies host_data (must be pinned CPU memory) into
        gpu_buffer on buf_idx's own stream, then records an event on that
        stream marking completion. Returns immediately -- the copy runs
        concurrently with whatever the default stream is doing.

        First makes the copy stream wait on buf_idx's read_done_event --
        this buffer was very likely used as a compute kernel's INPUT two
        cycles ago (WeightPipeline alternates 2 buffers), and without this
        wait, the copy (write) could start before that earlier kernel
        (read) has actually finished, a write-after-read race with no
        enforced order between the two independent streams involved. This
        was a real, silent bug here until caught by M7's Qwen3-1.7B
        experiments: 14B's much larger per-weight GEMV time happened to
        keep every reader far ahead of the next writer purely by timing
        luck, so the missing dependency never surfaced there; 1.7B's much
        faster kernels made it visible as non-deterministic greedy decode
        output. See docs/LEARNING_NOTES.md's M7 task 4 entry."""
        if not host_data.is_pinned():
            raise ValueError("prefetch: host_data must be pinned (see weight_store.PinnedWeightStore) for a real async copy")
        if gpu_buffer.shape != host_data.shape:
            raise ValueError(f"prefetch: gpu_buffer shape {gpu_buffer.shape} != host_data shape {host_data.shape}")
        stream = self.streams[buf_idx]
        stream.wait_event(self.read_done_events[buf_idx])
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

    def mark_read_done(self, buf_idx: int) -> None:
        """Records an event on the CURRENT stream marking "whatever just
        read buf_idx is done." Call this immediately after launching the
        kernel that consumes buf_idx's data (right after `wait(buf_idx)`
        and the kernel launch, not after a CPU-blocking sync -- recording
        only marks a point in the stream's queue, it doesn't itself wait).
        The NEXT prefetch() into this same buf_idx will wait on this event
        before its copy starts, closing the write-after-read hazard
        `prefetch`'s docstring describes."""
        self.read_done_events[buf_idx].record(torch.cuda.current_stream())

    def synchronize(self, buf_idx: int) -> None:
        """CPU-side blocking wait on buf_idx's transfer. Only for timing
        and tests -- the actual decode loop should use `wait`, which never
        blocks the CPU."""
        self.events[buf_idx].synchronize()
