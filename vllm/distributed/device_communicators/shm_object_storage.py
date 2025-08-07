# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.synchronize import Lock as LockType
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch

from vllm.config import MultiModalConfig
from vllm.logger import init_logger
from vllm.multimodal import MultiModalKwargs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

logger = init_logger(__name__)


class SingleWriterShmRingBuffer:
    """
    A single-writer, multiple-reader ring buffer implementation using shared
    memory. This class provides a thread-safe ring buffer where one process
    can write data while multiple processes/threads can read from it.
    
    Architecture:
    - Uses shared memory for cross-process communication
    - Maintains metadata for each allocated buffer chunk in the writer process
    - Supports custom "is_free" functions to determine when buffers can be
      reused
    - Each buffer chunk contains: [4-byte id][4-byte size][actual_data]
    
    Key Concepts:
    - monotonic_id_start/end: Track the range of active buffer IDs
    - data_buffer_start/end: Track the physical memory range in use
    - Automatic wraparound when reaching buffer end
    - Lazy garbage collection based on is_free_fn checks
    
    Example Usage Scenarios:
    
    Scenario 1: Simple Linear Allocation
    ```
    Buffer size: 100 bytes
    Initial state: [................................................. ]
                   ^start=end(0)
    
    After allocating 20 bytes (id=0):
    [id:0|size:20|data........][...................................]
    ^start(0)                  ^end(28)
    
    After allocating 30 bytes (id=1):  
    [id:0|size:20|data........][id:1|size:30|data..............][..]
    ^start(0)                                                   ^end(66)
    ```
    
    Scenario 2: Memory Reclamation
    ```
    Before freeing (both buffers still in use):
    [id:0|size:20|data........][id:1|size:30|data..............][..]
    ^start(0)                                                   ^end(66)
    
    After id:0 is marked free by readers:
    [FREED.................... ][id:1|size:30|data..............][..]
                                ^start(28)                       ^end(66)
    
    After both are freed:
    [FREED..............................................][..]
                                                         ^start=end(66)
    ```
    
    Scenario 3: Wraparound Allocation (continuing from Scenario 2)
    ```
    Starting from after memory reclamation in Scenario 2:
    [FREED..............................................][..]
                                                         ^start=end(66)
    
    Allocate 40 bytes (id=2) - only 34 bytes available at end, so wraparound:
    [id:2|size:40|data........................][FREED.............][..]
                                              ^end(148)            ^start(66)
    ```
    
    Scenario 4: Error Handling - Out of Space
    ```
    Starting from after wraparound allocation in Scenario 3:
    [id:2|size:40|data........................][FREED.............][..]
                                              ^end(148)            ^start(66)

    Trying to allocate 20 more bytes:
    occupied_size_new = end + size - start = 148 + 28 - 66 > buffer_size(100)
    -> Raises MemoryError: "Not enough space in the data buffer"
    ```
    
    Thread Safety:
    - Single writer: Only one process/thread should write (allocate_buf)
    - Multiple readers: Multiple processes/threads can read (access_buf) 
    - Reader synchronization handled by is_free_fn callback
    - Writer handles garbage collection (free_buf) based on reader feedback
    
    Memory Layout per Buffer Chunk:
    [4-byte monotonic_id][4-byte chunk_size][actual_data...]
    ^metadata_start                         ^data_start
    
    The monotonic_id ensures data integrity - readers can verify they're
    accessing the correct data even after buffer wraparound or reuse.
    """

    def __init__(
        self,
        data_buffer_size: int,
        is_free_fn: Optional[Callable] = lambda x: True,
        name: Optional[str] = None,
        create: bool = False,
    ):
        self.data_buffer_size = data_buffer_size
        self.is_writer = create
        self.is_free_fn = is_free_fn

        self.metadata_size = 8  # 4 bytes for id, 4 bytes for buffer size
        self.monotonic_id_end = 0
        self.monotonic_id_start = 0
        self.data_buffer_start = 0
        self.data_buffer_end = 0

        if create:
            # we are creating a buffer
            self.metadata = {
                self.monotonic_id_end: self.data_buffer_end
            }  # monotonic_id -> start address
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.data_buffer_size, name=name)
        else:
            # we are opening an existing buffer
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    # See https://docs.python.org/3/library/multiprocessing.shared_memory.html # noqa
                    # Some platforms allocate memory based on page size,
                    # so the shared memory block size may be larger or equal
                    # to the requested size. The size parameter is ignored
                    # when attaching to an existing block.
                    assert self.shared_memory.size >= self.data_buffer_size

                except FileNotFoundError:
                    # we might deserialize the object in a different node
                    # in this case, this object is not used,
                    # and we should suppress the error
                    pass
        logger.info("Shared memory created/opened with name: %s, size: %d",
                    self.shared_memory.name, self.data_buffer_size)

    def handle(self):
        return (
            self.data_buffer_size,
            self.is_free_fn,
            self.shared_memory.name,
        )

    def clear(self) -> None:
        """Clear the ring buffer."""
        assert self.is_writer, "Only the writer can clear the buffer."
        self.metadata.clear()
        self.monotonic_id_end = 0
        self.monotonic_id_start = 0
        self.data_buffer_start = 0
        self.data_buffer_end = 0

    def __del__(self):
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_writer:
                self.shared_memory.unlink()

    def allocate_buf(self, size: int) -> tuple[int, int]:
        assert self.is_writer, "Only the writer can allocate buffers."
        assert size > 0, "Size must be greater than 0"
        size += self.metadata_size  # add metadata size to the buffer size
        # reset to beginning if the buffer does have enough contiguous space
        buffer_end_reset = self.data_buffer_end % self.data_buffer_size
        if buffer_end_reset + size > self.data_buffer_size:
            buffer_end_reset = (self.data_buffer_end // self.data_buffer_size +
                                1) * self.data_buffer_size
        else:  # no reset needed
            buffer_end_reset = self.data_buffer_end

        # check if we have enough space in the data buffer
        # i.e. if the new end (self.data_buffer_end + size)
        # exceeds the start of the data buffer
        occupied_size_new = buffer_end_reset + size - self.data_buffer_start
        if occupied_size_new > self.data_buffer_size:
            raise MemoryError("Not enough space in the data buffer, "
                              "try calling try_free_buf() to free up space")
        self.data_buffer_end = buffer_end_reset

        # first 4 bytes as the monotonic id
        buf_idx = self.data_buffer_end % self.data_buffer_size
        self.shared_memory.buf[buf_idx:buf_idx + 4] = \
            self.monotonic_id_end.to_bytes(4, "little", signed=True)
        # next 4 bytes as the size of the data buffer
        self.shared_memory.buf[buf_idx + 4:buf_idx + 8] = \
            size.to_bytes(4, "little", signed=True)

        # record metadata
        self.metadata[self.monotonic_id_end] = self.data_buffer_end

        self.data_buffer_end += size
        self.monotonic_id_end += 1
        # return the start address and the monotonic id
        return self.data_buffer_end - size, self.monotonic_id_end - 1

    @contextmanager
    def access_buf(self, address: int):
        buf_idx = address % self.data_buffer_size

        # read metadata
        metadata_buff = self.shared_memory.buf[buf_idx:buf_idx +
                                               self.metadata_size]
        id = int.from_bytes(metadata_buff[:4], "little", signed=True)
        size = int.from_bytes(metadata_buff[4:8], "little", signed=True)

        # yield the data buffer and metadata
        data_buff = self.shared_memory.buf[buf_idx +
                                           self.metadata_size:buf_idx + size]
        with (memoryview(data_buff) as data_view, ):
            yield data_view, (id, size)

    def free_buf(self, nbytes=None) -> tuple[int, int]:
        # free the buffer by resetting the metadata
        # this is a no-op in shared memory,
        # but we need to keep track of the metadata
        assert self.is_writer, "Only the writer can free buffers."
        logger.info(
            "Freeing up space in the ring buffer, "
            "monotonic_id_start: %d, monotonic_id_end: %d",
            self.monotonic_id_start, self.monotonic_id_end)
        monotonic_id_before = self.monotonic_id_start
        if nbytes is None:
            nbytes = self.data_buffer_size
        freed_bytes = 0
        while self.monotonic_id_start < self.monotonic_id_end and\
            freed_bytes < nbytes:
            address = self.metadata[self.monotonic_id_start]
            with self.access_buf(address) as (data_buff, metadata):
                if self.is_free_fn(data_buff):
                    # check passed, we can free the buffer
                    del self.metadata[self.monotonic_id_start]
                    self.monotonic_id_start += 1
                    self.data_buffer_start = address
                    _, size = metadata
                    freed_bytes += size + self.metadata_size
                else:
                    # there are still readers, we cannot free the buffer
                    break

        logger.info(
            "Freed %d bytes from the ring buffer, "
            "monotonic_id_start: %d, monotonic_id_end: %d", freed_bytes,
            self.monotonic_id_start, self.monotonic_id_end)

        # wrap around the metadata indices
        if self.data_buffer_start >= self.data_buffer_size:
            self.data_buffer_start -= self.data_buffer_size
            self.data_buffer_end -= self.data_buffer_size

        monotonic_id_after = self.monotonic_id_start
        return monotonic_id_before, monotonic_id_after


@dataclass
class ShmObjectStorageHandle:
    max_object_size: int
    n_readers: int
    ring_buffer_handle: tuple[int, Callable, str]
    reader_lock: Optional[LockType]


class SingleWriterShmObjectStorage:
    """
    A single-writer, multiple-reader object storage system built on top of a
    shared memory ring buffer. Provides key-value storage with automatic memory
    management and cross-process serialization support.
    
    This storage system follows a FIFO (First-In-First-Out) eviction policy
    where the oldest objects are automatically freed when memory runs low.
    Memory is reclaimed based on reader reference counting - objects are only
    freed when all readers have finished accessing them.
    
    Architecture:
    - Single writer process can put(key, value) objects
    - Multiple reader processes can get(address, monotonic_id) objects
    - Built on SingleWriterShmRingBuffer for efficient shared memory management
    - Thread-safe operations with reader synchronization via locks
    
    Key Features:
    - **FIFO Eviction**: Oldest objects are evicted first when memory is full
    - **Reference Counting**: Objects are only freed when no readers are
      accessing them
    - **Duplicate Key Handling**: Existing keys are not overwritten, just
      re-referenced
    - **Specialized Serialization**: Optimized handling for torch.Tensor and
      MultiModalKwargs
    - **Cross-Process Safety**: Uses shared memory with proper synchronization
    - **Automatic Cleanup**: Garbage collection happens transparently during
      allocation

    Memory Layout per Object:
    [4-byte reference_count][metadata_size][serialized_object_data]
    
    Thread Safety:
    - Writer operations (put, clear) are single-threaded by design
    - Reader operations (get) are thread-safe with lock-based reference
      counting
    - Memory reclamation is handled exclusively by the writer process
    """

    def __init__(
        self,
        max_object_size: int,
        n_readers: int,
        ring_buffer: SingleWriterShmRingBuffer,
        reader_lock: Optional[LockType] = None,
    ):
        """
        Initialize the object storage.

        Args:
            max_object_size: Maximum size for a single chunk
            data_buffer_size: Total size of the ring buffer
            name: Name for existing shared memory (None to create new)
        """
        self.ring_buffer = ring_buffer
        self.is_writer = self.ring_buffer.is_writer
        self.max_object_size = max_object_size
        self.n_readers = n_readers
        self.flag_bytes = 4  # for in-use flag

        if self.is_writer:
            # Key-value mapping: key -> (address, monotonic_id)
            self.key_index: dict[str, tuple[int, int]] = {}
            # Reverse mapping: monotonic_id -> key
            self.id_index: dict[int, str] = {}
        else:
            if reader_lock is None:
                raise ValueError("Lock must be provided for readers.")

        self._reader_lock = reader_lock
        self.encoder = MsgpackEncoder()
        self.mm_decoder = MsgpackDecoder(MultiModalKwargs)
        self.tensor_decoder = MsgpackDecoder(torch.Tensor)

    def clear(self) -> None:
        """Clear the object storage."""
        if self.is_writer:
            self.ring_buffer.clear()
            self.key_index.clear()
            self.id_index.clear()
            logger.info("Object storage cleared and reinitialized.")

    def calcsize_serialize(self, value: Any) -> tuple[Any, int, bytes, int]:

        len_arr = None
        if isinstance(value, (torch.Tensor, MultiModalKwargs)):
            type_name = type(value).__name__
            value = self.encoder.encode(value)
            len_arr = [len(s) for s in value]
            nbytes = sum(len_arr)
        else:
            value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            type_name = type(value).__name__
            nbytes = len(value)

        object_metadata = (type_name, nbytes, len_arr)
        serialized_metadata = pickle.dumps(object_metadata,
                                           protocol=pickle.HIGHEST_PROTOCOL)
        return value, nbytes, serialized_metadata, len(serialized_metadata)

    def copy_to_buffer(
        self,
        data: Any,
        data_bytes: int,
        metadata: bytes,
        md_bytes: int,
        data_view: memoryview,
    ) -> None:
        if isinstance(data, bytes):
            data_view[-data_bytes:] = data
            data_view[self.flag_bytes:self.flag_bytes + md_bytes] = metadata
        elif isinstance(data, list):
            data_view[self.flag_bytes:self.flag_bytes + md_bytes] = metadata
            start_idx = self.flag_bytes + md_bytes
            for item_bytes in data:
                item_size = len(item_bytes)
                data_view[start_idx:start_idx + item_size] = item_bytes
                start_idx += item_size

        else:
            raise ValueError(
                f"Unsupported data type for serialization: {type(data)}")

    def put(self, key: str, value: Any) -> tuple[int, int]:
        """
        Store a key-value pair in the object storage.
        Attempts to free max_object_size bytes using FIFO order
        when the ring buffer runs out of space during a put() operation.
        

        Args:
            key: String key to identify the object
            value: Any serializable Python object

        Raises:
            MemoryError: If there's not enough space in the buffer
            ValueError: If the serialized object is too large
        """
        # **WILL NOT UPDATE THE VALUE IF THE KEY ALREADY EXISTS**
        if key in self.key_index:
            address, monotonic_id = self.key_index[key]
            with self.ring_buffer.access_buf(address) as (data_view, metadata):
                # >0 for in-use flag
                data_view[:self.flag_bytes] = (self.n_readers).to_bytes(
                    self.flag_bytes, "little", signed=True)
            return address, monotonic_id

        object_data, data_bytes, object_metadata, md_bytes = \
            self.calcsize_serialize(value)
        buffer_size = self.flag_bytes + data_bytes + md_bytes

        # Sanity checks
        if buffer_size > self.max_object_size:
            raise ValueError(
                f"Serialized object size ({buffer_size} bytes) exceeds "
                f"max object size ({self.max_object_size} bytes)")

        # Allocate new buffer
        try:
            address, monotonic_id = self.ring_buffer.allocate_buf(buffer_size)
        except MemoryError:
            # try to free up max_object_size bytes of space in the ring buffer
            freed_id_start, freed_id_end = self.ring_buffer.free_buf(
                self.max_object_size)
            # update the metadata after freeing up space
            for freed_id in range(freed_id_start, freed_id_end):
                key_to_free = self.id_index[freed_id]
                del self.key_index[key_to_free]
                del self.id_index[freed_id]
            # try again after freeing up space
            address, monotonic_id = self.ring_buffer.allocate_buf(buffer_size)

        # Write data to buffer
        with self.ring_buffer.access_buf(address) as (data_view, metadata):
            # >0 for in-use flag
            data_view[:self.flag_bytes] = (self.n_readers).to_bytes(
                self.flag_bytes, "little", signed=True)
            self.copy_to_buffer(object_data, data_bytes, object_metadata,
                                md_bytes, data_view)

        # Update key index
        self.key_index[key] = (address, monotonic_id)
        self.id_index[monotonic_id] = key
        return address, monotonic_id

    def get(self, address: int, monotonic_id: int) -> Any:
        # Read data from buffer
        with self.ring_buffer.access_buf(address) as (data_view, buf_metadata):
            # check id from metadata
            if buf_metadata[0] != monotonic_id:
                raise ValueError(
                    f"Data for address:id '{address}:{monotonic_id}'"
                    " has been modified or is invalid.")

            # pickle.loads do not read past the end of a pickled object
            # within a large buffer
            type_name, nbytes, len_arr = pickle.loads(
                data_view[self.flag_bytes:])
            serialized_data = bytearray(data_view[-nbytes:])

            if type_name == torch.Tensor.__name__:
                obj = []
                start_idx = 0
                for length in len_arr:
                    item_bytes = serialized_data[start_idx:start_idx + length]
                    obj.append(item_bytes)
                    start_idx += length
                obj = self.tensor_decoder.decode(obj)
            elif type_name == MultiModalKwargs.__name__:
                obj = []
                start_idx = 0
                for length in len_arr:
                    item_bytes = serialized_data[start_idx:start_idx + length]
                    obj.append(item_bytes)
                    start_idx += length
                obj = self.mm_decoder.decode(obj)
            elif type_name == bytes.__name__:
                obj = pickle.loads(serialized_data)
            else:
                raise ValueError(
                    f"Unsupported object type '{type_name}' in metadata")

            # decrease the in-use flag for reader reads
            if self._reader_lock is not None:
                with self._reader_lock:
                    reader_count = int.from_bytes(data_view[:self.flag_bytes],
                                                  "little",
                                                  signed=True)
                    reader_count -= 1
                    data_view[:self.flag_bytes] = reader_count.to_bytes(
                        self.flag_bytes, "little", signed=True)

        return obj

    def handle(self):
        """Get handle for sharing across processes."""
        return ShmObjectStorageHandle(
            max_object_size=self.max_object_size,
            n_readers=self.n_readers,
            ring_buffer_handle=self.ring_buffer.handle(),
            reader_lock=self._reader_lock,
        )

    def get_and_update_mm_cache(
        self,
        args: tuple,
    ) -> None:
        """Check if the first argument is a SchedulerOutput and update
        MultiModalKwargs from the object storage if needed."""
        if args and isinstance(args[0], SchedulerOutput):
            scheduler_output = args[0]
            for request_data in scheduler_output.scheduled_new_reqs:
                for i in range(len(request_data.mm_inputs)):
                    mm_input = request_data.mm_inputs[i]
                    if "address" in mm_input:
                        address, monotonic_id = \
                            mm_input["address"], mm_input["monotonic_id"]
                        request_data.mm_inputs[i] = \
                            self.get(address, monotonic_id)

    @staticmethod
    def is_enabled(mm_config: Optional[MultiModalConfig] = None) -> bool:
        if mm_config is None:
            return False
        if mm_config.disable_mm_preprocessor_cache:
            return False
        return mm_config.mm_preprocessor_cache_type == "shm"

    @staticmethod
    def create_from_handle(
            handle: ShmObjectStorageHandle) -> "SingleWriterShmObjectStorage":
        """
        Create a new SingleWriterShmObjectStorage from a handle.

        Args:
            handle: ShmObjectStorageHandle containing the necessary parameters

        Returns:
            A new SingleWriterShmObjectStorage instance
        """
        logger.info("Creating storage from handle: %s", handle)
        ring_buffer = SingleWriterShmRingBuffer(*handle.ring_buffer_handle)
        return SingleWriterShmObjectStorage(
            max_object_size=handle.max_object_size,
            n_readers=handle.n_readers,
            ring_buffer=ring_buffer,
            reader_lock=handle.reader_lock,
        )

    @staticmethod
    def default_is_free_check(buf: memoryview) -> bool:
        """
        Default is_free function that checks if the first 4 bytes are zero.
        This indicates that the buffer is free.
        """
        return int.from_bytes(buf[0:4], "little", signed=True) == 0
