import torch

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_global_pair_group,
    get_gloabl_rank,
)


_dtype = [
    torch.float32,
    torch.float,
    torch.float64,
    torch.double,
    torch.float16,
    torch.bfloat16,
    torch.half,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.short,
    torch.int32,
    torch.int,
    torch.int64,
    torch.long,
    # torch.complex32,
    # torch.complex64,
    # torch.cfloat,
    # torch.complex128,
    # torch.cdouble,
    # torch.quint8,
    # torch.qint8,
    # torch.qint32,
    torch.bool,
    # torch.quint4x2,
    # torch.quint2x4,
]

_dtype2id = {dtype: idx for idx, dtype in enumerate(_dtype)}
_id2dtype = {idx: dtype for idx, dtype in enumerate(_dtype)}


def barrier(ignore_pure_tp=False):
    if get_tensor_model_parallel_world_size() == 1 and get_pipeline_model_parallel_world_size() == 1:
        return
    if ignore_pure_tp and get_tensor_model_parallel_world_size() > 1 and get_pipeline_model_parallel_world_size() == 1:
        return 
    torch.distributed.barrier()


def pipeline_model_parallel_send_tensor_list(tensor_list):
    src_rank = get_gloabl_rank()
    dst_rank = get_pipeline_model_parallel_next_rank()
    pair_group = get_global_pair_group(src_rank, dst_rank)

    assert isinstance(tensor_list, list) and len(tensor_list) > 0
    assert all(hasattr(i, "device") and getattr(i, "device").type == "cuda" for i in tensor_list)

    tensor_list = [t.contiguous() for t in tensor_list]
    length = [len(tensor_list)]
    torch.distributed.broadcast_object_list(length, src=src_rank, group=pair_group)

    shape_list = [list(i.shape) for i in tensor_list]
    torch.distributed.broadcast_object_list(shape_list, src=src_rank, group=pair_group)

    dtype_list = [_dtype2id[i.dtype] for i in tensor_list]
    torch.distributed.broadcast_object_list(dtype_list, src=src_rank, group=pair_group)    
    for i in tensor_list:
        torch.distributed.broadcast(i, src=src_rank, group=pair_group)


def pipeline_model_parallel_recv_tensor_list():
    src_rank = get_pipeline_model_parallel_prev_rank()
    dst_rank = get_gloabl_rank() 
    pair_group = get_global_pair_group(src_rank, dst_rank)

    length = [None]
    torch.distributed.broadcast_object_list(length, src=src_rank, group=pair_group)
    length = length[0]

    shape_list = [None] * length
    torch.distributed.broadcast_object_list(shape_list, src=src_rank, group=pair_group)

    dtype_list = [None] * length
    torch.distributed.broadcast_object_list(dtype_list, src=src_rank, group=pair_group)

    tensor_list = [None] * length
    for i in range(length):
        object = torch.empty(shape_list[i], dtype=_id2dtype[dtype_list[i]], device=torch.cuda.current_device())
        torch.distributed.broadcast(object, src=src_rank, group=pair_group)
        tensor_list[i] = object

    return tensor_list


def tensor_model_parallel_all_reduce(input_):
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # All-reduce.
    torch.distributed.all_reduce(input_,
                                 group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_, dim=-1):
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor
