
def _assert_ptr_aligned(t: torch.Tensor, *, alignment: int = 16, name: str = "tensor") -> None:
    """
    Triton JIT 会把指针的 divisibility/alignment 作为编译特化条件之一；
    如果同一个 kernel 在不同调用中遇到不同的对齐情况，可能触发 JIT cache miss。
    这里在 Python 侧提前检查，便于定位产生非对齐 view / slice 的来源。
    """
    if not isinstance(t, torch.Tensor):
        return
    if t.numel() == 0:
        return
    ptr = t.data_ptr()
    if ptr % alignment == 0:
        return

    # storage 指针 & offset 有助于诊断是否为 view/slice 导致
    try:
        storage_ptr = t.untyped_storage().data_ptr()
    except Exception:
        # 兼容老版本 torch
        storage_ptr = t.storage().data_ptr()  # type: ignore[attr-defined]

    raise RuntimeError(
        f"[htree][triton][alignment] '{name}' data_ptr not {alignment}-byte aligned: "
        f"data_ptr=0x{ptr:x} (mod {alignment} = {ptr % alignment}), "
        f"storage_ptr=0x{storage_ptr:x}, storage_offset={t.storage_offset()}, "
        f"dtype={t.dtype}, device={t.device}, shape={tuple(t.shape)}, stride={tuple(t.stride())}, "
        f"is_contiguous={t.is_contiguous()}"
    )


def _assert_ptrs_aligned(named_tensors: Iterable[Tuple[str, torch.Tensor]], *, alignment: int = 16) -> None:
    for name, t in named_tensors:
        _assert_ptr_aligned(t, alignment=alignment, name=name)


# 用法
# _assert_ptrs_aligned(
#     [
#         ("global_output", global_output),
#         ("global_sum", global_sum),
#         ("output", output),
#     ],
#     alignment=16,
# )