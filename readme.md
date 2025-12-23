# htree 项目文档

本仓库实现了 htree（分层树结构 KV 压缩 + 稀疏注意力）的：
- **Triton 版本**
- **PyTorch 参考版本**

支持：
- **GQA**（Group Query Attention）
- **MHA**（Multi-Head Attention）

## 文档

- [doc/parallel_design.md](doc/parallel_design.md)：Triton kernel 设计与并行化思路（分层归约、候选遍历、Top-K、online softmax 等）

## 代码结构

- Triton kernel
   - [src/parallel.py](src/parallel.py)：htree Triton（GQA ）
   - [src/parallel_mha.py](src/parallel_mha.py)：htree Triton（MHA）

- Naive 参考实现（PyTorch）
   - [src/naive_gqa.py](src/naive_gqa.py)：GQA 参考实现（含 RoPE / stable top-k / online softmax 的对齐逻辑）
   - [src/naive_mha.py](src/naive_mha.py)：MHA 参考实现

- 工具脚本
   - [test_all_positions_gqa.py](test_all_positions_gqa.py)：全位置正确性对比（GQA：naive_gqa vs Triton GQA）
   - [test_all_positions_mha.py](test_all_positions_mha.py)：全位置正确性对比（MHA：naive_mha vs Triton MHA）
   - [compare_special_pos.py](compare_special_pos.py)：针对特定 query 位置的深度调试（主要用于 GQA shared Top-K 逻辑）
   - [speed_test.py](speed_test.py)：速度 benchmark（对比 NSA vs htree；仅 forward，默认 GQA）
