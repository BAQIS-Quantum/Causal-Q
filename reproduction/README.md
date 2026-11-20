# Causal-Q

**Code and data attachment for manuscript HR10050N · Version 2026.11.20**

This package reproduces the numerical results of *A Model-Free, Data-Driven Causal Graph Framework for Characterization and Calibration of Superconducting Quantum Chips*. The [results directory](results/) contains Figures 2–10, Tables I–VIII, appendix statistics, and supplemental Figures S1–S3. Figure 1 is a workflow diagram.

《A Model-Free, Data-Driven Causal Graph Framework for Characterization and Calibration of Superconducting Quantum Chips》的数值复现包。打开 [results](results/) 即可查看终稿图 2–10、表 I–VIII、附录统计及补充图 S1–S3。图 1 是流程示意图，没有数值输入。

## Reproduce the paper / 运行

Run these commands from the unpacked `causal-Q` directory. The reference environment is CPython 3.10.14 on CPU. The pinned [requirements](requirements.txt) install the included NPEET 1.0.1 wheel, built from [upstream commit 8b0d948](https://github.com/gregversteeg/NPEET/tree/8b0d9485423f74e5eb199324cf362765596538d3). To create all PDFs, install `rsvg-convert` (librsvg) and `pdfunite` (Poppler) on your PATH. Without them the numerical outputs, Figure 2 SVG, and the individual supplemental PDFs are still available. On Windows, activate the environment with `.venv\Scripts\activate`.

在解压后的 `causal-Q` 目录运行以下命令。核对论文数值时使用的环境为 CPython 3.10.14、NumPy 1.26.4、SciPy 1.14.1，完整依赖见 `requirements.txt`。计算使用 CPU。互信息估计器 NPEET 及其许可证随包提供。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --out results
```

默认入口从随包的模拟样本、信息估计值及逐次查询记录生成论文图表。它计算秩相关、平均精度、复查覆盖率和最佳查询曲线，并从逐项条件独立分数重建附录 D 的联合图。表 IV 的两个终点会用更严格的积分容差再计算一次。

The default command uses the archived observations, information estimates, and per-query records. It computes the paper's ranking statistics, recheck coverage, and optimization curves, reconstructs the Appendix D collider counts, and reintegrates the Table IV endpoints with tighter tolerances. To regenerate samples from seeds, estimate MI/CMI, and run the pulse dynamics and optimization, use a separate output directory. The worker count can be 1–4.

要从随机种子重新生成样本、估计 MI/CMI，并运行脉冲积分和校准优化，使用单独的输出目录。这个模式耗时较长，`--workers` 可取 1–4。

```bash
python reproduce.py --from-scratch --out results-full --workers 4
```

The regenerated Figure 4 graphs feed Figures 5–6 and Appendix D. For the segmented-chain amplitude-block comparison, the full run reintegrates the published trial amplitudes. The reported single-qubit-block AUC is **0.6920**. To let the optimizer choose new trials, run `python calibration/simulate.py --out independent --case chain --study policy --repeat-block-search`. This independent search gave **0.6833** on the reference machine. For Table IV, independent fixed-gate optimization yielded the reported mean Haar fidelities of **0.999470** and **0.999673** to six decimal places. The tighter integration check in Methods uses the archived endpoints.

该命令把新算出的图 4 边集传给图 5、图 6 和附录 D。分段链的三组幅度分块优化对浮点舍入敏感。完整模式在论文记录的试探幅度上重新积分并计算 AUC，可复现逐比特方法的 0.6920。需要让求解器独立选择试探幅度时，运行 `python calibration/simulate.py --out independent --case chain --study policy --repeat-block-search`。本机独立搜索得到 0.6833。

The final CSV column for Table IV records the maximum change on tighter integration of each archived endpoint. At double precision, this diagnostic can vary by a rounding unit between runs. Methods reports the largest change across both plans ($3.4\times10^{-16}$ as an upper bound).

独立重跑固定门优化时，两种排列的终态平均 Haar 保真度按表 IV 的六位小数分别为 0.999470 和 0.999673。Methods 报告的更严格积分检查使用论文原始终点。即使使用同一个终点，CSV 末列的积分差值也可随浮点舍入变化；两种排列的最大变化都在论文给出的 $3.4\times10^{-16}$ 上界内。

如需在新机器上导出全部 PDF，可安装 librsvg 的 `rsvg-convert` 和 Poppler 的 `pdfunite`。只安装 Python 依赖也能生成全部数值、图 2 的 SVG 和补充材料的三份单页 PDF。Windows 下激活虚拟环境的命令为 `.venv\Scripts\activate`。

## Figure and table index / 图表索引

The table below links each paper output to its input directory. The five modules cover flux dependencies, pulse dependencies, graph-based calibration, fixed-target gates, and collider orientation. The archived inputs are in each module's `data/` directory.

| 终稿位置 | 结果 | 数据及计算 |
| --- | --- | --- |
| 图 2、3 | [四比特磁通图](results/flux/figures/figure2.svg)、[十六比特磁通图](results/flux/figures/figure3_recomputed.pdf)、[筛边与耦合排序指标](results/flux/reference_metrics.json) | `flux/data/reference/` 保存四组参考样本、耦合矩阵、MI 与端点条件分数。 |
| 图 4 | [脉冲依赖图](results/pulse/Fig4.pdf)、[逐拓扑统计](results/pulse/summary.json) | `pulse/data/` 含无耦合 40,000、分段链 20,000、稠密 40,000 个八比特样本。 |
| 图 5、表 I–III、附录 B | [图 5](results/calibration/figure5.pdf)、[结构与查询数](results/calibration/table_1.csv)、[复查与调度](results/calibration/table_2.csv)、[终态优化结果](results/calibration/table_3.csv)、[支持矩阵扰动](results/calibration/appendix_b.csv) | `calibration/data/` 保存工作点、响应记录及优化轨迹。每个工作点的目标与初始载频参考系固定。 |
| 图 6、表 IV | [固定目标门的查询曲线](results/fixed_gate/figure6.pdf)、[首次达标查询数](results/fixed_gate/figure6.csv)、[两种频率排列的保真度](results/fixed_gate/table_IV.csv) | `fixed_gate/data/` 保存逐查询、逐比特的 Haar 保真度和联合优化设置；目标门为 $X_{\pi/2}$。 |
| 附录 A、表 V | [三种频率模型的秩相关](results/flux/table_V_frequency_models.csv) | 使用相同的磁通行，比较标量近似、裸频率精确谱和 transmon 响应精确谱。 |
| 附录 C、图 7–10、表 VI–VII | [图 7](results/flux/figures/flux_robustness.pdf)、[图 8](results/flux/figures/flux_k_sensitivity_appendix.pdf)、[图 9](results/flux/figures/flux_k_other_cases_appendix.pdf)、[图 10](results/flux/figures/flux_weak_pair_distribution_appendix.pdf)、[重复采样](results/flux/table_VI_flux_rank_correlations.csv)、[配对噪声比较](results/flux/table_VII_paired_frequency_noise.csv)、[补充图 S1–S3](results/flux/figures/supplement_flux_robustness.pdf) | `flux/data/robustness/` 含五组独立抽样，以及同一批样本在不同噪声、相关性和邻居数下的分数。 |
| 附录 D、表 VIII | [七组样本的定向计数](results/orientation/table_VIII.csv) | `orientation/data/` 保存每项 MI/CMI 的未舍入比特数。各算例的输出 JSON 列出逐边分类及文中的碰撞结构示例。 |

磁通图的筛边参数为 $k_{\rm MI}=130$、$k_{\rm CMI}=3$，显示权重使用 $k_{\rm MI}=3$，基准配对分数使用 $k_{\rm CMI}=30$。图 4 热图使用 $k_{\rm MI}=130$。附录 D 的联合定向单独使用 $k=10$、0.02 bit 阈值和最大二阶条件集。图 4、5 的四探针分数和图 6 的单比特通道 Haar 保真度分别按各自定义输出。
