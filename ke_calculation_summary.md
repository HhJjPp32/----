# CFDST 有效面积系数 ke 计算逻辑汇总

> 基于文件 [calculate_ke_final_first_try.py](calculate_ke_final_first_try.py)
> 参考文献：Zhenlin Li et al. (2022)
> 截面类型：CFDST（Concrete-Filled Double Skin Tubular，方/矩形双钢管夹层混凝土柱）

---

## 1. 物理背景

CFDST 柱由**外钢管**与**内钢管**组成，两钢管之间填充混凝土。在轴向受压时，钢管对混凝土产生**约束作用**，但由于截面几何与板件局部失稳，混凝土并非全截面有效，而是仅有一部分混凝土处于"有效约束区"。

**有效面积系数 ke** 即用于量化有效约束面积占混凝土总面积的比例：

$$
k_e = \frac{A_{ce}}{A_c}
$$

其中：
- $A_c$：混凝土总（净）面积
- $A_{ce}$：有效约束的混凝土面积

---

## 2. 输入参数（共 8 个）

| 参数 | 含义 |
|:---:|:---|
| `bo` | 外钢管截面**宽度** |
| `ho` | 外钢管截面**高度** |
| `to` | 外钢管**壁厚** |
| `ro1` | 外钢管**外角半径** |
| `bi` | 内钢管截面**宽度** |
| `hi` | 内钢管截面**高度** |
| `ti` | 内钢管**壁厚** |
| `ri1` | 内钢管**外角半径** |

---

## 3. 计算流程总览

```
输入 8 个参数
   │
   ▼
① 计算外管内角半径 ro2
   │
   ▼
② 计算混凝土总面积 Ac
   │
   ▼
③ 数值积分计算无效面积 Abi、Ahi
   │
   ▼
④ 计算有效面积 Ace = Ac − 2·Abi − 2·Ahi
   │
   ▼
⑤ 输出 ke = Ace / Ac
```

---

## 4. 详细公式

### 4.1 外管内角半径 ro2（对应论文式 5）

由于钢管壁厚的存在，混凝土所"看到"的外管内角半径需按壁厚比例缩减。代码中按宽度方向与高度方向分别计算：

$$
r_{o2,w} = \frac{h_o - 2 t_o}{h_o} \cdot r_{o1}
\qquad
r_{o2,h} = \frac{b_o - 2 t_o}{b_o} \cdot r_{o1}
$$

后续面积计算使用平均值：

$$
\overline{r_{o2}} = \frac{r_{o2,w} + r_{o2,h}}{2}
$$

> 对应代码 [calculate_ke_final_first_try.py:46-52](calculate_ke_final_first_try.py#L46-L52) 与 [calculate_ke_final_first_try.py:73](calculate_ke_final_first_try.py#L73)

---

### 4.2 混凝土总面积 Ac

外管"内轮廓"所围面积减去内管"外轮廓"所围面积，并扣除两端圆角（圆角部分实际无混凝土）。

$$
A_{outer} = (b_o - 2 t_o)(h_o - 2 t_o) - (4 - \pi)\,\overline{r_{o2}}^{\,2}
$$

$$
A_{inner} = b_i \, h_i - (4 - \pi)\,r_{i1}^{2}
$$

$$
\boxed{A_c = A_{outer} - A_{inner}}
$$

其中 $(4-\pi)r^2$ 项表示一个"方形 − 圆"差值，即四个角部去除的小三角形面积。

> 对应代码 [calculate_ke_final_first_try.py:74-76](calculate_ke_final_first_try.py#L74-L76)

---

### 4.3 无效面积 Abi 与 Ahi（数值积分）

定义两个方向的几何参数，分别求取宽度方向无效面积 $A_{bi}$ 与高度方向无效面积 $A_{hi}$：

| 派生量 | 宽度方向（用于 $A_{bi}$） | 高度方向（用于 $A_{hi}$，宽高互换） |
|:---:|:---:|:---:|
| $h_{o2}$ | $h_o - 2 t_o$ | $b_o - 2 t_o$ |
| $b_{co}$ | $b_o - 2 t_o - 2 r_{o2,w}$ | $h_o - 2 t_o - 2 r_{o2,h}$ |
| $b_{ci}$ | $b_i - 2 r_{i1}$ | $h_i - 2 r_{i1}$ |
| $h_i$（局部） | $h_i$ | $b_i$ |
| $h_{ci}$ | $h_i - 2 r_{i1}$ | $b_i - 2 r_{i1}$ |
| $b_i$（局部） | $b_i$ | $h_i$ |

每个方向通过数值积分计算（利用对称性，仅积分 $[0, x_{max}]$ 后乘 2）：

$$
A_{bi} \text{ or } A_{hi} \;=\; 2 \int_{0}^{x_{max}} d_{inv}(x)\,\mathrm{d}x
$$

其中：
- 积分上限 $x_{max} = \max\left(\dfrac{b_{co}}{2} + r_{o2}, \; \dfrac{b_{ci}}{2} + r_{i1}\right)$
- $d_{inv}(x)$：x 处的**无效深度**（见 4.4 节）
- 实现：`scipy.integrate.quad`

> 对应代码 [calculate_ke_final_first_try.py:101-136](calculate_ke_final_first_try.py#L101-L136)

---

### 4.4 无效深度 $d_{inv}(x)$ 计算

无效深度由两部分共同决定：**(A) 物理几何间隙** 与 **(B) 抛物线约束理论侵蚀**，二者取较小值。

#### (A) 物理几何间隙 gap(x)

##### 外管上半部边界 $y_{out}(x)$

$$
y_{out}(x) =
\begin{cases}
\dfrac{h_{o2}}{2}, & 0 \le x \le \dfrac{b_{co}}{2} \quad \text{（直边段）} \\[6pt]
\dfrac{h_{co}}{2} + \sqrt{\,r_{o2}^{2} - \left(x - \dfrac{b_{co}}{2}\right)^{2}}, & \dfrac{b_{co}}{2} < x \le \dfrac{b_{co}}{2} + r_{o2} \quad \text{（圆角段）}\\[8pt]
0, & x > \dfrac{b_{co}}{2} + r_{o2}
\end{cases}
$$

其中 $h_{co} = h_{o2} - 2 r_{o2}$。

##### 内管上半部边界 $y_{in}(x)$

$$
y_{in}(x) =
\begin{cases}
\dfrac{h_i}{2}, & 0 \le x \le \dfrac{b_{ci}}{2} \\[6pt]
\dfrac{h_{ci}}{2} + \sqrt{\,r_{i1}^{2} - \left(x - \dfrac{b_{ci}}{2}\right)^{2}}, & \dfrac{b_{ci}}{2} < x \le \dfrac{b_{ci}}{2} + r_{i1} \\[8pt]
\sqrt{\,h_{ci}\!\left(\dfrac{h_{ci}}{4} - \!\left(x - \dfrac{b_i}{2}\right)\right)}, & \dfrac{b_{ci}}{2} + r_{i1} < x \le \dfrac{b_i}{2} + \dfrac{h_{ci}}{4} \quad \text{（抛物线过渡）}\\[8pt]
0, & \text{其他}
\end{cases}
$$

##### 物理间隙

$$
\text{gap}(x) = \max\left(0,\; y_{out}(x) - y_{in}(x)\right)
$$

#### (B) 抛物线约束理论侵蚀

钢板局部屈曲约束区采用抛物线模型描述：

- 外管抛物线（宽方向）：

$$
d_{out}(x) =
\begin{cases}
\dfrac{b_{co}}{4} - \dfrac{x^{2}}{b_{co}}, & 0 \le x \le \dfrac{b_{co}}{2} \\
0, & \text{其他}
\end{cases}
$$

- 内管抛物线（宽方向）：

$$
d_{in}(x) =
\begin{cases}
\dfrac{b_{ci}}{4} - \dfrac{x^{2}}{b_{ci}}, & 0 \le x \le \dfrac{b_{ci}}{2} \\
0, & \text{其他}
\end{cases}
$$

#### (C) 取最小值得到无效深度

$$
\boxed{d_{inv}(x) = \min\big(\text{gap}(x), \; d_{out}(x) + d_{in}(x)\big)}
$$

> 对应代码 [calculate_ke_final_first_try.py:139-205](calculate_ke_final_first_try.py#L139-L205)

---

### 4.5 有效面积 Ace 与有效面积系数 ke

由于截面具有两条对称轴，两侧均有无效区域，故：

$$
A_{ce} = A_c - 2\,A_{bi} - 2\,A_{hi}
$$

代码中再做截断，避免因数值误差出现负值：

$$
A_{ce} \leftarrow \max(A_{ce}, 0)
$$

最终：

$$
\boxed{k_e = \dfrac{A_{ce}}{A_c}}
$$

> 对应代码 [calculate_ke_final_first_try.py:83-86](calculate_ke_final_first_try.py#L83-L86)

---

## 5. 函数调用关系

```
calculate_ke_final(input_file)             ← 单文件入口
   │
   ├── 读取 input.txt 中 8 个参数
   │
   ├── 调用 calc_invalid_area_unified()  ──┐
   │                                       │
   │                                       └── 调用 get_invalid_depth()
   │                                            （被 scipy.integrate.quad 数值积分）
   └── 输出 ke 等结果

calculate_ke_from_csv(input_csv, output_csv)  ← 批量 CSV 处理入口
   │
   └── 逐行调用 calculate_ke_params()
        │
        └── 同上：calc_invalid_area_unified() → get_invalid_depth()
```

---

## 6. 输出结果

| 输出量 | 含义 | 单位 |
|:---:|:---|:---:|
| `Ac` | 混凝土总（净）面积 | $\mathrm{mm}^2$ |
| `Abi` | 宽度方向无效面积（单侧） | $\mathrm{mm}^2$ |
| `Ahi` | 高度方向无效面积（单侧） | $\mathrm{mm}^2$ |
| `Ace` | 有效混凝土面积 | $\mathrm{mm}^2$ |
| `ke` | 有效面积系数（无量纲，0~1） | — |

---

## 7. 关键约定与注意事项

1. **对称性**：积分仅在第一象限（或半区间）进行，再乘以 2 或 4 还原全截面。
2. **方向解耦**：宽度方向与高度方向分别计算，使用同一函数 `calc_invalid_area_unified` 但传入不同的几何参数（"宽高互换"）。
3. **角部处理**：所有面积公式都包含 $(4 - \pi) r^2$ 修正项以扣除圆角与方形之间的差值面积。
4. **数值稳健性**：根号内出现负值时强制截为 0；$A_{ce}$ 出现负值时截为 0；$A_c \le 0$ 时返回 $k_e = 0$。
5. **积分实现**：使用 `scipy.integrate.quad`，`limit=100` 限制子区间数量。

---

## 8. 与同目录 [CFDST_ke_Final.py](CFDST_ke_Final.py) 的差异

| 项 | `calculate_ke_final_first_try.py` | `CFDST_ke_Final.py` |
|:---|:---|:---|
| 外管内角半径 $r_{o2}$ | 按宽/高方向分别用比例式 $r_{o2}=\frac{h_o-2t_o}{h_o}r_{o1}$ | 直接 $r_{o2}=\max(0, r_{o1}-t_o)$ |
| 无效面积积分对象 | "无效深度" = min(gap, d_out + d_in)，仅含 2 条抛物线 | "损失深度" = min(gap, d1+d2+d3+d4)，含 4 条抛物线（外宽/外高/内宽/内高） |
| 截面对称性还原系数 | 宽、高方向各算半区，最终 $A_c - 2A_{bi} - 2A_{hi}$ | 仅在第一象限积分，最终 $\times 4$ |
| 安全限制 | $A_{ce} \ge 0$ 截断 | $A_{loss} \le A_c$ 截断 |

两套实现思路一致（"物理间隙 ∩ 抛物线包络"），但圆角缩减规则与抛物线项数不同，结果可能存在差异。
