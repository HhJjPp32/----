# CFDST_ke_Final.py 计算逻辑与思路总结

> 源文件：[CFDST_ke_Final.py](CFDST_ke_Final.py)
> 截面类型：RR-CFDST（Rectangular–Rectangular Concrete-Filled Double Skin Tubular，矩形-矩形双钢管夹层混凝土柱）
> 算法名称：**物理损失包络法**

---

## 1. 算法核心思想

将"哪些混凝土处于**有效约束区**"这一问题，转化为"哪些混凝土处于**无效区**"，再通过 $A_{ce} = A_c - A_{loss}$ 反推。

**无效区由两个独立判据共同决定**：

| 判据 | 物理含义 |
|:---|:---|
| **物理间隙 (gap)** | 截面上某点是否真的存在混凝土 |
| **抛物线理论侵蚀 (4 条抛物线之和)** | 钢板局部失稳后该点理论上无法约束的深度 |

**取二者的最小值（"包络截断"）**：理论侵蚀再大也不能超过实际存在的混凝土厚度，否则就是非物理的负面积。

$$
\boxed{d_{loss}(x) = \min\big(\text{gap}(x), \; d_1 + d_2 + d_3 + d_4\big)}
$$

最终：

$$
A_{loss} = 4 \int_0^{b_{o2}/2} d_{loss}(x)\,\mathrm{d}x, \qquad k_e = \frac{A_c - A_{loss}}{A_c}
$$

---

## 2. 输入参数（8 个）

| 符号 | 含义 |
|:---:|:---|
| `bo` | 外钢管截面宽度 |
| `ho` | 外钢管截面高度 |
| `to` | 外钢管壁厚 |
| `ro1` | 外钢管**外**角半径 |
| `bi` | 内钢管截面宽度 |
| `hi` | 内钢管截面高度 |
| `ti` | 内钢管壁厚（仅用于参数完整性，本文件计算 ke 时未使用） |
| `ri1` | 内钢管**外**角半径 |

---

## 3. 计算流程总览

```
            ┌────────────────────────┐
            │ 读取 8 个几何参数        │
            └───────────┬────────────┘
                        │
            ┌───────────▼────────────┐
            │ ① 几何预处理            │
            │   ro2, bo2, ho2,       │
            │   bco, hco, bci, hci   │
            └───────────┬────────────┘
                        │
            ┌───────────▼────────────┐
            │ ② 混凝土总面积 Ac       │
            └───────────┬────────────┘
                        │
            ┌───────────▼────────────┐
            │ ③ 第一象限数值积分       │
            │   ∫ d_loss(x) dx       │
            │   d_loss = min(        │
            │     gap(x),            │
            │     d1+d2+d3+d4        │
            │   )                    │
            └───────────┬────────────┘
                        │
            ┌───────────▼────────────┐
            │ ④ A_loss = 4 × Q1 积分 │
            │   截断至 ≤ Ac           │
            └───────────┬────────────┘
                        │
            ┌───────────▼────────────┐
            │ ⑤ ke = (Ac-A_loss)/Ac  │
            └────────────────────────┘
```

---

## 4. 详细公式

### 4.1 几何预处理

> 对应代码：[CFDST_ke_Final.py:71-78](CFDST_ke_Final.py#L71-L78)

| 派生量 | 公式 | 含义 |
|:---:|:---:|:---|
| $r_{o2}$ | $\max(0,\; r_{o1} - t_o)$ | 外管**内**角半径（混凝土感受到的角半径） |
| $b_{o2}$ | $b_o - 2 t_o$ | 外混凝土核心宽度 |
| $h_{o2}$ | $h_o - 2 t_o$ | 外混凝土核心高度 |
| $b_{co}$ | $b_{o2} - 2 r_{o2}$ | 外**直边**宽度 |
| $h_{co}$ | $h_{o2} - 2 r_{o2}$ | 外**直边**高度 |
| $b_{ci}$ | $b_i - 2 r_{i1}$ | 内**直边**宽度 |
| $h_{ci}$ | $h_i - 2 r_{i1}$ | 内**直边**高度 |

---

### 4.2 混凝土净总面积 $A_c$

> 对应代码：[CFDST_ke_Final.py:81-83](CFDST_ke_Final.py#L81-L83)

外管内轮廓所围面积扣除内管外轮廓所围面积，并各自扣除四个圆角与方角之间的差值 $(4-\pi)r^2$：

$$
A_{outer} = b_{o2}\, h_{o2} - (4 - \pi)\, r_{o2}^{\,2}
$$

$$
A_{inner} = b_i\, h_i - (4 - \pi)\, r_{i1}^{\,2}
$$

$$
\boxed{A_c = A_{outer} - A_{inner}}
$$

> $A_c \le 0$ 时返回 `None`，表示几何参数不合理。

---

### 4.3 无效深度 $d_{loss}(x)$ —— 算法核心

> 对应代码：[CFDST_ke_Final.py:6-63](CFDST_ke_Final.py#L6-L63)

由两部分组合：**(A) 物理间隙** 与 **(B) 4 条抛物线之和**，最终取最小值。

---

#### (A) 物理间隙 $\text{gap}(x)$

##### 外边界 $y_{out}(x)$

$$
y_{out}(x) =
\begin{cases}
\dfrac{h_{o2}}{2}, & 0 \le x \le \dfrac{b_{co}}{2} \quad \text{（外管直边段）} \\[8pt]
\dfrac{h_{co}}{2} + \sqrt{r_{o2}^{2} - \left(x - \dfrac{b_{co}}{2}\right)^{2}}, & \dfrac{b_{co}}{2} < x \le \dfrac{b_{o2}}{2} \quad \text{（外管圆角段）}\\[8pt]
0, & x > \dfrac{b_{o2}}{2}
\end{cases}
$$

##### 内边界 $y_{in}(x)$

$$
y_{in}(x) =
\begin{cases}
\dfrac{h_i}{2}, & 0 \le x \le \dfrac{b_{ci}}{2} \quad \text{（内管直边段）} \\[8pt]
\dfrac{h_{ci}}{2} + \sqrt{r_{i1}^{2} - \left(x - \dfrac{b_{ci}}{2}\right)^{2}}, & \dfrac{b_{ci}}{2} < x \le \dfrac{b_i}{2} \quad \text{（内管圆角段）}\\[8pt]
0, & x > \dfrac{b_i}{2} \quad \text{（侧边区域，混凝土从 } y=0 \text{ 开始）}
\end{cases}
$$

##### 物理间隙

$$
\text{gap}(x) = \max\big(0,\; y_{out}(x) - y_{in}(x)\big)
$$

> **物理意义**：在水平坐标 $x$ 处，竖直方向上"外管—内管"之间真实存在的混凝土厚度。无论理论侵蚀公式给出多大值，实际损失都不可能超过 gap。

---

#### (B) 4 条抛物线理论侵蚀

四条抛物线分别对应外管/内管在宽度/高度方向上钢板的局部失稳约束包络。

##### P1 — 外宽拱（向下）

> 由外管**宽度边**屈曲产生，开口朝向 $-y$。

$$
d_1(x) =
\begin{cases}
\dfrac{b_{co}}{4} - \dfrac{x^{2}}{b_{co}}, & 0 \le x \le \dfrac{b_{co}}{2} \\
0, & \text{其他}
\end{cases}
$$

##### P3 — 内宽拱（向上）

> 由内管**宽度边**屈曲产生，开口朝向 $+y$。

$$
d_3(x) =
\begin{cases}
\dfrac{b_{ci}}{4} - \dfrac{x^{2}}{b_{ci}}, & 0 \le x \le \dfrac{b_{ci}}{2} \\
0, & \text{其他}
\end{cases}
$$

##### P2 — 外高拱（向左，映射到垂直损失）

> 由外管**高度边**屈曲产生，开口朝向 $-x$。利用反函数 $x = y^2/h_{co} + b_{o2}/2 - h_{co}/4$，将水平方向的拱深度反映射为垂直方向的损失：

$$
d_2(x) =
\begin{cases}
\sqrt{h_{co}\!\left(x - x_{P2,\text{start}}\right)}, & x_{P2,\text{start}} \le x \le \dfrac{b_{o2}}{2} \\
0, & \text{其他}
\end{cases}
$$

其中 $x_{P2,\text{start}} = \dfrac{b_{o2}}{2} - \dfrac{h_{co}}{4}$。

##### P4 — 内高拱（向右，映射到垂直损失）

> 由内管**高度边**屈曲产生，开口朝向 $+x$。反函数 $x = -y^2/h_{ci} + b_i/2 + h_{ci}/4$：

$$
d_4(x) =
\begin{cases}
\sqrt{h_{ci}\!\left(x_{P4,\text{end}} - x\right)}, & \dfrac{b_i}{2} \le x \le x_{P4,\text{end}} \\
0, & \text{其他}
\end{cases}
$$

其中 $x_{P4,\text{end}} = \dfrac{b_i}{2} + \dfrac{h_{ci}}{4}$。

---

#### (C) 包络截断

$$
d_{loss}(x) = \min\Big(\text{gap}(x),\; d_1(x) + d_2(x) + d_3(x) + d_4(x)\Big)
$$

| 情形 | 物理解释 |
|:---|:---|
| 若 $d_1+d_2+d_3+d_4 < \text{gap}$ | 钢板约束尚未触底，理论侵蚀决定实际损失 |
| 若 $d_1+d_2+d_3+d_4 \ge \text{gap}$ | 钢板约束已"穿透"混凝土，实际损失等于物理间隙（不可能更多） |

---

### 4.4 第一象限积分与还原

> 对应代码：[CFDST_ke_Final.py:92-102](CFDST_ke_Final.py#L92-L102)

由于截面对两条对称轴均对称，仅在第一象限做数值积分（$x \in [0,\, b_{o2}/2]$），最终乘 4 还原全截面：

$$
A_{loss,Q1} = \int_{0}^{b_{o2}/2} d_{loss}(x)\,\mathrm{d}x
$$

$$
A_{loss} = 4 \cdot A_{loss,Q1}
$$

数值实现：`scipy.integrate.quad`。

**物理安全限制**：

$$
A_{loss} \leftarrow \min(A_{loss},\; A_c)
$$

防止数值误差导致 $A_{loss} > A_c$ 出现负 $k_e$。

---

### 4.5 最终结果

> 对应代码：[CFDST_ke_Final.py:106](CFDST_ke_Final.py#L106)

$$
\boxed{k_e = \dfrac{A_c - A_{loss}}{A_c}}
$$

---

## 5. 函数调用关系

```
cfdst_ke_final()                  ← 单文件入口（读 input.txt）
    │
    └── calculate_ke(bo, ho, to, ro1, bi, hi, ti, ri1)
           │   ① 几何预处理 ② 计算 Ac
           │   ③ scipy.integrate.quad 数值积分
           │       └── integrand(x) → get_loss_depth(x, ...)
           │   ④ Aloss = 4 × Q1，截断
           │   ⑤ 计算 ke
           └── 返回 dict: {bo, ho, ..., Ac, Aloss_total, ke}

process_csv(input_csv, output_csv)  ← CSV 批量入口
    └── 逐行调用 calculate_ke(...)，写出结果 CSV
```

---

## 6. 输出量

| 字段 | 含义 | 单位 |
|:---:|:---|:---:|
| `Ac` | 混凝土净总面积 | $\mathrm{mm}^2$ |
| `Aloss_total` | 全截面无效面积 | $\mathrm{mm}^2$ |
| `ke` | 有效面积系数（无量纲，0 ~ 1） | — |

---

## 7. 关键约定

1. **对称性**：所有积分仅在第一象限进行，最后 ×4 还原。
2. **数值稳健性**：所有平方根内出现负值时强制为 0；$A_{loss}$ 不允许超过 $A_c$；$A_c \le 0$ 直接返回 `None`。
3. **抛物线方向**：4 条抛物线分别对应外管/内管 × 宽边/高边屈曲，且高边的拱通过反函数映射到垂直方向，使所有损失统一在"垂直深度"维度上加和。
4. **包络截断的物理意义**：理论计算的"约束侵蚀"不能超过实际存在的"混凝土厚度"，这是该算法被命名为"包络法"的关键。
