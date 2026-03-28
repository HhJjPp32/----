import numpy as np
from scipy.integrate import quad


def get_loss_depth(x, bo2, ho2, bco, hco, ro2, bi, hi, bci, hci, ri1):
    """
    计算在坐标 x 处的垂直损失深度
    """
    x = abs(x)

    # --- 1. 物理间隙（混凝土在哪里？）---
    # 外边界 y_out
    if x <= bco / 2:
        y_out = ho2 / 2
    elif x <= bo2 / 2:
        y_out = hco / 2 + np.sqrt(max(0, ro2**2 - (x - bco/2)**2))
    else:
        y_out = 0

    # 内边界 y_in（纯几何）
    if x <= bci / 2:
        y_in = hi / 2
    elif x <= bi / 2:
        y_in = hci / 2 + np.sqrt(max(0, ri1**2 - (x - bci/2)**2))
    else:
        y_in = 0  # 侧边区域：混凝土从 y=0 开始

    gap = max(0, y_out - y_in)  # x 处的实际混凝土厚度

    # --- 2. 理论损失项（4个抛物线）---
    d1 = 0.0
    d2 = 0.0
    d3 = 0.0
    d4 = 0.0

    # P1: 外宽拱（向下）
    if bco > 0 and x <= bco / 2:
        d1 = bco / 4 - x**2 / bco

    # P3: 内宽拱（向上）
    if bci > 0 and x <= bci / 2:
        d3 = bci / 4 - x**2 / bci

    # P2: 外高拱（向左，映射到垂直损失）
    x_P2_start = bo2 / 2 - hco / 4
    if hco > 0 and x >= x_P2_start and x <= bo2 / 2:
        # 反函数: x = y^2/hco + bo2/2 - hco/4
        d2 = np.sqrt(max(0, hco * (x - x_P2_start)))

    # P4: 内高拱（向右，映射到垂直损失）
    x_P4_end = bi / 2 + hci / 4
    if hci > 0 and x >= bi / 2 and x <= x_P4_end:
        # 反函数: x = -y^2/hci + bi/2 + hci/4
        d4 = np.sqrt(max(0, hci * (x_P4_end - x)))

    # --- 3. 包络截断 ---
    # 该 x 处的总理论侵蚀
    total_theoretical_loss = d1 + d2 + d3 + d4

    # 实际损失不能超过物理间隙
    depth = min(gap, total_theoretical_loss)
    return depth


def cfdst_ke_final():
    """
    基于物理损失包络法计算 RR-CFDST 的 ke
    此版本将"物理间隙"与"理论损失"分离
    """
    # --- 步骤 0: 输入加载 ---
    try:
        with open('input.txt', 'r') as f:
            content = f.read().strip()
            data = [float(x) for x in content.split()]
    except FileNotFoundError:
        raise FileNotFoundError("input.txt not found.")

    if len(data) < 8:
        raise ValueError("Insufficient data in input.txt")

    bo = data[0]   # 外截面宽度
    ho = data[1]   # 外截面高度
    to = data[2]   # 外钢管厚度
    ro1 = data[3]  # 外圆角半径
    bi = data[4]   # 内截面宽度
    hi = data[5]   # 内截面高度
    ti = data[6]   # 内钢管厚度
    ri1 = data[7]  # 内圆角半径

    # --- 步骤 1: 几何预处理 ---
    ro2 = max(0, ro1 - to)    # 外核心内半径
    bo2 = bo - 2 * to         # 外核心宽度
    ho2 = ho - 2 * to         # 外核心高度
    bco = bo2 - 2 * ro2       # 外直边宽度
    hco = ho2 - 2 * ro2       # 外直边高度
    bci = bi - 2 * ri1        # 内直边宽度
    hci = hi - 2 * ri1        # 内直边高度

    # 计算净混凝土总面积 (Ac)
    A_outer = bo2 * ho2 - (4 - np.pi) * ro2**2
    A_inner = bi * hi - (4 - np.pi) * ri1**2
    Ac = A_outer - A_inner

    # --- 步骤 2: 通过积分计算无效面积 ---
    # 在第一象限积分"无效深度"
    # Area_Loss = 4 * integral( Loss_Depth(x) dx )

    # 积分范围: 0 到 bo2/2
    x_limit = bo2 / 2

    # 定义被积函数
    def integrand(x):
        return get_loss_depth(x, bo2, ho2, bco, hco, ro2, bi, hi, bci, hci, ri1)

    # 使用 scipy.integrate.quad 进行积分
    Aloss_Q1, _ = quad(integrand, 0, x_limit)

    Aloss_total = 4 * Aloss_Q1
    Aloss_total = min(Aloss_total, Ac)  # 物理安全限制

    # --- 步骤 3: 最终结果 ---
    ke = (Ac - Aloss_total) / Ac

    # 输出结果
    print("\n" + "=" * 44)
    print("   RR-CFDST Physical Transparent Results    ")
    print("=" * 44)
    print(f"Net Concrete Area (Ac):      {Ac:.2f} mm^2")
    print(f"Total Ineffective Area:      {Aloss_total:.2f} mm^2")
    print(f"Confinement Factor (ke):     {ke:.6f}")
    print("=" * 44)

    return ke


if __name__ == "__main__":
    cfdst_ke_final()
