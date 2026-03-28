import numpy as np
import pandas as pd
from scipy import integrate, optimize

def calculate_Abi(b_co, h_o2, b_ci, h_i, r_i1, h_ci=None, b_i=None):
    """
    计算沿宽度方向的无效约束面积 Abi (Area of ineffective confined concrete along the width direction)

    基于 Li et al. (2022) 提出的5种计算场景：
    - Case 1: 外钢管拱与内钢管拱不相交
    - Case 2: 外钢管宽边拱与内钢管宽边拱相交
    - Case 3: 外钢管拱与内钢管圆角相交
    - Case 4: 外钢管拱与内钢管高边拱相交
    - Case 5: 外钢管拱与线 y=h_o2/2 相交且超出内钢管高边拱

    特殊情况:
    - 当 b_ci=0 时，只计算外钢管的无效约束面积 (b_co²/6)
    - 当 b_co=0 且 b_ci=0 时，返回 0

    参数:
    -----------
    b_co : float
        核心混凝土宽度（外钢管内平直边长度）[mm]
    h_o2 : float
        核心混凝土深度（外钢管内高度）[mm]
    b_ci : float
        内钢管平直边宽度 [mm]
    h_i : float
        内钢管外深度 [mm]
    r_i1 : float
        内钢管外角半径 [mm]
    h_ci : float, optional
        内钢管平直边深度（Case 4和5需要）[mm]
    b_i : float, optional
        内钢管总宽度（Case 4和5需要，若未提供则假设 b_i ≈ b_ci + 2*r_i1）[mm]

    返回:
    --------
    Abi : float
        无效约束面积 [mm²]
    case : int
        识别的计算场景 (0-5)，0表示特殊情况
    """

    # 特殊情况处理
    if b_co == 0 and b_ci == 0:
        return 0.0, 0

    if b_co == 0 and b_ci > 0:
        return (b_ci ** 2) / 6.0, 0

    if b_ci == 0:
        return (b_co ** 2) / 6.0, 0

    # 计算几何参数
    delta_h = h_o2 - h_i

    if b_i is None:
        b_i = b_ci + 2 * r_i1

    # 检查基本几何约束
    if b_co <= b_ci:
        raise ValueError(f"外钢管宽度 b_co ({b_co}) 必须大于内钢管宽度 b_ci ({b_ci})")
    if delta_h < 0:
        raise ValueError(f"核心混凝土高度 h_o2 ({h_o2}) 必须大于内钢管高度 h_i ({h_i})")

    # Case 1: 两抛物线不相交
    if (b_co + b_ci) / 4 <= delta_h / 2:
        Abi = (b_co ** 2) / 6.0 + (b_ci ** 2) / 6.0
        return Abi, 1

    # Case 2: 外钢管拱与内钢管宽边拱相交
    term_case2 = (b_co ** 2 - b_ci ** 2) / (4 * b_co)

    # 修正：当 r_i1=0 时，扩展 Case 2 条件以覆盖 Case 3 的间隙
    # 原始条件: term_case2 <= delta_h / 2
    # 当 r_i1=0 时，Case 3 无法满足，所以扩展为 term_case2 <= delta_h / 2 + r_i1
    case2_upper_limit = delta_h / 2 + r_i1 if r_i1 > 0 else float('inf')

    if (b_co + b_ci) / 4 > delta_h / 2 and term_case2 <= case2_upper_limit:
        numerator = b_co * b_ci * (b_co + b_ci - 2 * delta_h)
        denominator = 4 * (b_co + b_ci)

        if numerator < 0:
            Abi = (b_co ** 2) / 6.0 + (b_ci ** 2) / 6.0
            return Abi, 1

        x0 = np.sqrt(numerator / denominator)
        integral_part = 2 * (-x0 ** 3 / (3 * b_co) + x0 ** 2 / 2)
        rect_part = 2 * (b_co / 2 - x0) * (delta_h / 2)
        Abi = integral_part + rect_part
        return Abi, 2

    # Case 3: 外钢管拱与内钢管圆角相交
    if delta_h / 2 < term_case2 <= delta_h / 2 + r_i1:
        def equation_case3(x):
            left = -x ** 2 / b_co + b_co / 4
            radicand = r_i1 ** 2 - (x - b_ci / 2) ** 2
            if radicand < 0:
                return 1e6
            right = delta_h / 2 + r_i1 - np.sqrt(radicand)
            return left - right

        try:
            x1 = optimize.brentq(equation_case3, b_ci / 2, b_co / 2)
        except ValueError:
            x0 = np.sqrt(b_co * b_ci * (b_co + b_ci - 2 * delta_h) / (4 * (b_co + b_ci)))
            integral_part = 2 * (-x0 ** 3 / (3 * b_co) + x0**2 / 2)
            rect_part = 2 * (b_co / 2 - x0) * (delta_h / 2)
            Abi = integral_part + rect_part
            return Abi, 2

        part1 = b_ci * delta_h / 2

        def integrand_case3(x):
            return delta_h / 2 + r_i1 - np.sqrt(r_i1 ** 2 - (x - b_ci / 2) ** 2)

        integral2, _ = integrate.quad(integrand_case3, b_ci / 2, x1)
        part2 = 2 * integral2

        def integrand_case3_parabola(x):
            return -x ** 2 / b_co + b_co / 4

        integral3, _ = integrate.quad(integrand_case3_parabola, x1, b_co / 2)
        part3 = 2 * integral3

        Abi = part1 + part2 + part3
        return Abi, 3

    # 以下情况需要 h_ci 参数
    if h_ci is None:
        raise ValueError("当前几何条件属于Case 4或5，需要提供参数 h_ci (内钢管平直边深度)")

    # Case 4: 拱与高边拱相交
    term_case4 = (b_co ** 2 - b_i ** 2) / (4 * b_co)
    condition_4a = (delta_h / 2 + r_i1) < term_case4
    condition_4b = term_case4 > h_o2 / 2
    condition_4c = (2 * b_i + h_ci) ** 2 / 4 >= b_co * (b_co - 2 * h_o2)

    if condition_4a and condition_4b and condition_4c:
        def find_intersection_case4(x):
            y1 = -x ** 2 / b_co + b_co / 4
            if x > (b_i / 2 + h_ci / 2):
                return 1e6
            y2 = delta_h / 2 + np.sqrt(max(0, h_ci * ((b_i + h_ci / 2) / 2 - x)))
            return y1 - y2

        try:
            x2 = optimize.brentq(find_intersection_case4, b_i / 2, min(b_co / 2, b_i / 2 + h_ci / 2 + 0.1))
        except ValueError:
            x2 = b_co / 2

        part1 = b_i * (delta_h / 2 + r_i1) - b_ci * r_i1 - np.pi * r_i1 ** 2 / 2

        def integrand_case4_inner(x):
            term = ((h_ci + 2 * b_i) / 4 - x) * h_ci
            if term < 0:
                return h_o2 / 2
            return h_o2 / 2 - np.sqrt(term)

        integral4a, _ = integrate.quad(integrand_case4_inner, b_i / 2, x2)
        part2 = 2 * integral4a

        def integrand_case4_outer(x):
            return -x ** 2 / b_co + b_co / 4

        integral4b, _ = integrate.quad(integrand_case4_outer, x2, b_co / 2)
        part3 = 2 * integral4b

        Abi = part1 + part2 + part3
        return Abi, 4

    # Case 5: 拱与线 y=h_o2/2 相交且超出高边拱
    condition_5a = term_case4 > h_o2 / 2
    condition_5b = (2 * b_i + h_ci) ** 2 / 4 < b_co * (b_co - 2 * h_o2)

    if condition_5a and condition_5b:
        radicand_x3 = b_co * (b_co - 2 * h_o2) / 4

        if radicand_x3 < 0:
            x3 = b_co / 2
        else:
            x3 = np.sqrt(radicand_x3)

        if x3 <= b_i / 2:
            x3 = b_i / 2 + 0.01

        part1 = b_i * (delta_h / 2 + r_i1) - b_ci * r_i1 - np.pi * r_i1 ** 2 / 2
        part2 = (x3 - b_i / 2) * h_o2 - (h_ci ** 2) / 6

        def integrand_case5(x):
            return -x ** 2 / b_co + b_co / 4

        integral5, _ = integrate.quad(integrand_case5, x3, b_co / 2)
        part3 = 2 * integral5

        Abi = part1 + part2 + part3
        return Abi, 5

    raise ValueError("无法根据给定参数确定计算场景，请检查几何参数是否合理")


def process_csv(input_file, output_file):
    """处理CSV文件并计算Abi"""
    df = pd.read_csv(input_file)

    print(f"读取了 {len(df)} 行数据")
    print(f"列名: {df.columns.tolist()}")
    print("\n前5行数据:")
    print(df.head())

    results = []
    errors = []

    for idx, row in df.iterrows():
        try:
            b_co = row['b_co']
            h_o2 = row['h_02']
            b_ci = row['b_ci']
            h_i = row['h_i']
            r_i1 = row['r_i1']
            h_ci = row['h_ci'] if 'h_ci' in row else None
            b_i = row['b_i'] if 'b_i' in row else None

            if pd.isna(h_ci) or h_ci == 0:
                h_ci = None
            if pd.isna(b_i) or b_i == 0:
                b_i = None

            if h_o2 <= 0:
                error_msg = f"行 {idx+1}: h_o2 必须大于0, 当前值={h_o2}"
                errors.append({'row': idx + 1, 'error': error_msg})
                results.append({'Abi': None, 'case': None, 'error': error_msg})
                continue

            Abi, case = calculate_Abi(b_co, h_o2, b_ci, h_i, r_i1, h_ci, b_i)

            results.append({'Abi': Abi, 'case': case, 'error': None})

        except Exception as e:
            error_msg = f"行 {idx+1}: {str(e)}"
            errors.append({'row': idx + 1, 'error': str(e)})
            results.append({'Abi': None, 'case': None, 'error': str(e)})

    df['Abi'] = [r['Abi'] for r in results]
    df['case'] = [r['case'] for r in results]
    df['error'] = [r['error'] for r in results]

    df.to_csv(output_file, index=False)

    print(f"\n处理完成！")
    print(f"总行数: {len(df)}")
    print(f"成功计算: {len([r for r in results if r['Abi'] is not None])}")
    print(f"计算失败: {len(errors)}")

    if errors:
        print(f"\n错误详情:")
        for e in errors:
            print(f"  行 {e['row']}: {e['error']}")

    print(f"\nCase分布:")
    print(df['case'].value_counts().sort_index())

    print(f"\nAbi统计:")
    print(df['Abi'].describe())

    return df


if __name__ == "__main__":
    input_file = r"D:\AI_study\shujukuzhuanhua\caculate_Ahi.csv"
    output_file = r"D:\AI_study\shujukuzhuanhua\caculate_Ahi_results.csv"

    df = process_csv(input_file, output_file)
    print(f"\n结果已保存到: {output_file}")
