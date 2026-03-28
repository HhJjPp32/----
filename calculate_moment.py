import math
import csv
import os

def calc_rounded_rect_I(b: float, h: float, r: float):
    """
    计算圆角矩形对形心轴 x 和 y 的惯性矩。
    采用组合法：中间矩形 + 上下条状矩形 + 四个 1/4 圆。
    参数：
        b : 矩形宽度（水平方向尺寸）
        h : 矩形高度（竖直方向尺寸）
        r : 圆角半径
    返回：
        (Ix, Iy) : 分别对 x 轴和 y 轴的惯性矩
    """
    if r == 0:
        # 退化到普通矩形
        Ix = b * h**3 / 12.0
        Iy = h * b**3 / 12.0
        return Ix, Iy

    # ---------- 中间矩形 ----------
    Ix1 = (1.0 / 12.0) * b * (h - 2 * r) ** 3

    # ---------- 上下条状矩形 ----------
    # 单条面积
    A2_strip = (b - 2 * r) * r
    # 单条对自身形心轴的惯性矩
    Ix2c = (1.0 / 12.0) * (b - 2 * r) * r ** 3
    # 移轴距离（形心到整体 x 轴的距离）
    d2 = (h - r) / 2.0
    # 单个条状对整体 x 轴的惯性矩
    Ix2_single = Ix2c + A2_strip * d2 ** 2
    # 上下两条之和
    Ix2 = 2.0 * Ix2_single

    # ---------- 四个 1/4 圆 ----------
    # 单个 1/4 圆面积
    A3_q = (math.pi / 4.0) * r ** 2
    # 1/4 圆对自身形心轴的惯性矩（平行于 x 轴）
    Ix3c = (math.pi / 16.0 - 4.0 / (9.0 * math.pi)) * r ** 4
    # 移轴距离（形心到整体 x 轴的距离）
    d3 = (4.0 * r) / (3.0 * math.pi) + (h - 2.0 * r) / 2.0
    # 单个 1/4 圆对整体 x 轴的惯性矩
    Ix3_single = Ix3c + A3_q * d3 ** 2
    # 四个 1/4 圆之和
    Ix3 = 4.0 * Ix3_single

    # 总 Ix
    Ix = Ix1 + Ix2 + Ix3

    # ---------- y 方向（对称） ----------
    # 中间矩形（旋转方向）
    Iy1 = (1.0 / 12.0) * h * (b - 2 * r) ** 3

    # 左右条状矩形（对应 y 方向）
    A2_strip_y = (h - 2 * r) * r
    Iy2c = (1.0 / 12.0) * (h - 2 * r) * r ** 3
    d2_y = (b - r) / 2.0
    Iy2_single = Iy2c + A2_strip_y * d2_y ** 2
    Iy2 = 2.0 * Iy2_single

    # 四个 1/4 圆对 y 轴
    Iy3c = (math.pi / 16.0 - 4.0 / (9.0 * math.pi)) * r ** 4
    d3_y = (4.0 * r) / (3.0 * math.pi) + (b - 2.0 * r) / 2.0
    Iy3_single = Iy3c + A3_q * d3_y ** 2
    Iy3 = 4.0 * Iy3_single

    Iy = Iy1 + Iy2 + Iy3

    return Ix, Iy


def calc_CFDST_I(b_out, h_out, t_out, r0_out,
                 b_in,  h_in,  t_in,  r0_in):
    """
    计算中空夹层钢管混凝土 (CFDST) 截面各组成部分的惯性矩。
    参数：
        b_out : 外钢管外轮廓宽度
        h_out : 外钢管外轮廓高度
        t_out : 外钢管壁厚
        r0_out: 外钢管外圆角半径
        b_in  : 内钢管外轮廓宽度
        h_in  : 内钢管外轮廓高度
        t_in  : 内钢管壁厚
        r0_in : 内钢管外圆角半径
    返回：
        字典，包含以下键值对：
            'CFDST'   : (Ix, Iy) 整体截面惯性矩
            'out_steel': (Ix, Iy) 外钢管自身惯性矩
            'concrete' : (Ix, Iy) 混凝土部分惯性矩
            'in_steel' : (Ix, Iy) 内钢管自身惯性矩
    """
    # 情况①：外钢管外轮廓
    Ix1, Iy1 = calc_rounded_rect_I(b_out, h_out, r0_out)

    # 情况②：外钢管内轮廓
    b2 = b_out - 2 * t_out
    h2 = h_out - 2 * t_out
    r2 = max(0.0, r0_out - t_out)
    Ix2, Iy2 = calc_rounded_rect_I(b2, h2, r2)

    # 情况③：内钢管外轮廓
    Ix3, Iy3 = calc_rounded_rect_I(b_in, h_in, r0_in)

    # 情况④：内钢管内轮廓
    b4 = b_in - 2 * t_in
    h4 = h_in - 2 * t_in
    r4 = max(0.0, r0_in - t_in)
    Ix4, Iy4 = calc_rounded_rect_I(b4, h4, r4)

    # 组合结果
    results = {
        'CFDST': (Ix1 - Ix4, Iy1 - Iy4),           # 整体截面 = 外轮廓 - 内管内孔
        'out_steel': (Ix1 - Ix2, Iy1 - Iy2),       # 外钢管 = 外轮廓 - 内轮廓
        'concrete': (Ix2 - Ix3, Iy2 - Iy3),        # 混凝土 = 外管内壁 - 内管外壁
        'in_steel': (Ix3 - Ix4, Iy3 - Iy4)         # 内钢管 = 内管外轮廓 - 内管内轮廓
    }
    return results


def process_csv(input_csv_path, output_csv_path=None):
    """
    读取 CSV 文件并批量计算 CFDST 截面惯性矩。

    CSV 文件应包含以下列：
    b_out, h_out, t_out, r0_out, r1_out, b_in, h_in, t_in, r0_in, r1_in

    参数：
        input_csv_path: 输入 CSV 文件路径
        output_csv_path: 输出 CSV 文件路径（默认为 input_csv_path + '_results.csv'）
    """
    if output_csv_path is None:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = f"{base}_results{ext}"

    results = []

    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 读取表头

        for row_num, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == '' for cell in row):
                continue

            try:
                # 解析 CSV 列：b_out, h_out, t_out, r0_out, r1_out, b_in, h_in, t_in, r0_in, r1_in
                b_out = float(row[0].strip())
                h_out = float(row[1].strip())
                t_out = float(row[2].strip())
                r0_out = float(row[3].strip()) if row[3].strip() else 0.0
                # r1_out = float(row[4].strip()) if row[4].strip() else 0.0  # 内圆角半径，计算中不需要
                b_in = float(row[5].strip())
                h_in = float(row[6].strip())
                t_in = float(row[7].strip())
                r0_in = float(row[8].strip()) if row[8].strip() else 0.0
                # r1_in = float(row[9].strip()) if row[9].strip() else 0.0  # 内圆角半径，计算中不需要

                # 计算惯性矩
                inertias = calc_CFDST_I(b_out, h_out, t_out, r0_out,
                                        b_in, h_in, t_in, r0_in)

                # 整理结果
                result = {
                    'row_num': row_num,
                    'b_out': b_out,
                    'h_out': h_out,
                    't_out': t_out,
                    'r0_out': r0_out,
                    'b_in': b_in,
                    'h_in': h_in,
                    't_in': t_in,
                    'r0_in': r0_in,
                    'CFDST_Ix': inertias['CFDST'][0],
                    'CFDST_Iy': inertias['CFDST'][1],
                    'out_steel_Ix': inertias['out_steel'][0],
                    'out_steel_Iy': inertias['out_steel'][1],
                    'concrete_Ix': inertias['concrete'][0],
                    'concrete_Iy': inertias['concrete'][1],
                    'in_steel_Ix': inertias['in_steel'][0],
                    'in_steel_Iy': inertias['in_steel'][1],
                }
                results.append(result)

            except (ValueError, IndexError) as e:
                print(f"警告：第 {row_num} 行数据解析错误: {e}")
                continue

    # 写入结果 CSV
    fieldnames = ['row_num', 'b_out', 'h_out', 't_out', 'r0_out',
                  'b_in', 'h_in', 't_in', 'r0_in',
                  'CFDST_Ix', 'CFDST_Iy',
                  'out_steel_Ix', 'out_steel_Iy',
                  'concrete_Ix', 'concrete_Iy',
                  'in_steel_Ix', 'in_steel_Iy']

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"计算完成！共处理 {len(results)} 行数据")
    print(f"结果已保存至: {output_csv_path}")
    return results


# ========== 使用示例 ==========
if __name__ == '__main__':
    # 示例 1：单个截面计算
    print("=" * 60)
    print("示例 1：单个截面计算")
    print("=" * 60)
    b_out, h_out, t_out, r0_out = 200.0, 300.0, 10.0, 15.0
    b_in,  h_in,  t_in,  r0_in  = 100.0, 200.0, 8.0,  10.0

    inertias = calc_CFDST_I(b_out, h_out, t_out, r0_out,
                            b_in,  h_in,  t_in,  r0_in)

    for name, (Ix, Iy) in inertias.items():
        print(f"{name:12} Ix = {Ix:.3f} mm^4, Iy = {Iy:.3f} mm^4")

    # 示例 2：批量处理 CSV 文件
    print("\n" + "=" * 60)
    print("示例 2：批量处理 CSV 文件")
    print("=" * 60)

    csv_path = r"D:\AI_study\shujukuzhuanhua\caculate_moment.csv"
    if os.path.exists(csv_path):
        process_csv(csv_path)
    else:
        print(f"CSV 文件不存在: {csv_path}")
        print("请确认文件路径正确")
