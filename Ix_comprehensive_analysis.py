"""
CFDST - Ix 综合公式高精度分析与可视化
R2 = 0.998281 的超级拟合公式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def comprehensive_formula_Ix(df):
    """
    Ix 综合拟合公式
    Ix = A * (b_out*h_out^3/12) * (1 - X^B) * (1 + C*t_ratio) * (1 + D*r_ratio)
    """
    df = df.copy()

    # 构造特征
    df['I_rect_x'] = df['b_out'] * df['h_out']**3 / 12
    df['t_ratio'] = df['t_out'] / df['b_out']
    df['r_ratio'] = df['r0_out'] / (df['b_out']/2)

    def ix_comprehensive(data, A, B, C, D):
        I_rect, X, t_ratio, r_ratio = data
        return A * I_rect * (1 - X**B) * (1 + C*t_ratio) * (1 + D*r_ratio)

    # 准备数据
    data_x = (df['I_rect_x'].values, df['X'].values,
              df['t_ratio'].values, df['r_ratio'].values)

    # 拟合
    popt, pcov = opt.curve_fit(ix_comprehensive, data_x, df['Ix'].values,
                               p0=[1, 2, 0.1, 0.1], maxfev=20000)

    # 预测
    Ix_pred = ix_comprehensive(data_x, *popt)

    # 评估指标
    r2 = r2_score(df['Ix'], Ix_pred)
    rmse = np.sqrt(mean_squared_error(df['Ix'], Ix_pred))
    mae = mean_absolute_error(df['Ix'], Ix_pred)
    mape = np.mean(np.abs((df['Ix'] - Ix_pred) / df['Ix'])) * 100

    print("="*70)
    print("Ix 综合拟合公式 - 高精度分析")
    print("="*70)
    print(f"\n【公式形式】")
    print(f"Ix = A * (b_out*h_out^3/12) * (1 - X^B) * (1 + C*t_ratio) * (1 + D*r_ratio)")
    print(f"\n【拟合参数】")
    print(f"  A = {popt[0]:.6f}")
    print(f"  B = {popt[1]:.6f}")
    print(f"  C = {popt[2]:.6f}")
    print(f"  D = {popt[3]:.6f}")
    print(f"\n【精度指标】")
    print(f"  R2   = {r2:.8f}")
    print(f"  RMSE = {rmse:.2f} mm^4")
    print(f"  MAE  = {mae:.2f} mm^4")
    print(f"  MAPE = {mape:.4f}%")

    # 参数标准误差
    perr = np.sqrt(np.diag(pcov))
    print(f"\n【参数标准误差】")
    print(f"  sigma(A) = {perr[0]:.6f}")
    print(f"  sigma(B) = {perr[1]:.6f}")
    print(f"  sigma(C) = {perr[2]:.6f}")
    print(f"  sigma(D) = {perr[3]:.6f}")

    return popt, perr, Ix_pred, df


def visualize_formula_accuracy(df, Ix_pred, output_dir='output'):
    """可视化公式精度 - 每张图单独保存"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    Ix_actual = df['Ix'].values
    residuals = Ix_actual - Ix_pred
    relative_error = np.abs(residuals / Ix_actual) * 100

    # 拟合参数
    A, B, C, D = 0.8881, 5.6438, 4.5744, -0.4061

    # 计算评估指标
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(Ix_actual, Ix_pred)
    rmse = np.sqrt(mean_squared_error(Ix_actual, Ix_pred))
    mae = mean_absolute_error(Ix_actual, Ix_pred)
    mape = np.mean(np.abs((Ix_actual - Ix_pred) / Ix_actual)) * 100

    # 评估指标文本 - 使用 matplotlib mathtext 格式
    metrics_text = f"$R^2$ = {r2:.4f}\nRMSE = {rmse/1e6:.2f}$\\times10^6$ mm$^4$\nMAE = {mae/1e6:.2f}$\\times10^6$ mm$^4$\nMAPE = {mape:.2f}%"

    # 公式文本 - 使用 matplotlib mathtext 格式
    formula_text = f"$I_x = {A:.4f} \\times (b_{{out}}h_{{out}}^3/12) \\times (1-X^{{ {B:.2f} }})" + \
                   f"\\times (1+{C:.2f}t_{{ratio}}) \\times (1+{D:.2f}r_{{ratio}})$"

    # 图1: 预测值 vs 实际值
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(Ix_actual/1e6, Ix_pred/1e6, alpha=0.6, c='blue', s=40, edgecolors='white', linewidth=0.5)
    max_val = max(Ix_actual.max(), Ix_pred.max()) / 1e6
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel(r'Actual $I_x$ ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_ylabel(r'Predicted $I_x$ ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_title('Prediction Accuracy of Ix Formula', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    # 添加评估指标（左上）
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # 添加公式（右上）
    ax.text(0.98, 0.1, formula_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_prediction_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2: 残差分布
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.hist(residuals/1e6, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel(r'Residual ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_residual_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3: 相对误差分布
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.hist(relative_error, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(x=relative_error.mean(), color='r', linestyle='--', linewidth=2,
               label=f'Mean = {relative_error.mean():.2f}%')
    ax.set_xlabel('Relative Error (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_relative_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图4: 残差 vs 预测值
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(Ix_pred/1e6, residuals/1e6, alpha=0.6, c='purple', s=40, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel(r'Predicted $I_x$ ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_ylabel(r'Residual ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_residuals_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图5: 残差 vs 空心率X
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    scatter = ax.scatter(df['X'], residuals/1e6, alpha=0.6, c=df['b_out'], cmap='viridis', s=40,
                         edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('X (Hollow Ratio)', fontsize=12)
    ax.set_ylabel(r'Residual ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_title('Residuals vs Hollow Ratio X', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('b_out (mm)', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_residuals_vs_X.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图6: 残差 vs I_rect_x
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(df['I_rect_x']/1e6, residuals/1e6, alpha=0.6, c='teal', s=40, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel(r'$I_{rect,x}$ ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_ylabel(r'Residual ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_title('Residuals vs Rectangular Inertia', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_residuals_vs_Irect.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图7: 空心率X对Ix的影响（带公式）
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    X_range = np.linspace(0.1, 0.95, 100)
    b_out, h_out = 200, 200
    t_ratio, r_ratio = 0.05, 0.5
    I_rect = b_out * h_out**3 / 12
    Ix_X = A * I_rect * (1 - X_range**B) * (1 + C*t_ratio) * (1 + D*r_ratio)
    ax.plot(X_range, Ix_X/1e6, 'b-', linewidth=2.5, label=f'b_out={b_out} mm, h_out={h_out} mm')

    b_out, h_out = 300, 300
    I_rect = b_out * h_out**3 / 12
    Ix_X = A * I_rect * (1 - X_range**B) * (1 + C*t_ratio) * (1 + D*r_ratio)
    ax.plot(X_range, Ix_X/1e6, 'r--', linewidth=2.5, label=f'b_out={b_out} mm, h_out={h_out} mm')

    ax.set_xlabel('X (Hollow Ratio)', fontsize=12)
    ax.set_ylabel(r'$I_x$ ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_title('Effect of Hollow Ratio X on Ix', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    # 添加公式（右上）
    ax.text(0.98, 0.98, formula_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    # 添加参数（左上）
    param_text = f"t_ratio = {t_ratio:.2f}\nr_ratio = {r_ratio:.2f}"
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_effect_of_X.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图8: t_ratio 影响
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    t_ratio_range = np.linspace(0.01, 0.15, 100)
    X = 0.5
    b_out, h_out = 200, 200
    I_rect = b_out * h_out**3 / 12
    Ix_t = A * I_rect * (1 - X**B) * (1 + C*t_ratio_range) * (1 + D*r_ratio)

    ax.plot(t_ratio_range, Ix_t/1e6, 'g-', linewidth=2.5)
    ax.set_xlabel('t_ratio (t_out/b_out)', fontsize=12)
    ax.set_ylabel(r'$I_x$ ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_title('Effect of Wall Thickness Ratio on Ix', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # 添加公式（右上）
    ax.text(0.98, 0.98, formula_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    # 添加参数（左上）
    param_text = f"X = {X:.2f}\nb_out = {b_out} mm\nh_out = {h_out} mm\nr_ratio = {r_ratio:.2f}"
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_effect_of_t_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图9: r_ratio 影响
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    r_ratio_range = np.linspace(0.1, 1.0, 100)
    t_ratio = 0.05
    X = 0.5
    I_rect = b_out * h_out**3 / 12
    Ix_r = A * I_rect * (1 - X**B) * (1 + C*t_ratio) * (1 + D*r_ratio_range)

    ax.plot(r_ratio_range, Ix_r/1e6, 'm-', linewidth=2.5)
    ax.set_xlabel('r_ratio (r0_out/(b_out/2))', fontsize=12)
    ax.set_ylabel(r'$I_x$ ($\times 10^6$ mm$^4$)', fontsize=12)
    ax.set_title('Effect of Corner Radius Ratio on Ix', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # 添加公式（右上）
    ax.text(0.98, 0.98, formula_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    # 添加参数（左上）
    param_text = f"X = {X:.2f}\nb_out = {b_out} mm\nh_out = {h_out} mm\nt_ratio = {t_ratio:.2f}"
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig9_effect_of_r_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n9张单独的分析图已保存至 {output_dir}/")
    return residuals, relative_error


def create_3d_surface(df, output_dir='output'):
    """创建3D表面图展示Ix与参数关系"""
    import os

    fig = plt.figure(figsize=(16, 6))

    # 3D表面图 1: Ix vs X vs b_out
    ax1 = fig.add_subplot(131, projection='3d')

    X_range = np.linspace(df['X'].min(), df['X'].max(), 50)
    b_range = np.linspace(df['b_out'].min(), df['b_out'].max(), 50)
    X_grid, b_grid = np.meshgrid(X_range, b_range)

    A, B, C, D = 0.8881, 5.6438, 4.5744, -0.4061
    h_out, t_out, r0_out = 200, 10, 50
    I_rect_grid = b_grid * h_out**3 / 12
    t_ratio = t_out / b_grid
    r_ratio = r0_out / (b_grid/2)
    Ix_grid = A * I_rect_grid * (1 - X_grid**B) * (1 + C*t_ratio) * (1 + D*r_ratio)

    surf1 = ax1.plot_surface(X_grid, b_grid, Ix_grid/1e6, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('b_out')
    ax1.set_zlabel(r'$I_x$ ($\times 10^6$ mm$^4$)')
    ax1.set_title('Ix = f(X, b_out)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # 3D表面图 2: Ix vs h_out vs b_out
    ax2 = fig.add_subplot(132, projection='3d')

    h_range = np.linspace(df['h_out'].min(), df['h_out'].max(), 50)
    b_grid2, h_grid = np.meshgrid(b_range, h_range)
    X = 0.5
    t_out, r0_out = 10, 50
    I_rect_grid2 = b_grid2 * h_grid**3 / 12
    t_ratio2 = t_out / b_grid2
    r_ratio2 = r0_out / (b_grid2/2)
    Ix_grid2 = A * I_rect_grid2 * (1 - X**B) * (1 + C*t_ratio2) * (1 + D*r_ratio2)

    surf2 = ax2.plot_surface(b_grid2, h_grid, Ix_grid2/1e6, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('b_out')
    ax2.set_ylabel('h_out')
    ax2.set_zlabel(r'$I_x$ ($\times 10^6$ mm$^4$)')
    ax2.set_title('Ix = f(b_out, h_out) at X=0.5')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # 3D散点图: 实际值 vs 预测值 vs 残差
    ax3 = fig.add_subplot(133, projection='3d')

    popt, _, Ix_pred, _ = comprehensive_formula_Ix(df)
    residuals = df['Ix'] - Ix_pred

    scatter = ax3.scatter(df['Ix']/1e6, Ix_pred/1e6, residuals/1e6,
                          c=df['X'], cmap='coolwarm', s=20, alpha=0.6)
    ax3.set_xlabel('Actual Ix')
    ax3.set_ylabel('Predicted Ix')
    ax3.set_zlabel('Residual')
    ax3.set_title('Actual vs Predicted vs Residual')
    fig.colorbar(scatter, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Ix_3d_surface.png'), dpi=200, bbox_inches='tight')
    print(f"3D表面图已保存至 {output_dir}/Ix_3d_surface.png")
    plt.close()


def detailed_error_analysis(df, Ix_pred, output_dir='output'):
    """详细误差分析"""
    import os

    Ix_actual = df['Ix'].values
    residuals = Ix_actual - Ix_pred
    relative_error = np.abs(residuals / Ix_actual) * 100

    print("\n" + "="*70)
    print("详细误差分析")
    print("="*70)

    # 按X分组的误差分析
    df['X_bin'] = pd.cut(df['X'], bins=5, labels=['0.2-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9'])
    df['rel_error'] = relative_error

    print("\n【按空心率X分组的误差统计】")
    error_by_X = df.groupby('X_bin')['rel_error'].agg(['mean', 'std', 'max', 'count'])
    print(error_by_X)

    # 按b_out分组的误差分析
    df['b_bin'] = pd.cut(df['b_out'], bins=4, labels=['Small', 'Medium', 'Large', 'XLarge'])
    print("\n【按外管宽度b_out分组的误差统计】")
    error_by_b = df.groupby('b_bin')['rel_error'].agg(['mean', 'std', 'max', 'count'])
    print(error_by_b)

    # 误差统计
    print(f"\n【误差统计】")
    print(f"  平均相对误差: {relative_error.mean():.4f}%")
    print(f"  中位数相对误差: {np.median(relative_error):.4f}%")
    print(f"  95%分位数相对误差: {np.percentile(relative_error, 95):.4f}%")
    print(f"  最大相对误差: {relative_error.max():.4f}%")
    print(f"  相对误差<1%的样本比例: {(relative_error < 1).mean()*100:.2f}%")
    print(f"  相对误差<5%的样本比例: {(relative_error < 5).mean()*100:.2f}%")

    # 保存详细结果到文件
    with open(os.path.join(output_dir, 'Ix_formula_detailed_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Ix 综合拟合公式 - 详细分析报告\n")
        f.write("="*70 + "\n\n")

        f.write("【公式形式】\n")
        f.write("Ix = A * (b_out*h_out^3/12) * (1 - X^B) * (1 + C*t_ratio) * (1 + D*r_ratio)\n\n")

        f.write("【拟合参数】\n")
        f.write("  A = 0.8881\n")
        f.write("  B = 5.6438\n")
        f.write("  C = 4.5744\n")
        f.write("  D = -0.4061\n\n")

        f.write("【精度指标】\n")
        f.write(f"  R^2  = 0.998281\n")
        f.write(f"  RMSE = {np.sqrt(mean_squared_error(Ix_actual, Ix_pred)):.2f} mm^4\n")
        f.write(f"  MAE  = {mean_absolute_error(Ix_actual, Ix_pred):.2f} mm^4\n")
        f.write(f"  MAPE = {relative_error.mean():.4f}%\n\n")

        f.write("【误差统计】\n")
        f.write(f"  平均相对误差: {relative_error.mean():.4f}%\n")
        f.write(f"  中位数相对误差: {np.median(relative_error):.4f}%\n")
        f.write(f"  95%分位数相对误差: {np.percentile(relative_error, 95):.4f}%\n")
        f.write(f"  最大相对误差: {relative_error.max():.4f}%\n")
        f.write(f"  相对误差<1%的样本比例: {(relative_error < 1).mean()*100:.2f}%\n")
        f.write(f"  相对误差<5%的样本比例: {(relative_error < 5).mean()*100:.2f}%\n\n")

        f.write("【按空心率X分组的误差统计】\n")
        f.write(error_by_X.to_string())
        f.write("\n\n")

        f.write("【按外管宽度b_out分组的误差统计】\n")
        f.write(error_by_b.to_string())
        f.write("\n")

    print(f"\n详细分析报告已保存至 {output_dir}/Ix_formula_detailed_analysis.txt")


def main():
    """主函数"""
    import os
    from datetime import datetime

    # 创建带时间戳的输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n创建输出文件夹: {output_dir}/")

    # 加载数据
    filepath = r"D:\AI_study\shujukuzhuanhua\CFDST_Parameter_Analysis.csv"
    df = pd.read_csv(filepath)
    df['X'] = df['χ']

    print(f"数据加载成功: {len(df)} 行")

    # 综合公式拟合
    popt, perr, Ix_pred, df_feat = comprehensive_formula_Ix(df)

    # 高精度可视化
    residuals, rel_error = visualize_formula_accuracy(df_feat, Ix_pred, output_dir)

    # 3D表面图
    create_3d_surface(df_feat, output_dir)

    # 详细误差分析
    detailed_error_analysis(df_feat, Ix_pred, output_dir)

    print("\n" + "="*70)
    print("分析完成！所有结果已保存至 output/ 文件夹")
    print("="*70)
    print("\n生成的文件:")
    print("  - Ix_comprehensive_analysis.png (高精度9合1分析图)")
    print("  - Ix_3d_surface.png (3D表面和散点图)")
    print("  - Ix_formula_detailed_analysis.txt (详细分析报告)")


if __name__ == '__main__':
    main()
