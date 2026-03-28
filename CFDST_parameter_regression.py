"""
CFDST参数分析与回归拟合
分析空心率X、ke、Ix、Iy与几何参数的关系
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(filepath):
    """加载数据"""
    # 指定列名（新文件无表头）
    column_names = ['b_out', 'h_out', 't_out', 'r0_out', 'b_in', 'h_in', 't_in', 'r0_in', 'χ', 'ke', 'Ix', 'Iy']
    df = pd.read_csv(filepath, header=None, names=column_names)
    print(f"数据加载成功，共 {len(df)} 行")
    print(f"列名: {list(df.columns)}")
    print("\n数据预览:")
    print(df.head())
    print("\n数据统计:")
    print(df.describe())
    return df


def feature_engineering(df):
    """特征工程：构造新的特征，比值统一为 外/内"""
    df = df.copy()

    # 基本几何特征 - 外轮廓尺寸比
    df['aspect_ratio'] = df['h_out'] / df['b_out']  # 高宽比

    # 壁厚比 (外/内)
    df['t_ratio'] = df['t_out'] / df['t_in']

    # 内外管尺寸比 (外/内)
    df['b_ratio'] = df['b_out'] / df['b_in']
    df['h_ratio'] = df['h_out'] / df['h_in']

    # 圆角半径比 (外/内)
    df['r_ratio'] = df['r0_out'] / df['r0_in']

    # 外管相对厚度 (相对于自身宽度)
    df['t_ratio_out'] = df['t_out'] / df['b_out']
    df['t_ratio_in'] = df['t_in'] / df['b_in']

    # 外管圆角率 (相对于自身半宽)
    df['r_ratio_out'] = df['r0_out'] / (df['b_out']/2)
    df['r_ratio_in'] = df['r0_in'] / (df['b_in']/2)

    # 空心率相关
    df['X'] = df['χ']  # 空心率
    df['X_squared'] = df['X'] ** 2
    df['X_cubed'] = df['X'] ** 3

    # 面积特征（估算）
    df['A_out'] = df['b_out'] * df['h_out']
    df['A_in'] = df['b_in'] * df['h_in']
    df['A_ratio'] = df['A_out'] / df['A_in']  # 面积比 (外/内)

    # 惯性矩的特征构造
    # 参考矩形惯性矩公式 I = b*h^3/12
    df['I_rect_x'] = df['b_out'] * df['h_out']**3 / 12
    df['I_rect_y'] = df['h_out'] * df['b_out']**3 / 12
    df['I_ratio_x'] = df['Ix'] / df['I_rect_x']
    df['I_ratio_y'] = df['Iy'] / df['I_rect_y']

    return df


def correlation_analysis(df, output_dir='.'):
    """相关性分析"""
    import os
    print("\n" + "="*60)
    print("相关性分析")
    print("="*60)

    # 选择与目标变量相关的列
    cols = ['b_out', 'h_out', 't_out', 'r0_out', 'b_in', 'h_in', 't_in', 'r0_in',
            'X', 'ke', 'Ix', 'Iy']
    corr_matrix = df[cols].corr()

    print("\n与 Ix 的相关性:")
    print(corr_matrix['Ix'].sort_values(ascending=False))

    print("\n与 Iy 的相关性:")
    print(corr_matrix['Iy'].sort_values(ascending=False))

    print("\n与 X（空心率）的相关性:")
    print(corr_matrix['X'].sort_values(ascending=False))

    print("\n与 ke 的相关性:")
    print(corr_matrix['ke'].sort_values(ascending=False))

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Ix vs 各参数
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['X'], df['Ix']/1e6, c=df['b_out'], cmap='viridis', alpha=0.6)
    ax1.set_xlabel('空心率 X')
    ax1.set_ylabel('Ix (x10^6 mm^4)')
    ax1.set_title('Ix vs 空心率')
    plt.colorbar(scatter, ax=ax1, label='b_out')

    # Iy vs 各参数
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['X'], df['Iy']/1e6, c=df['h_out'], cmap='plasma', alpha=0.6)
    ax2.set_xlabel('空心率 X')
    ax2.set_ylabel('Iy (x10^6 mm^4)')
    ax2.set_title('Iy vs 空心率')
    plt.colorbar(scatter, ax=ax2, label='h_out')

    # Ix vs I_rect_x
    ax3 = axes[0, 2]
    ax3.scatter(df['I_rect_x']/1e6, df['Ix']/1e6, alpha=0.6)
    ax3.plot([0, df['I_rect_x'].max()/1e6], [0, df['I_rect_x'].max()/1e6], 'r--', label='y=x')
    ax3.set_xlabel('矩形惯性矩 I_rect_x (x10^6 mm^4)')
    ax3.set_ylabel('实际 Ix (x10^6 mm^4)')
    ax3.set_title('实际Ix vs 矩形惯性矩')
    ax3.legend()

    # I_ratio分布
    ax4 = axes[1, 0]
    ax4.hist(df['I_ratio_x'], bins=30, alpha=0.7, label='I_ratio_x')
    ax4.hist(df['I_ratio_y'], bins=30, alpha=0.7, label='I_ratio_y')
    ax4.set_xlabel('惯性矩比值')
    ax4.set_ylabel('频数')
    ax4.set_title('惯性矩比值分布')
    ax4.legend()

    # ke vs X
    ax5 = axes[1, 1]
    ax5.scatter(df['X'], df['ke'], alpha=0.6, c=df['t_out'], cmap='coolwarm')
    ax5.set_xlabel('空心率 X')
    ax5.set_ylabel('ke')
    ax5.set_title('ke vs X')

    # ke vs t_out
    ax6 = axes[1, 2]
    ax6.scatter(df['t_out'], df['ke'], alpha=0.6, c=df['X'], cmap='viridis')
    ax6.set_xlabel('t_out (mm)')
    ax6.set_ylabel('ke')
    ax6.set_title('ke vs t_out')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"\n相关性分析图已保存至 {output_dir}/correlation_analysis.png")
    plt.close()

    return corr_matrix


def linear_regression_analysis(df):
    """线性回归分析"""
    print("\n" + "="*60)
    print("线性回归分析")
    print("="*60)

    # 特征选择
    feature_cols = ['b_out', 'h_out', 't_out', 'r0_out', 'b_in', 'h_in', 't_in', 'r0_in', 'X']
    X = df[feature_cols]
    y_x = df['Ix']
    y_y = df['Iy']

    # 划分训练集和测试集
    X_train, X_test, yx_train, yx_test = train_test_split(X, y_x, test_size=0.2, random_state=42)
    _, _, yy_train, yy_test = train_test_split(X, y_y, test_size=0.2, random_state=42)

    # 线性回归
    model_x = LinearRegression()
    model_y = LinearRegression()

    model_x.fit(X_train, yx_train)
    model_y.fit(X_train, yy_train)

    # 预测
    yx_pred = model_x.predict(X_test)
    yy_pred = model_y.predict(X_test)

    # 评估
    print("\nIx 线性回归结果:")
    print(f"  R2 = {r2_score(yx_test, yx_pred):.6f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(yx_test, yx_pred)):.2f}")
    print("\n  回归系数:")
    for feat, coef in zip(feature_cols, model_x.coef_):
        print(f"    {feat}: {coef:.6f}")
    print(f"  截距: {model_x.intercept_:.2f}")

    print("\nIy 线性回归结果:")
    print(f"  R2 = {r2_score(yy_test, yy_pred):.6f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(yy_test, yy_pred)):.2f}")
    print("\n  回归系数:")
    for feat, coef in zip(feature_cols, model_y.coef_):
        print(f"    {feat}: {coef:.6f}")
    print(f"  截距: {model_y.intercept_:.2f}")

    return model_x, model_y


def polynomial_regression(df, degree=2):
    """多项式回归"""
    print(f"\n" + "="*60)
    print(f"多项式回归 (degree={degree})")
    print("="*60)

    # 选择关键特征进行多项式拟合
    feature_cols = ['b_out', 'h_out', 't_out', 'b_in', 'h_in', 't_in', 'X']
    X = df[feature_cols]
    y_x = df['Ix']
    y_y = df['Iy']

    # 多项式回归管道
    poly_model_x = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])

    poly_model_y = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])

    # 训练
    X_train, X_test, yx_train, yx_test = train_test_split(X, y_x, test_size=0.2, random_state=42)
    _, _, yy_train, yy_test = train_test_split(X, y_y, test_size=0.2, random_state=42)

    poly_model_x.fit(X_train, yx_train)
    poly_model_y.fit(X_train, yy_train)

    # 预测
    yx_pred = poly_model_x.predict(X_test)
    yy_pred = poly_model_y.predict(X_test)

    # 评估
    print(f"\nIx 多项式回归结果 (degree={degree}):")
    print(f"  R2 = {r2_score(yx_test, yx_pred):.6f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(yx_test, yx_pred)):.2f}")

    print(f"\nIy 多项式回归结果 (degree={degree}):")
    print(f"  R2 = {r2_score(yy_test, yy_pred):.6f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(yy_test, yy_pred)):.2f}")

    # 获取特征名
    feature_names = poly_model_x.named_steps['poly'].get_feature_names_out(feature_cols)

    # 输出前10个重要系数
    coefs_x = poly_model_x.named_steps['linear'].coef_
    coefs_y = poly_model_y.named_steps['linear'].coef_

    print("\nIx 前10个重要项:")
    important_idx = np.argsort(np.abs(coefs_x))[-10:][::-1]
    for idx in important_idx:
        print(f"  {feature_names[idx]}: {coefs_x[idx]:.6e}")

    print("\nIy 前10个重要项:")
    important_idx = np.argsort(np.abs(coefs_y))[-10:][::-1]
    for idx in important_idx:
        print(f"  {feature_names[idx]}: {coefs_y[idx]:.6e}")

    return poly_model_x, poly_model_y, feature_names


def physics_based_formula(df, output_dir='.'):
    """基于物理意义的公式拟合"""
    import os
    print("\n" + "="*60)
    print("基于物理意义的公式拟合")
    print("="*60)

    # 假设 Ix = A * b_out * h_out^3 * f(X) / 12
    # 其中 f(X) 是关于空心率的函数

    df = df.copy()
    df['I_rect_x'] = df['b_out'] * df['h_out']**3 / 12
    df['I_rect_y'] = df['h_out'] * df['b_out']**3 / 12

    # 计算修正系数
    df['kx'] = df['Ix'] / df['I_rect_x']
    df['ky'] = df['Iy'] / df['I_rect_y']

    # 拟合 kx = f(X) = a * (1 - X^n)
    def kx_func(X, a, n):
        return a * (1 - X**n)

    def kx_func_poly(X, a, b, c):
        return a + b*X + c*X**2

    # 拟合
    try:
        popt_x, _ = opt.curve_fit(kx_func_poly, df['X'], df['kx'])
        popt_y, _ = opt.curve_fit(kx_func_poly, df['X'], df['ky'])

        print("\n拟合公式 Ix = kx * b_out * h_out^3 / 12")
        print(f"其中 kx = {popt_x[0]:.4f} + {popt_x[1]:.4f}*X + {popt_x[2]:.4f}*X^2")

        print("\n拟合公式 Iy = ky * h_out * b_out^3 / 12")
        print(f"其中 ky = {popt_y[0]:.4f} + {popt_y[1]:.4f}*X + {popt_y[2]:.4f}*X^2")

        # 评估
        kx_pred = kx_func_poly(df['X'], *popt_x)
        ky_pred = kx_func_poly(df['X'], *popt_y)

        Ix_pred = kx_pred * df['I_rect_x']
        Iy_pred = ky_pred * df['I_rect_y']

        print(f"\nIx 公式 R2 = {r2_score(df['Ix'], Ix_pred):.6f}")
        print(f"Iy 公式 R2 = {r2_score(df['Iy'], Iy_pred):.6f}")

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        ax1.scatter(df['X'], df['kx'], alpha=0.5, label='实际值')
        X_sorted = np.sort(df['X'].unique())
        ax1.plot(X_sorted, kx_func_poly(X_sorted, *popt_x), 'r-', label='拟合曲线')
        ax1.set_xlabel('空心率 X')
        ax1.set_ylabel('kx')
        ax1.set_title('kx vs X')
        ax1.legend()

        ax2 = axes[1]
        ax2.scatter(df['X'], df['ky'], alpha=0.5, label='实际值')
        ax2.plot(X_sorted, kx_func_poly(X_sorted, *popt_y), 'r-', label='拟合曲线')
        ax2.set_xlabel('空心率 X')
        ax2.set_ylabel('ky')
        ax2.set_title('ky vs X')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'physics_formula.png'), dpi=150, bbox_inches='tight')
        print(f"\n物理公式拟合图已保存至 {output_dir}/physics_formula.png")
        plt.close()

    except Exception as e:
        print(f"拟合失败: {e}")


def comprehensive_formula(df):
    """综合拟合：考虑壁厚和圆角的影响"""
    print("\n" + "="*60)
    print("综合公式拟合")
    print("="*60)

    df = df.copy()

    # 构造特征
    df['h3'] = df['h_out']**3
    df['b3'] = df['b_out']**3
    df['X2'] = df['X']**2
    df['X3'] = df['X']**3
    # 使用已定义的特征: t_ratio (外/内), t_ratio_out (外管相对自身宽度)
    # r_ratio_out (外管圆角率)

    # 尝试拟合 Ix = A * b_out * h_out^3 * (1 - X^B) * (1 + C*t_ratio) * (1 + D*r_ratio)
    def ix_comprehensive(data, A, B, C, D):
        b_out, h_out, X, t_ratio, r_ratio = data
        I_rect = b_out * h_out**3 / 12
        return A * I_rect * (1 - X**B) * (1 + C*t_ratio) * (1 + D*r_ratio)

    # 准备数据
    data_x = (df['b_out'].values, df['h_out'].values, df['X'].values,
              df['t_ratio'].values, df['r_ratio'].values)

    try:
        popt, _ = opt.curve_fit(ix_comprehensive, data_x, df['Ix'].values,
                                p0=[1, 2, 0.1, 0.1], maxfev=10000)

        print(f"\nIx 综合拟合公式:")
        print(f"Ix = {popt[0]:.4f} * (b_out*h_out^3/12) * (1 - X^{popt[1]:.4f}) * ")
        print(f"     (1 + {popt[2]:.4f}*t_ratio) * (1 + {popt[3]:.4f}*r_ratio)")

        # 评估
        Ix_pred = ix_comprehensive(data_x, *popt)
        print(f"\nR2 = {r2_score(df['Ix'], Ix_pred):.6f}")

    except Exception as e:
        print(f"综合拟合失败: {e}")


def simplified_formula(df, output_dir='.'):
    """简化公式：仅基于主要参数"""
    import os
    print("\n" + "="*60)
    print("简化公式拟合")
    print("="*60)

    df = df.copy()

    # 简化模型：Ix ≈ C * b_out * h_out^3 * (1 - X^2) / 12
    df['I_rect_x'] = df['b_out'] * df['h_out']**3 / 12
    df['I_rect_y'] = df['h_out'] * df['b_out']**3 / 12

    # 多元线性回归拟合 kx 和 ky
    from sklearn.linear_model import LinearRegression

    # Ix
    X_features = df[['I_rect_x', 'X', 'X_squared']].values
    y_x = df['Ix'].values

    model_x = LinearRegression()
    model_x.fit(X_features, y_x)

    print("\nIx 简化公式 (多元线性):")
    print(f"Ix = {model_x.coef_[0]:.6f} * I_rect_x + {model_x.coef_[1]:.6f} * X + {model_x.coef_[2]:.6f} * X^2 + {model_x.intercept_:.2f}")

    yx_pred = model_x.predict(X_features)
    print(f"R2 = {r2_score(y_x, yx_pred):.6f}")

    # Iy
    X_features_y = df[['I_rect_y', 'X', 'X_squared']].values
    y_y = df['Iy'].values

    model_y = LinearRegression()
    model_y.fit(X_features_y, y_y)

    print("\nIy 简化公式 (多元线性):")
    print(f"Iy = {model_y.coef_[0]:.6f} * I_rect_y + {model_y.coef_[1]:.6f} * X + {model_y.coef_[2]:.6f} * X^2 + {model_y.intercept_:.2f}")

    yy_pred = model_y.predict(X_features_y)
    print(f"R2 = {r2_score(y_y, yy_pred):.6f}")

    # 可视化预测效果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.scatter(df['Ix']/1e6, yx_pred/1e6, alpha=0.5)
    ax1.plot([0, df['Ix'].max()/1e6], [0, df['Ix'].max()/1e6], 'r--', label='y=x')
    ax1.set_xlabel('实际 Ix (x10^6 mm^4)')
    ax1.set_ylabel('预测 Ix (x10^6 mm^4)')
    ax1.set_title(f'Ix 预测效果 (R2={r2_score(y_x, yx_pred):.4f})')
    ax1.legend()

    ax2 = axes[1]
    ax2.scatter(df['Iy']/1e6, yy_pred/1e6, alpha=0.5)
    ax2.plot([0, df['Iy'].max()/1e6], [0, df['Iy'].max()/1e6], 'r--', label='y=x')
    ax2.set_xlabel('实际 Iy (x10^6 mm^4)')
    ax2.set_ylabel('预测 Iy (x10^6 mm^4)')
    ax2.set_title(f'Iy 预测效果 (R2={r2_score(y_y, yy_pred):.4f})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_accuracy.png'), dpi=150, bbox_inches='tight')
    print(f"\n预测效果图已保存至 {output_dir}/prediction_accuracy.png")
    plt.close()

    return model_x, model_y


def ke_regression_analysis(df, output_dir='output'):
    """
    ke 的回归拟合分析
    使用基础几何参数拟合 ke
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("ke 回归拟合分析")
    print("="*60)

    # 基础特征
    feature_cols = ['b_out', 'h_out', 't_out', 'r0_out', 'b_in', 'h_in', 't_in', 'r0_in', 'X']
    X = df[feature_cols]
    y_ke = df['ke']

    results = {}

    # 1. 线性回归
    print("\n【1】线性回归")
    X_train, X_test, y_train, y_test = train_test_split(X, y_ke, test_size=0.2, random_state=42)
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)
    y_pred = model_linear.predict(X_test)
    r2_linear = r2_score(y_test, y_pred)
    print(f"  R2 = {r2_linear:.6f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.6f}")
    print("\n  回归系数:")
    for feat, coef in zip(feature_cols, model_linear.coef_):
        print(f"    {feat}: {coef:.6f}")
    print(f"  截距: {model_linear.intercept_:.6f}")
    results['linear'] = {'model': model_linear, 'r2': r2_linear}

    # 2. 多项式回归 (2阶)
    print("\n【2】多项式回归 (degree=2)")
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X_train, y_train)
    y_pred_poly = poly_model.predict(X_test)
    r2_poly = r2_score(y_test, y_pred_poly)
    print(f"  R2 = {r2_poly:.6f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.6f}")

    # 输出 ke 多项式回归公式
    poly_feature_names = poly_model.named_steps['poly'].get_feature_names_out(feature_cols)
    ke_formula = get_polynomial_formula(poly_model, poly_feature_names, "ke")
    print(f"\n  ke 多项式回归公式:")
    print(f"  {ke_formula}")

    results['poly2'] = {'model': poly_model, 'r2': r2_poly, 'formula': ke_formula, 'feature_names': poly_feature_names}

    # 3. 基于物理意义的公式拟合
    print("\n【3】物理意义公式拟合")
    # 假设 ke = f(X, t_ratio_out, b_ratio)
    # 尝试 ke = A * (1 - X^B) * (1 + C*t_ratio_out) * (1 + D*(b_in/b_out))
    # 注意：b_ratio = b_out/b_in (外/内), 所以 b_in/b_out = 1/b_ratio

    def ke_physics(data, A, B, C, D):
        X, t_ratio_out, b_ratio = data
        return A * (1 - X**B) * (1 + C*t_ratio_out) * (1 + D/b_ratio)  # 1/b_ratio = b_in/b_out

    # 准备数据 - 使用已定义的特征
    # t_ratio_out = t_out / b_out (外管相对厚度)
    # b_ratio = b_out / b_in (外/内)
    data_ke = (df['X'].values, df['t_ratio_out'].values, df['b_ratio'].values)

    try:
        popt, _ = opt.curve_fit(ke_physics, data_ke, df['ke'].values,
                                p0=[1, 0.5, 0.1, -0.1], maxfev=10000)
        ke_pred = ke_physics(data_ke, *popt)
        r2_physics = r2_score(df['ke'], ke_pred)

        print(f"\n  拟合公式: ke = {popt[0]:.4f} * (1 - X^{popt[1]:.4f}) * ")
        print(f"            (1 + {popt[2]:.4f}*t_ratio_out) * (1 + {popt[3]:.4f}*(b_in/b_out))")
        print(f"  其中: t_ratio_out = t_out/b_out, (b_in/b_out) = 1/(b_out/b_in)")
        print(f"  R2 = {r2_physics:.6f}")
        results['physics'] = {'params': popt, 'r2': r2_physics, 'func': ke_physics}
    except Exception as e:
        print(f"  物理公式拟合失败: {e}")
        r2_physics = 0

    # 4. 简化公式：ke 与 X 的关系
    print("\n【4】简化公式：ke = f(X)")
    # 尝试 ke = A + B*X + C*X^2
    X_simple = df[['X', 'X_squared']].values
    model_simple = LinearRegression()
    model_simple.fit(X_simple, df['ke'])
    ke_pred_simple = model_simple.predict(X_simple)
    r2_simple = r2_score(df['ke'], ke_pred_simple)

    print(f"\n  ke = {model_simple.intercept_:.4f} + {model_simple.coef_[0]:.4f}*X + {model_simple.coef_[1]:.4f}*X^2")
    print(f"  R2 = {r2_simple:.6f}")
    results['simple'] = {'model': model_simple, 'r2': r2_simple}

    # 5. 考虑壁厚比和内外管尺寸比的公式
    print("\n【5】综合公式")
    # ke = A * (1 - X^B) * (t_out/t_in)^C = A * (1 - X^B) * (t_ratio)^C
    # t_ratio = t_out / t_in (外/内)

    def ke_comprehensive(data, A, B, C):
        X, t_ratio = data
        t_ratio = np.clip(t_ratio, 0.1, 10)  # 避免极端值
        return A * (1 - X**B) * (t_ratio**C)

    # 使用已定义的 t_ratio = t_out / t_in
    data_comp = (df['X'].values, df['t_ratio'].values)
    try:
        popt2, _ = opt.curve_fit(ke_comprehensive, data_comp, df['ke'].values,
                                 p0=[1, 0.5, 0.05], maxfev=10000)
        ke_pred_comp = ke_comprehensive(data_comp, *popt2)
        r2_comp = r2_score(df['ke'], ke_pred_comp)

        print(f"\n  ke = {popt2[0]:.4f} * (1 - X^{popt2[1]:.4f}) * (t_out/t_in)^{popt2[2]:.4f}")
        print(f"  R2 = {r2_comp:.6f}")
        results['comprehensive'] = {'params': popt2, 'r2': r2_comp}
    except Exception as e:
        print(f"  综合公式拟合失败: {e}")
        r2_comp = 0

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ke vs X 散点图
    ax1 = axes[0, 0]
    ax1.scatter(df['X'], df['ke'], alpha=0.5, c=df['t_out'], cmap='viridis')
    ax1.set_xlabel('X (空心率)')
    ax1.set_ylabel('ke')
    ax1.set_title('ke vs X (颜色=t_out)')

    # ke vs t_ratio_out
    ax2 = axes[0, 1]
    ax2.scatter(df['t_ratio_out'], df['ke'], alpha=0.5, c=df['X'], cmap='plasma')
    ax2.set_xlabel('t_ratio_out (外管相对厚度)')
    ax2.set_ylabel('ke')
    ax2.set_title('ke vs t_ratio_out (颜色=X)')

    # 线性回归预测效果
    ax3 = axes[0, 2]
    y_pred_linear = model_linear.predict(X)
    ax3.scatter(df['ke'], y_pred_linear, alpha=0.5)
    ax3.plot([0, 1.2], [0, 1.2], 'r--', label='y=x')
    ax3.set_xlabel('实际 ke')
    ax3.set_ylabel('预测 ke (线性)')
    ax3.set_title(f'线性回归 (R2={r2_linear:.4f})')
    ax3.legend()

    # 多项式回归预测效果
    ax4 = axes[1, 0]
    y_pred_poly_full = poly_model.predict(X)
    ax4.scatter(df['ke'], y_pred_poly_full, alpha=0.5)
    ax4.plot([0, 1.2], [0, 1.2], 'r--', label='y=x')
    ax4.set_xlabel('实际 ke')
    ax4.set_ylabel('预测 ke (多项式)')
    ax4.set_title(f'多项式回归 (R2={r2_poly:.4f})')
    ax4.legend()
    # 简化公式预测效果
    ax5 = axes[1, 1]
    ax5.scatter(df['ke'], ke_pred_simple, alpha=0.5)
    ax5.plot([0, 1.2], [0, 1.2], 'r--', label='y=x')
    ax5.set_xlabel('实际 ke')
    ax5.set_ylabel('预测 ke (简化)')
    ax5.set_title(f'简化公式 (R2={r2_simple:.4f})')
    ax5.legend()

    # 残差分析
    ax6 = axes[1, 2]
    residuals = df['ke'] - y_pred_poly_full
    ax6.scatter(y_pred_poly_full, residuals, alpha=0.5)
    ax6.axhline(y=0, color='r', linestyle='--')
    ax6.set_xlabel('预测 ke')
    ax6.set_ylabel('残差')
    ax6.set_title('残差分析')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ke_regression_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"\nke 分析图已保存至 {output_dir}/ke_regression_analysis.png")
    plt.close()

    # 保存结果到文本文件
    with open(os.path.join(output_dir, 'ke_regression_results.txt'), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ke 回归拟合结果\n")
        f.write("="*60 + "\n\n")

        f.write("【1】线性回归\n")
        f.write(f"R2 = {r2_linear:.6f}\n")
        f.write("系数:\n")
        for feat, coef in zip(feature_cols, model_linear.coef_):
            f.write(f"  {feat}: {coef:.6f}\n")
        f.write(f"截距: {model_linear.intercept_:.6f}\n\n")

        f.write("【2】多项式回归 (degree=2)\n")
        f.write(f"R2 = {r2_poly:.6f}\n")
        f.write(f"ke 多项式回归公式:\n{ke_formula}\n\n")

        f.write("【3】简化公式\n")
        f.write(f"ke = {model_simple.intercept_:.4f} + {model_simple.coef_[0]:.4f}*X + {model_simple.coef_[1]:.4f}*X^2\n")
        f.write(f"R2 = {r2_simple:.6f}\n\n")

        if r2_physics > 0:
            f.write("【4】物理意义公式\n")
            f.write(f"ke = {popt[0]:.4f} * (1 - X^{popt[1]:.4f}) * (1 + {popt[2]:.4f}*t_ratio_out) * (1 + {popt[3]:.4f}*(b_in/b_out))\n")
            f.write(f"其中: t_ratio_out = t_out/b_out, (b_in/b_out) = 1/(b_out/b_in)\n")
            f.write(f"R2 = {r2_physics:.6f}\n\n")

        if r2_comp > 0:
            f.write("【5】综合公式\n")
            f.write(f"ke = {popt2[0]:.4f} * (1 - X^{popt2[1]:.4f}) * (t_out/t_in)^{popt2[2]:.4f}\n")
            f.write(f"R2 = {r2_comp:.6f}\n\n")

    print(f"ke 回归结果已保存至 {output_dir}/ke_regression_results.txt")

    return results


def get_polynomial_formula(model, feature_names, target_name="y", precision=6):
    """
    从多项式回归管道中提取公式字符串
    model: Pipeline, 包含 PolynomialFeatures 和 LinearRegression
    feature_names: 多项式特征名称列表（可通过 poly_model_x.named_steps['poly'].get_feature_names_out() 获得）
    target_name: 因变量名称
    precision: 系数显示精度
    """
    coef = model.named_steps['linear'].coef_
    intercept = model.named_steps['linear'].intercept_

    terms = []
    for name, c in zip(feature_names, coef):
        if abs(c) > 1e-12:  # 忽略零系数
            sign = '+' if c > 0 else '-'
            coeff_abs = abs(c)
            if abs(coeff_abs - 1.0) < 1e-12:
                term = f"{sign} {name}"
            else:
                term = f"{sign} {coeff_abs:.{precision}f} * {name}"
            terms.append(term)

    if abs(intercept) > 1e-12:
        formula = f"{target_name} = {intercept:.{precision}f}" + "".join(terms)
    else:
        formula = f"{target_name} = " + "".join(terms).lstrip('+')
    return formula


def main():
    """主函数"""
    from datetime import datetime
    import os

    # 创建带时间戳的输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n创建输出文件夹: {output_dir}/")

    # 文件路径
    filepath = r"D:\AI_study\shujukuzhuanhua\CFDST_Parameter_Analysis2.csv"

    # 加载数据
    df = load_data(filepath)

    # 特征工程
    df = feature_engineering(df)

    # 相关性分析
    corr_matrix = correlation_analysis(df, output_dir)

    # 线性回归
    model_x, model_y = linear_regression_analysis(df)

    # 多项式回归
    poly_model_x, poly_model_y, feature_names = polynomial_regression(df, degree=2)

    # 输出完整公式
    print("\n" + "="*60)
    print("多项式回归完整公式")
    print("="*60)
    formula_ix = get_polynomial_formula(poly_model_x, feature_names, "Ix")
    formula_iy = get_polynomial_formula(poly_model_y, feature_names, "Iy")
    print("Ix =", formula_ix)
    print("Iy =", formula_iy)

    # 基于物理意义的公式
    physics_based_formula(df, output_dir)

    # 综合公式
    comprehensive_formula(df)

    # 简化公式
    simplified_formula(df, output_dir)

    # ke 回归分析
    ke_results = ke_regression_analysis(df, output_dir)

    print("\n" + "="*60)
    print(f"分析完成！所有结果已保存至 {output_dir}/ 文件夹")
    print("="*60)


if __name__ == '__main__':
    main()
