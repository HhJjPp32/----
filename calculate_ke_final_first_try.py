"""
calculate_ke_final.py
Calculate effective area coefficient (ke) for CFDST (Concrete-Filled Double Skin Tubular) columns
Based on Zhenlin Li et al. (2022), considering corner radius effects of outer and inner tubes
Converted from MATLAB version: Calculate_ke_Final.m
"""

import numpy as np
from scipy import integrate
import os
import pandas as pd


def calculate_ke_final(input_file='input.txt'):
    """
    Main function: Calculate effective area coefficient ke for CFDST columns

    Args:
        input_file: Input filename containing 8 parameters: bo, ho, to, ro1, bi, hi, ti, ri1
    """
    # Check if input file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'Error: {input_file} not found, please check file path')

    # Read input data
    data = np.loadtxt(input_file)
    if data.size < 8:
        raise ValueError('input.txt must contain at least 8 parameters: bo, ho, to, ro1, bi, hi, ti, ri1')

    bo = data[0]
    ho = data[1]
    to = data[2]
    ro1 = data[3]
    bi = data[4]
    hi = data[5]
    ti = data[6]
    ri1 = data[7]

    print('=== Input Parameters ===')
    print(f'Outer: bo={bo:.2f}, ho={ho:.2f}, to={to:.2f}, ro1={ro1:.2f}')
    print(f'Inner: bi={bi:.2f}, hi={hi:.2f}, ti={ti:.2f}, ri1={ri1:.2f}')
    print('====================\n')

    # 1. Calculate outer tube corner radius ro2 (based on Eq.5)
    # Corner radius is reduced according to wall thickness
    ro2_w = 0.0
    if ho > 0:
        ro2_w = (ho - 2 * to) / ho * ro1

    ro2_h = 0.0
    if bo > 0:
        ro2_h = (bo - 2 * to) / bo * ro1

    # Width direction parameters (for Abi)
    ho2_w = ho - 2 * to
    bco_w = bo - 2 * to - 2 * ro2_w
    bci_w = bi - 2 * ri1
    hi_w = hi
    hci_w = hi - 2 * ri1
    bi_w = bi

    # Depth direction parameters (for Ahi)
    # Swap width/height to calculate invalid area in perpendicular direction
    ho2_h = bo - 2 * to
    bco_h = ho - 2 * to - 2 * ro2_h
    bci_h = hi - 2 * ri1
    hi_h = bi
    hci_h = bi - 2 * ri1
    bi_h = hi

    # 2. Calculate total concrete area Ac
    # Use average ro2 for area calculation
    ro2_avg = (ro2_w + ro2_h) / 2
    A_outer_inner = (bo - 2 * to) * (ho - 2 * to) - (4 - np.pi) * ro2_avg ** 2
    A_inner_outer = bi * hi - (4 - np.pi) * ri1 ** 2
    Ac = A_outer_inner - A_inner_outer

    # 3. Calculate invalid area (using unified method)
    Abi = calc_invalid_area_unified(ho2_w, bco_w, ro2_w, hi_w, bci_w, ri1, hci_w, bi_w)
    Ahi = calc_invalid_area_unified(ho2_h, bco_h, ro2_h, hi_h, bci_h, ri1, hci_h, bi_h)

    # 4. Calculate effective area Ace and ke
    Ace = Ac - 2 * Abi - 2 * Ahi
    if Ace < 0:
        Ace = 0.0
    ke = Ace / Ac

    # Output results
    print('=== Results ===')
    print(f'Total Concrete Area Ac  = {Ac:.4f}')
    print(f'Width Invalid Area Abi   = {Abi:.4f}')
    print(f'Height Invalid Area Ahi   = {Ahi:.4f}')
    print(f'Effective Area Ace   = {Ace:.4f}')
    print('----------------')
    print(f'Efficiency Factor ke    = {ke:.6f}')
    print('================')

    return ke, Ace, Ac, Abi, Ahi


def calc_invalid_area_unified(h_o2, b_co, r_o2, h_i, b_ci, r_i1, h_ci, b_i):
    """
    Unified function to calculate invalid area (applies to both width and height directions)

    Args:
        h_o2: Outer tube height (after subtracting 2*wall thickness)
        b_co: Outer tube flat segment width
        r_o2: Outer tube corner radius
        h_i: Inner tube height
        b_ci: Inner tube flat segment width
        r_i1: Inner tube corner radius
        h_ci: Inner tube flat segment height
        b_i: Inner tube width

    Returns:
        Area: Invalid area
    """
    # Determine integration range, use max of outer/inner tube half-widths
    # Ensure integration interval covers all possible geometric regions (including corner areas)
    x_max = max(b_co / 2 + r_o2, b_ci / 2 + r_i1)

    # Special case handling: invalid area is 0 when there's no geometry
    if x_max <= 0:
        return 0.0

    # Use scipy.integrate.quad for numerical integration
    # Integrate from 0 to x_max, then multiply by 2 (symmetry)

    def integrand(x):
        return get_invalid_depth(x, h_o2, b_co, r_o2, h_i, b_ci, r_i1, h_ci, b_i)

    # Perform integration using quad with appropriate error tolerance
    result, error = integrate.quad(integrand, 0, x_max, limit=100)
    Area = 2 * result

    return Area


def get_invalid_depth(x, h_o2, b_co, r_o2, h_i, b_ci, r_i1, h_ci, b_i):
    """
    Calculate invalid depth (at position x)

    Args:
        x: Horizontal position coordinate
        h_o2, b_co, r_o2: Outer tube geometric parameters
        h_i, b_ci, r_i1, h_ci, b_i: Inner tube geometric parameters

    Returns:
        val: Invalid depth at this position
    """
    x = abs(x)

    # --- Calculate gap between outer and inner tubes ---
    # 1.1 Calculate outer tube half-height Y_out (considering corner radius)
    h_co = h_o2 - 2 * r_o2
    if x <= b_co / 2:
        y_out = h_o2 / 2  # Flat segment
    elif x <= b_co / 2 + r_o2:
        term = r_o2 ** 2 - (x - b_co / 2) ** 2
        if term < 0:
            term = 0.0
        y_out = h_co / 2 + np.sqrt(term)  # Corner segment
    else:
        y_out = 0.0  # Beyond cross-section range

    # 1.2 Calculate inner tube half-height Y_in (considering corner radius)
    if x <= b_ci / 2:
        y_in = h_i / 2  # Inner tube flat segment
    elif x <= b_ci / 2 + r_i1:
        term = r_i1 ** 2 - (x - b_ci / 2) ** 2
        if term < 0:
            term = 0.0
        y_in = h_ci / 2 + np.sqrt(term)  # Inner tube corner segment
    elif h_ci > 0 and x <= b_i / 2 + h_ci / 4:
        # Parabolic transition region - handles case where inner height < width
        term = h_ci * (h_ci / 4 - (x - b_i / 2))
        if term < 0:
            term = 0.0
        y_in = np.sqrt(term)
    else:
        y_in = 0.0

    # 1.3 Calculate concrete effective filling gap
    gap = y_out - y_in
    if gap < 0:
        gap = 0.0

    # --- Calculate parabolic constraint boundary contributions ---
    # 2.1 Outer tube parabolic constraint (depth)
    if b_co > 0 and x <= b_co / 2:
        d_out = b_co / 4 - x ** 2 / b_co
    else:
        d_out = 0.0

    # 2.2 Inner tube parabolic constraint (depth)
    if b_ci > 0 and x <= b_ci / 2:
        d_in = b_ci / 4 - x ** 2 / b_ci
    else:
        d_in = 0.0

    # --- Combine to calculate invalid depth ---
    # Take minimum of gap and parabolic constraint as invalid depth
    val = min(gap, d_out + d_in)

    return val


def calculate_ke_from_csv(input_csv, output_csv=None):
    """
    Batch process CFDST data from CSV file and add results as new columns

    Args:
        input_csv: Path to input CSV file with columns: bo, ho, t, r01, bi, hi, ti, r0,in
        output_csv: Path to output CSV file. If None, will use input_filename_results.csv
    """
    # Check if input file exists
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f'Error: {input_csv} not found, please check file path')

    # Set default output filename if not specified
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_results{ext}"

    # Read CSV file (handle BOM and different column names)
    df = pd.read_csv(input_csv, encoding='utf-8-sig')

    # Map column names (handle spaces in column names)
    column_mapping = {}
    for col in df.columns:
        col_clean = col.strip().lower().replace(' ', '').replace('_', '')
        if col_clean in ['bo', 'ho', 't', 'to', 'r01', 'ro1', 'bi', 'hi', 'ti', 'r0,in', 'ri1']:
            column_mapping[col] = col_clean

    # Rename columns for easier access
    df_renamed = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required_cols = ['bo', 'ho', 't', 'r01', 'bi', 'hi', 'ti']
    missing_cols = [c for c in required_cols if c not in df_renamed.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

    # Determine ri1 column name (might be 'r0,in' or 'ri1')
    ri1_col = None
    for col in ['r0,in', 'ri1']:
        if col in df_renamed.columns:
            ri1_col = col
            break
    if ri1_col is None:
        raise ValueError("Missing ri1 column (expected 'r0,in' or 'ri1')")

    # Initialize result lists
    results = []

    # Process each row
    total_rows = len(df_renamed)
    print(f"Processing {total_rows} rows from {input_csv}...")

    for idx, row in df_renamed.iterrows():
        try:
            # Extract parameters
            bo = float(row['bo'])
            ho = float(row['ho'])
            to = float(row['t'])
            ro1 = float(row['r01'])
            bi = float(row['bi'])
            hi = float(row['hi'])
            ti = float(row['ti'])
            ri1 = float(row[ri1_col])

            # Calculate
            ke, Ace, Ac, Abi, Ahi = calculate_ke_params(bo, ho, to, ro1, bi, hi, ti, ri1)

            results.append({
                'ke': ke,
                'Ace': Ace,
                'Ac': Ac,
                'Abi': Abi,
                'Ahi': Ahi,
                'status': 'success'
            })

            if (idx + 1) % 100 == 0 or idx == total_rows - 1:
                print(f"  Processed {idx + 1}/{total_rows} rows...")

        except Exception as e:
            print(f"  Error at row {idx + 1}: {e}")
            results.append({
                'ke': np.nan,
                'Ace': np.nan,
                'Ac': np.nan,
                'Abi': np.nan,
                'Ahi': np.nan,
                'status': f'error: {e}'
            })

    # Add results to original dataframe
    result_df = pd.DataFrame(results)
    df_output = pd.concat([df, result_df], axis=1)

    # Save to output CSV
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_csv}")
    print(f"Total processed: {len(results)} rows")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')} rows")

    return df_output


def calculate_ke_params(bo, ho, to, ro1, bi, hi, ti, ri1):
    """
    Calculate ke given all parameters (for CSV batch processing)

    Returns:
        ke, Ace, Ac, Abi, Ahi
    """
    # 1. Calculate outer tube corner radius ro2 (based on Eq.5)
    ro2_w = 0.0
    if ho > 0:
        ro2_w = (ho - 2 * to) / ho * ro1

    ro2_h = 0.0
    if bo > 0:
        ro2_h = (bo - 2 * to) / bo * ro1

    # Width direction parameters (for Abi)
    ho2_w = ho - 2 * to
    bco_w = bo - 2 * to - 2 * ro2_w
    bci_w = bi - 2 * ri1
    hi_w = hi
    hci_w = hi - 2 * ri1
    bi_w = bi

    # Depth direction parameters (for Ahi)
    ho2_h = bo - 2 * to
    bco_h = ho - 2 * to - 2 * ro2_h
    bci_h = hi - 2 * ri1
    hi_h = bi
    hci_h = bi - 2 * ri1
    bi_h = hi

    # 2. Calculate total concrete area Ac
    ro2_avg = (ro2_w + ro2_h) / 2
    A_outer_inner = (bo - 2 * to) * (ho - 2 * to) - (4 - np.pi) * ro2_avg ** 2
    A_inner_outer = bi * hi - (4 - np.pi) * ri1 ** 2
    Ac = A_outer_inner - A_inner_outer

    # 3. Calculate invalid area
    Abi = calc_invalid_area_unified(ho2_w, bco_w, ro2_w, hi_w, bci_w, ri1, hci_w, bi_w)
    Ahi = calc_invalid_area_unified(ho2_h, bco_h, ro2_h, hi_h, bci_h, ri1, hci_h, bi_h)

    # 4. Calculate effective area Ace and ke
    Ace = Ac - 2 * Abi - 2 * Ahi
    if Ace < 0:
        Ace = 0.0
    ke = Ace / Ac if Ac > 0 else 0.0

    return ke, Ace, Ac, Abi, Ahi


if __name__ == '__main__':
    # Example 1: Process single file
    # calculate_ke_final()

    # Example 2: Process CSV file with multiple rows
    input_path = r"D:\AI_study\shujukuzhuanhua\calculate_ke.csv"
    output_path = r"D:\AI_study\shujukuzhuanhua\calculate_ke_results.csv"
    calculate_ke_from_csv(input_path, output_path)
