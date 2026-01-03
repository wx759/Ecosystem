import pandas as pd

df = pd.read_csv(r"C:\Users\lyk\Downloads\draw_wih_nosie-2026-1-2_16_54_01.csv")

for col in df.columns:
    # 统一先转成字符串并去掉单引号
    cleaned = df[col].astype(str).str.replace("'", "", regex=False)

    # 再尝试转成数值，失败的自动变 NaN（不会 warning）
    df[col] = pd.to_numeric(cleaned, errors="coerce")
