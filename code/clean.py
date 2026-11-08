"""ETL: 从原始抓取 CSV 清洗到分析用 819 条。

输入: data/raw/genshin_listings_1798.csv  (列: 品类,商品ID,标题,价格,需求热度,卖家地区,卖家标签)
输出: data/processed/genshin_pricing_819.csv

用法: python code/clean.py
"""
import os, re, sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from crawler import (detect_character, rarity_of, condition_of, REGION_MAP, NOISE_PAT)

ROOT = os.path.join(os.path.dirname(__file__), '..')
RAW = os.path.join(ROOT, 'data', 'raw', 'genshin_listings_1798.csv')
OUT = os.path.join(ROOT, 'data', 'processed', 'genshin_pricing_819.csv')

def main():
    df = pd.read_csv(RAW)
    n0 = len(df)
    # R1 去重
    df = df.drop_duplicates(subset='商品ID')
    # R2 价格异常
    df = df[df['价格'].notna() & (df['价格'] >= 5) & (df['价格'] <= 5000)]
    # R3 服务噪音
    df = df[~df['标题'].str.contains(NOISE_PAT, na=False)]
    # R4 特征工程
    df['角色'] = df['标题'].apply(detect_character)
    df['稀有度'] = df['标题'].apply(rarity_of)
    df['成色'] = df['标题'].apply(condition_of)
    df['大区'] = df['卖家地区'].map(REGION_MAP)
    # R5 删除无明确角色的整盒/系列款
    df = df[df['角色'].notna()]
    # R6 分层抽样 819
    train, _ = train_test_split(df, train_size=819, random_state=52, stratify=df['品类'])
    train = train.reset_index(drop=True)
    train['大区'] = train['大区'].fillna('未知')
    train['卖家标签'] = train['卖家标签'].fillna('未标注')
    train['需求热度'] = train['需求热度'].fillna(0).astype(int)
    train.to_csv(OUT, index=False)
    print(f'清洗漏斗: {n0} -> {len(df)} -> {len(train)}')
    print(f'已写出 {OUT}')

if __name__ == '__main__':
    main()
