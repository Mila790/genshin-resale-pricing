"""多元线性回归：二手原神周边挂牌价的决定因素。

用法:
    python code/regression.py                          # 用默认数据
    python code/regression.py path/to/data.csv        # 指定数据

输出: 训练/测试 R²、5 折交叉验证 R²、系数与 p 值。
"""
import sys, os
import pandas as pd, numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

DEFAULT = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'genshin_pricing_819.csv')

def ols_with_pvalues(X, y):
    """普通最小二乘，返回系数、标准误、p 值。X 已含 const。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = (resid @ resid) / dof
    # X'X 可能因共线性奇异，使用伪逆可稳定返回近似解。
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=dof))
    p = np.where(se > 0, p, 1.0)
    return beta, se, p

def run(data_path=DEFAULT):
    df = pd.read_csv(data_path)
    # 缺失值：需求热度缺失记 0，并加一个缺失标记列（区分"真 0"和"未知"）
    df['需求热度缺失'] = df['需求热度'].isna().astype(int)
    df['需求热度'] = df['需求热度'].fillna(0)
    for c in ['角色','成色','大区','卖家标签']:
        df[c] = df[c].fillna('未标注')

    X = pd.get_dummies(df[['品类','稀有度','成色','大区','卖家标签']], drop_first=True)
    X['需求热度'] = df['需求热度']
    X['需求热度缺失'] = df['需求热度缺失']
    X = X.astype(float)
    y = np.log(df['价格'].values)

    # 80/20 留出
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=52)
    sc = StandardScaler().fit(X_tr)
    m = LinearRegression().fit(sc.transform(X_tr), y_tr)
    r2_tr = m.score(sc.transform(X_tr), y_tr)
    r2_te = m.score(sc.transform(X_te), y_te)

    # 5 折交叉验证
    sc_all = StandardScaler().fit(X)
    cv = cross_val_score(LinearRegression(), sc_all.transform(X), y, cv=5, scoring='r2')

    # 全样本 OLS（带 p 值）
    X_sm = np.column_stack([np.ones(len(X)), X.values])
    beta, se, p = ols_with_pvalues(X_sm, y)
    names = ['const'] + list(X.columns)

    print(f'样本量 n = {len(df)}')
    print(f'训练 R² = {r2_tr:.3f}')
    print(f'测试 R² = {r2_te:.3f}')
    print(f'5 折交叉验证 R² = {cv.mean():.3f} ± {cv.std():.3f}')
    print('\n主要系数（对 ln 价格，* = p<0.05）:')
    out = pd.DataFrame({'β_std': m.coef_, 'β_raw': beta[1:], 'p值': p[1:]}, index=X.columns)
    out['显著'] = out['p值'].apply(lambda v: '*' if v < 0.05 else '')
    print(out.sort_values('β_std', ascending=False).round(3).to_string())
    return out

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    run(path)
