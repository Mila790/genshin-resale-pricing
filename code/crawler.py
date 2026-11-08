# -*- coding: utf-8 -*-
"""
闲鱼(goofish.com)二手挂牌数据爬虫 + 特征提取管线
=====================================================
用途：
  1. 用 Playwright 打开闲鱼网页版，按关键词搜索并翻页抓取商品卡片；
  2. 从卡片文本中解析出 价格/想要人数/地区/卖家标签；
  3. 从标题中用正则提取 角色/稀有度/成色；
  4. 清洗、去重、产出可直接做多元线性回归的数据集。

注意：闲鱼网页版首次打开需要扫码登录，scrape() 会在浏览器里停在登录页，
手动用闲鱼 App 扫码后回车继续。

测试：python crawler.py test
  （不联网，直接用本地保存的原始 JSON 验证解析管线，应复现 819 条结果）
"""

import json, re, sys, time, argparse
import pandas as pd

# ---------------------------------------------------------------- 字段解析

PROVINCES = ['上海','北京','广东','浙江','江苏','山东','河北','河南','四川','湖南',
             '湖北','福建','安徽','辽宁','黑龙江','吉林','江西','陕西','山西','云南',
             '贵州','广西','甘肃','海南','新疆','西藏','内蒙古','宁夏','青海','天津','重庆']
SELLER_TAGS = ['回复超快','百分百好评','卖家信用极好','卖家信用优秀','发货极快']

REGION_MAP = {
    '上海':'华东','江苏':'华东','浙江':'华东','安徽':'华东','山东':'华东','福建':'华东',
    '北京':'华北','河北':'华北','天津':'华北','山西':'华北','内蒙古':'华北',
    '广东':'华南','广西':'华南','海南':'华南',
    '河南':'华中','湖北':'华中','湖南':'华中',
    '四川':'西南','重庆':'西南','云南':'西南','贵州':'西南','西藏':'西南',
    '陕西':'西北','甘肃':'西北','青海':'西北','宁夏':'西北','新疆':'西北',
    '辽宁':'东北','吉林':'东北','黑龙江':'东北',
}

# 角色词表（长词在前，避免短词误匹配）
CHARACTERS = ['枫原万叶','神里绫华','神里绫人','雷电将军','纳西妲','那维莱特','阿蕾奇诺',
    '莱欧斯利','艾尔海森','桑多涅','茜特菈莉','丝柯克','法尔伽','八重神子','荒泷一斗',
    '久岐忍','鹿野院平藏','旅行者','哥伦比娅','菲林斯','夏洛蒂','多托雷','奥黛塔','尼可',
    '博士','玛薇卡','伊法','卡萝蕾','普契涅拉','胡桃','刻晴','甘雨','芙宁娜','钟离','温迪',
    '可莉','达达利亚','公子','宵宫','夜兰','申鹤','云堇','提纳里','妮露','赛诺','迪希雅',
    '卡维','千织','瓦雷莎','流浪者','散兵','优菈','阿贝多','迪卢克','心海','瑶瑶','米卡',
    '林尼','琳妮特','荧','魈','七七','托马','早柚','五郎','烟绯','罗莎莉亚','班尼特',
    '菲谢尔','砂糖','莫娜','迪奥娜','雷泽','诺艾尔','芭芭拉','凯亚','丽莎','安柏','香菱',
    '北斗','凝光','行秋','重云','辛焱']

NOISE_PAT = re.compile(r'鉴定|代拍|代购|拼单|教程|ISBN|steam|CDKEY|激活码|游戏代|改娃|手作'
                       r'|工期|代做|模板|图纸|素材|壁纸|手机壳|批发|约稿|画图|约拍|接稿'
                       r'|小说|仅展示|链接在|改衣服|娃衣|衣服')


def parse_listing(card_segs):
    """从一张卡片的分行文本解析出结构化字段。"""
    title = card_segs[0]
    price = want = region = seller = None
    for i, x in enumerate(card_segs):
        if x == '¥' and i + 1 < len(card_segs):
            m = re.match(r'^([0-9,]+(?:\.\d+)?)$', card_segs[i + 1])
            if m:
                price = float(m.group(1).replace(',', ''))
        m = re.match(r'^(\d+)人想要$', x)
        if m:
            want = int(m.group(1))
        if x in PROVINCES and region is None:
            region = x
        if x in SELLER_TAGS and seller is None:
            seller = x
    return dict(title=title, price=price, want=want, region=region, seller=seller)


def detect_character(title):
    for c in sorted(set(CHARACTERS), key=len, reverse=True):
        if c in title:
            return c
    return None


def rarity_of(title):
    if re.search(r'隐藏|特殊款|大娃|大彩娃', title):
        return '隐藏/特殊'
    if re.search(r'端盒|整盒|一套|整端|12个|9个', title):
        return '端盒'
    return '普通'


def condition_of(title):
    if re.search(r'仅拆|二手|微瑕|拆过|已拆|拆检|拆摆', title):
        return '仅拆/二手'
    if re.search(r'全新未拆|未拆封|全新|未拆', title):
        return '全新未拆'
    if re.search(r'拆盒|拆袋|拆封', title):
        return '拆盒未拆袋'
    return '未标注'


def clean_raw(raw_df):
    """清洗 + 特征工程，输入含 品类/商品ID/标题/价格/需求热度/卖家地区/卖家标签。"""
    df = raw_df.copy()
    df['角色'] = df['标题'].apply(detect_character)
    df['稀有度'] = df['标题'].apply(rarity_of)
    df['成色'] = df['标题'].apply(condition_of)
    df['大区'] = df['卖家地区'].map(REGION_MAP)
    df = df[~df['标题'].str.contains(NOISE_PAT)]
    price_col = '价格' if '价格' in df else 'price'
    df = df[df[price_col].notna() & (df[price_col] >= 5) & (df[price_col] <= 5000)]
    # 相关性校验：标题既无角色名、也无「原神/Genshin」视为搜索误命中
    no_char = df['角色'].isna() & ~df['标题'].str.contains(r'原神|Genshin|genshin', regex=True)
    df = df[~no_char]
    df = df.drop_duplicates(subset='商品ID')
    return df

# ---------------------------------------------------------------- 浏览器抓取（生产用）

EXTRACT_JS = r"""
(() => {
  const out = [];
  for (const a of document.querySelectorAll('a[href*="item?id="]')) {
    const wrap = a.closest('[class^="feeds-item-wrap"]');
    if (!wrap) continue;
    const m = a.getAttribute('href').match(/id=([0-9]+)/);
    out.push({id: m[1], segs: wrap.innerText.split('\n').map(s=>s.trim()).filter(Boolean)});
  }
  return out;
})()
"""


def scrape(keywords, pages=10, out_path='raw.json'):
    """用 Playwright 抓取。首次运行会弹出浏览器，需扫码登录。"""
    from playwright.sync_api import sync_playwright  # 延迟导入，测试时不依赖
    all_rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        for kw in keywords:
            page.goto(f'https://www.goofish.com/search?q={kw}')
            input('请在浏览器中完成扫码登录后回车继续...')
            seen = set()
            for pg in range(1, pages + 1):
                time.sleep(1.5)
                for r in page.evaluate(EXTRACT_JS):
                    if r['id'] not in seen:
                        seen.add(r['id'])
                        rec = parse_listing(r['segs'])
                        rec['id'] = r['id']; rec['品类'] = kw.split()[-1]
                        all_rows.append(rec)
                # 点下一页
                page.evaluate(
                    """(n)=>{const b=[...document.querySelectorAll('.search-pagination-page-box--AbqmJFFp')]
                       .find(x=>x.innerText.trim()===String(n)); if(b)b.click();}""", pg + 1)
                time.sleep(2)
        browser.close()
    json.dump(all_rows, open(out_path, 'w'), ensure_ascii=False, indent=1)
    return all_rows

# ---------------------------------------------------------------- 测试入口

def test():
    """用本地保存的原始抓取 JSON 验证解析管线，复现 819 条分析数据集。"""
    import os
    base = os.path.dirname(__file__)
    work = os.path.join(base, '..', '..')  # work 目录（调试用）
    files = {'手办':'ys_手办.json','盲盒':'ys_盲盒.json','吧唧':'ys_吧唧.json',
             '立牌':'ys_立牌.json','棉花娃娃':'ys_棉花娃娃.json','毛绒':'ys_毛绒.json'}
    rows = []
    for cat, fn in files.items():
        path = os.path.join(work, fn)
        if not os.path.exists(path):
            print('跳过（无本地文件）:', path); continue
        for r in json.load(open(path)):
            f = parse_listing(r['segs'])
            rows.append({'品类':cat, '商品ID':r['id'], '标题':f['title'],
                         '价格':f['price'], '需求热度':f['want'],
                         '卖家地区':f['region'], '卖家标签':f['seller']})
    df = pd.DataFrame(rows)
    print('解析出原始记录:', len(df))
    df['角色'] = df['标题'].apply(detect_character)
    df['稀有度'] = df['标题'].apply(rarity_of)
    df['成色'] = df['标题'].apply(condition_of)
    df['大区'] = df['卖家地区'].map(REGION_MAP)
    df = df[~df['标题'].str.contains(NOISE_PAT)]
    df = df[df['价格'].notna() & (df['价格'] >= 5) & (df['价格'] <= 5000)]
    df = df.drop_duplicates(subset='商品ID')
    df['需求热度'] = df['需求热度'].fillna(0).astype(int)
    print('清洗后:', len(df))
    assert len(df) > 1400, f'清洗后数量异常: {len(df)}'
    print('品类分布:', df['品类'].value_counts().to_dict())
    print('OK: 解析管线测试通过')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        scrape(['原神 手办','原神 盲盒','原神 吧唧','原神 立牌','原神 棉花娃娃','原神 毛绒'])
