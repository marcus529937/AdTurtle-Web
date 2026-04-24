import json
import os
import time
import re
import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JSON_PATH = os.path.join(DATA_DIR, "tw_top50_by_mktcap.json")
URL = "https://www.taifex.com.tw/cht/9/futuresQADetail"
TTL_DAYS = 30


def _fetch_top50() -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(URL, headers=headers, timeout=20)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = []
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        for tr in trs:
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) >= 4:
                rank_str = str(tds[0]).strip()
                if re.match(r"^\d+$", rank_str):
                    rank = int(rank_str)
                    if rank > 50:
                        break

                    code = str(tds[1]).strip()
                    name = str(tds[2]).strip()
                    weight_pct = str(tds[3]).strip()

                    rows.append({
                        "rank": rank,
                        "code": code,
                        "name": name,
                        "weight_pct": weight_pct,
                    })
        if rows:
            break

    return rows[:50]


def ensure_top50_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    need_fetch = True

    if os.path.exists(JSON_PATH):
        mtime = os.path.getmtime(JSON_PATH)
        age_days = (time.time() - mtime) / 86400
        if age_days < TTL_DAYS:
            need_fetch = False

    if need_fetch:
        print("[top50_updater] 正在從期交所抓取前 50 名單…")
        try:
            rows = _fetch_top50()
            if rows:
                with open(JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
                print(f"[top50_updater] 已儲存 {len(rows)} 筆到 {JSON_PATH}")
            else:
                print("[top50_updater] 抓取結果為空，保留舊資料")
        except Exception as e:
            print(f"[top50_updater] 抓取失敗：{e}")

    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return []