"""Test Hyperliquid Forex symbols"""
import sys, io, time, requests
from datetime import datetime, timezone
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HL_URL = "https://api.hyperliquid.xyz/info"
H = {"Content-Type": "application/json"}

test_names = [
    "xyz:EUR",
    "xyz:GBP",
    "xyz:JPY",
    "xyz:USDJPY",
    "xyz:CHF",
]

print("=" * 60)
print("HYPERLIQUID - Direct candle test for Forex (xyz namespace) variations")
print("=" * 60)
end_ms = int(time.time() * 1000)
start_ms = end_ms - (5 * 24 * 3600 * 1000) # 5 days ago

for coin in test_names:
    try:
        p = {"type":"candleSnapshot","req":{"coin":coin,"interval":"1h","startTime":start_ms,"endTime":end_ms}}
        d = requests.post(HL_URL, json=p, headers=H, timeout=15).json()
        if isinstance(d, list) and d:
            c = d[-1]
            ts = datetime.fromtimestamp(c["t"]/1000, tz=timezone.utc)
            print(f"  OK {coin:18s} bars={len(d):4d} last={ts:%Y-%m-%d %H:%M} O={c['o']:>8} H={c['h']:>8} L={c['l']:>8} C={c['c']:>8}")
    except Exception as e:
        pass
