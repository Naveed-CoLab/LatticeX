"""Screenshots of the real dashboard for the README (docs/assets/screen_*.png), in light and dark.

    python -m scripts.readme_shots                                    # the dashboard running on this PC (port 8600)
    python -m scripts.readme_shots --url https://3-27-118-9.sslip.io  # the hosted site
    python -m scripts.readme_shots --ticker AAPL                      # another stock for the stock-page shots

Needs Playwright, once: pip install playwright (not in requirements.txt: only this script uses it). It uses the
Microsoft Edge or Chrome already on the computer; `python -m playwright install chromium` is needed only if neither
is installed. With a login on the site, add --password.

Writes, each as _light.png and _dark.png:
  screen_stock      a stock page at 1440 x 900 (price, rank, likely ranges, chart)
  screen_overview   the Overview page at 1440 x 900
  screen_record     the Track record page at 1440 x 900
  screen_phone      the stock page on a phone (390 x 844)
"""
import argparse
import sys

from src.config import ROOT

SHOTS = [  # name, address after #, viewport, pixel ratio, what must be on screen before the picture
    ("screen_stock", "{t}/20", (1440, 900), 1, "#fc .fcard, #fc > *"),
    ("screen_overview", "overview", (1440, 900), 1, "#overview > *"),
    ("screen_record", "record", (1440, 900), 1, "main:not([hidden]) table"),
    ("screen_phone", "{t}/20", (390, 844), 2, "#fc .fcard, #fc > *"),
]

LOADED = "() => !document.querySelector('.skel, .stage-skel, .side-skel')"
LOGOS = """() => [...document.images].filter(i => { const r = i.getBoundingClientRect();
  return r.bottom > 0 && r.top < innerHeight && r.width > 0; }).every(i => i.complete)"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600", help="address of a running dashboard")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--password", default="", help="only if the site asks for one")
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    a = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("Playwright is missing: pip install playwright && python -m playwright install chromium")
    base = a.url.rstrip("/") + "/#"
    auth = {"username": "readme", "password": a.password} if a.password else None
    with sync_playwright() as p:
        browser = None
        for channel in ("msedge", "chrome", None):                   # browsers already installed first
            try:
                browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except Exception:
                continue
        if browser is None:
            sys.exit("No browser found: install Microsoft Edge or Chrome, or run: python -m playwright install chromium")
        for theme in ("light", "dark"):
            for name, frag, (w, h), dpr, ready in SHOTS:
                ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=dpr,
                                          color_scheme=theme, http_credentials=auth, reduced_motion="reduce")
                page = ctx.new_page()
                page.goto(base + frag.format(t=a.ticker), wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_selector(ready, timeout=60_000)
                page.wait_for_function(LOADED, timeout=60_000)
                try:                                               # logos on screen: wait, but not forever
                    page.wait_for_function(LOGOS, timeout=20_000)
                except Exception:
                    print(f"  ({name}, {theme}: some logos did not load; letter tiles show instead)")
                page.wait_for_timeout(2500)                        # chart drawing and the cone's opening animation
                path = f"{a.out}/{name}_{theme}.png"
                page.screenshot(path=path)
                print(f"  {path}")
                ctx.close()
        browser.close()
    print("Done. Check the pictures, then commit docs/assets/screen_*.png.")


if __name__ == "__main__":
    main()
