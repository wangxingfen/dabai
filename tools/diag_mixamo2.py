"""独立 Playwright 诊断：走 clash 代理打开 Mixamo，看登录态 + 下载面板的 without skin 控件。
用法: python tools/diag_mixamo2.py
"""
import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

PROXY = "http://127.0.0.1:7890"
COOKIES = r"D:\AI\dabai\data\mixamo_cookies.json"
SHOT = r"D:\AI\dabai\tools\diag_mixamo2.png"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, proxy={"server": PROXY})
        ctx = await browser.new_context()
        if os.path.exists(COOKIES):
            try:
                with open(COOKIES, encoding="utf-8") as f:
                    cks = json.load(f)
                await ctx.add_cookies(cks)
                print("loaded", len(cks), "cookies")
            except Exception as e:
                print("cookie load err", e)
        page = await ctx.new_page()
        await page.goto("https://www.mixamo.com/#/?query=Walking", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(6000)
        print("URL:", page.url)
        txt = await page.evaluate("document.body.innerText")
        print("HAS_LOGIN:", ("Log In" in txt) or ("Sign In" in txt) or ("Sign Up" in txt))
        cards = await page.evaluate("document.querySelectorAll('.product.product-animation').length")
        print("CARDS:", cards)
        # 若已登录：点第一个动作 + 点 DOWNLOAD，抓面板
        if cards > 0:
            await page.locator(".product.product-animation").first.click(timeout=6000)
            await page.wait_for_timeout(2500)
            try:
                btn = page.locator("button.btn-block.btn.btn-primary").filter(has_text="DOWNLOAD").first
                await btn.click(timeout=6000)
            except Exception as e:
                print("dl click err", str(e)[:100])
            await page.wait_for_timeout(2500)
            sel = await page.evaluate(
                """() => {
                const out=[];
                document.querySelectorAll('select').forEach(s=>{
                  if(!(s.offsetParent||s.getClientRects().length)) return;
                  out.push({val:s.value, opts:Array.from(s.options).map(o=>o.textContent.trim()).slice(0,10)});
                });
                return out;
                }"""
            )
            print("SELECTS:", json.dumps(sel, ensure_ascii=False))
            hits = await page.evaluate(
                """() => {
                const out=[];
                document.querySelectorAll('body *').forEach(el=>{
                  if(el.children.length) return;
                  const t=(el.innerText||el.textContent||'').trim();
                  if(t && t.length<80 && /skin|without|with|skeleton|rig|mesh/i.test(t)) out.push(t);
                });
                return [...new Set(out)].slice(0,30);
                }"""
            )
            print("SKIN_HITS:", json.dumps(hits, ensure_ascii=False))
        await page.screenshot(path=SHOT)
        print("SHOT:", SHOT)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())