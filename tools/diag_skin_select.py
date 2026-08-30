"""诊断：Mixamo 下载面板 Skin 下拉框定位 + Without Skin 选择是否真正生效。
用法: python tools/diag_skin_select.py
"""
import asyncio
import json
import os

from playwright.async_api import async_playwright

PROXY = "http://127.0.0.1:7890"
COOKIES = r"D:\AI\dabai\data\mixamo_cookies.json"
SHOT = r"D:\AI\dabai\tools\diag_skin_select.png"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, proxy={"server": PROXY})
        ctx = await browser.new_context()
        if os.path.exists(COOKIES):
            with open(COOKIES, encoding="utf-8") as f:
                await ctx.add_cookies(json.load(f))
            print("loaded cookies")
        page = await ctx.new_page()
        await page.goto("https://www.mixamo.com/#/?query=Idle", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(9000)
        # 等待动作卡片出现（SPA 加载）
        try:
            await page.wait_for_selector(".product.product-animation", timeout=20000)
        except Exception as e:
            print("wait cards err", str(e)[:100])
        cards = await page.evaluate("document.querySelectorAll('.product.product-animation').length")
        print("CARDS:", cards)
        if cards == 0:
            print("未登录或无动作，退出")
            await browser.close()
            return

        # 点第一个动作 + DOWNLOAD 打开下载面板
        await page.locator(".product.product-animation").first.click(timeout=6000)
        await page.wait_for_timeout(2500)
        try:
            btn = page.locator("button.btn-block.btn.btn-primary").filter(has_text="DOWNLOAD").first
            await btn.click(timeout=6000)
        except Exception as e:
            print("dl click err", str(e)[:100])
        await page.wait_for_timeout(2500)

        # 1) dump 所有 select 的 label + options
        sel_info = await page.evaluate(
            """() => {
            const out=[];
            document.querySelectorAll('select').forEach((s,i)=>{
              if(!(s.offsetParent||s.getClientRects().length)) return;
              const label=(s.closest('label')?.innerText||s.previousElementSibling?.innerText||s.parentElement?.innerText||'').trim();
              out.push({i, label:label.slice(0,60), val:s.value, opts:Array.from(s.options).map(o=>o.textContent.trim()).slice(0,8)});
            });
            return out;
            }"""
        )
        print("SELECTS:", json.dumps(sel_info, ensure_ascii=False))

        # 2) 测试我的定位逻辑：找 Skin select 索引
        idx = await page.evaluate(
            """() => {
            const all = Array.from(document.querySelectorAll('select'));
            for (let i=0;i<all.length;i++){
              const s=all[i];
              const label=(s.closest('label')?.innerText||s.previousElementSibling?.innerText||s.parentElement?.innerText||'').trim();
              const opts=Array.from(s.options).map(o=>o.textContent.trim());
              if(/skin/i.test(label) || opts.some(o=>/without skin/i.test(o))) return i;
            }
            return -1;
            }"""
        )
        print("SKIN_SELECT_INDEX:", idx)

        # 3) 用 Playwright select_option 选择 Without Skin
        if idx >= 0:
            sel = page.locator("select").nth(idx)
            try:
                await sel.select_option(label="Without Skin", timeout=3000)
                print("select_option(label=Without Skin) OK")
            except Exception as e:
                print("select_option label err:", str(e)[:100])
                try:
                    await sel.select_option(index=1, timeout=3000)
                    print("select_option(index=1) OK")
                except Exception as e2:
                    print("select_option index err:", str(e2)[:100])
            await page.wait_for_timeout(800)
            # 验证选择是否生效
            after = await page.evaluate(
                """() => {
                const all = Array.from(document.querySelectorAll('select'));
                const out=[];
                all.forEach((s,i)=>{
                  if(!(s.offsetParent||s.getClientRects().length)) return;
                  out.push({i, val:s.value, selText:s.options[s.selectedIndex]?.textContent.trim()});
                });
                return out;
                }"""
            )
            print("AFTER_SELECT:", json.dumps(after, ensure_ascii=False))

        await page.screenshot(path=SHOT)
        print("SHOT:", SHOT)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
