"""用 Playwright 直接连接正在运行的 Mixamo 浏览器，诊断登录态与下载面板。
用法: python tools/diag_mixamo.py
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from playwright.async_api import async_playwright

WS_ENDPOINT = "http://127.0.0.1:9222"  # 占位，实际用服务暴露的调试端口


async def main():
    from mixamo_download_service import service
    # 直接复用服务里的浏览器对象（同进程内）
    if not service.browser:
        print("NO_BROWSER_OBJ")
        return
    ctx = service.context
    page = service.page
    print("URL:", page.url)
    # 1. 登录态：看页面是否有 Log In / Sign Up
    txt = await page.evaluate("document.body.innerText")
    has_login = "Log In" in txt or "Sign In" in txt or "Sign Up" in txt
    print("HAS_LOGIN_BTN:", has_login)
    # 2. 动作网格卡片数
    cards = await page.evaluate("document.querySelectorAll('.product.product-animation').length")
    print("CARDS:", cards)
    # 3. 当前所有 select 及选项（含 with/without skin）
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
    # 4. 所有含 skin/without 的可见文本
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
    # 5. 截图
    shot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_mixamo.png")
    await page.screenshot(path=shot, full_page=False)
    print("SHOT:", shot)


if __name__ == "__main__":
    asyncio.run(main())