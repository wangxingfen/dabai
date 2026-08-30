#!/usr/bin/env python3
"""
Mixamo 自动化批量下载工具（基于 Playwright + fq 代理加速）

流程：
  1. 首次运行：打开浏览器让用户手动登录 Mixamo，登录成功后保存 cookies
  2. 之后运行：自动加载 cookies，遍历动作列表，逐个搜索并下载 FBX
  3. 自动检测 fq 本地代理（clash/xray/hysteria 等），没有时可自动启动

用法：
  # 首次使用：登录并保存 cookies
  python mixamo_auto_downloader.py --login

  # 下载全部动作（自动检测代理）
  python mixamo_auto_downloader.py --download

  # 只下载某个分类
  python mixamo_auto_downloader.py --download --category idle

  # 只下载指定动作
  python mixamo_auto_downloader.py --download --names idle_normal,walk_happy

  # 下载到指定目录
  python mixamo_auto_downloader.py --download --output ./downloads

  # 指定代理协议
  python mixamo_auto_downloader.py --download -p clash

  # 查看当前 cookies 状态
  python mixamo_auto_downloader.py --status
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

SCRIPT_DIR = Path(__file__).parent
COOKIES_PATH = SCRIPT_DIR / ".." / "data" / "mixamo_cookies.json"
CONFIG_PATH = SCRIPT_DIR / ".." / "web" / "anim" / "animation-library.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / ".." / "data" / "mixamo_downloads"

MIXAMO_URL = "https://www.mixamo.com/"

# ========== fq 代理配置（与 yt_fq.py 保持一致） ==========
FQ_ROOT = r"D:\AI\Chrome141_AllNew_2025.10.3"

# 协议名 -> 本地代理地址/端口
PROXIES = {
    "clash":      {"server": "http://127.0.0.1:7890",  "port": 7890},
    "xray":       {"server": "socks5://127.0.0.1:1080", "port": 1080},
    "hysteria":   {"server": "socks5://127.0.0.1:1080", "port": 1080},
    "singbox":    {"server": "socks5://127.0.0.1:1080", "port": 1080},
    "naive":      {"server": "socks5://127.0.0.1:1080", "port": 1080},
    "hysteria2":  {"server": "socks5://127.0.0.1:1080", "port": 1080},
    "juicity":    {"server": "socks5://127.0.0.1:1080", "port": 1080},
    "mieru":      {"server": "socks5://127.0.0.1:3080", "port": 3080},
    "shadowquic": {"server": "socks5://127.0.0.1:4080", "port": 4080},
}

AUTO_START_CMD = ["fq.cmd", "start", "clash", "-NoChrome", "-NoElevate"]


def port_listening(port: int) -> bool:
    """检测本地端口是否在监听"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    r = s.connect_ex(("127.0.0.1", port))
    s.close()
    return r == 0


def _fq_start(proto: str):
    """通过 fq CLI 启动代理核心并等待端口就绪"""
    info = PROXIES[proto]
    print(f"[proxy] 无运行中的代理，自动执行: fq start {proto} -NoChrome ...")
    subprocess.run(AUTO_START_CMD, cwd=FQ_ROOT,
                   shell=(os.name != "nt"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(20):
        if port_listening(info["port"]):
            print(f"[proxy] {proto} 已就绪: {info['server']}")
            return
        time.sleep(1)
    sys.exit(f"[error] fq start {proto} 后端口 {info['port']} 仍未监听，请手动检查: fq status")


def ensure_proxy(proto=None, auto_start=True) -> str:
    """
    返回可用的本地代理地址。
    - 指定了 proto 就用指定协议，没运行就启动
    - 没指定就自动检测任一端口
    """
    if proto:
        info = PROXIES.get(proto.lower())
        if not info:
            sys.exit(f"[error] 未知协议 {proto}，可选: {', '.join(PROXIES)}")
        if port_listening(info["port"]):
            print(f"[proxy] 使用 {proto}: {info['server']}")
            return info["server"]
        if not auto_start:
            sys.exit(f"[error] {proto} 端口未监听，请先 fq start {proto} -NoChrome")
        _fq_start(proto)
        return info["server"]

    # 自动检测：任一端口在监听即可用
    for name, info in PROXIES.items():
        if port_listening(info["port"]):
            print(f"[proxy] 检测到运行中的代理 {name}: {info['server']}")
            return info["server"]

    if not auto_start:
        sys.exit("[error] 没有运行中的代理，请先 fq start <协议> -NoChrome")
    _fq_start("clash")
    return PROXIES["clash"]["server"]


def load_config() -> dict:
    """加载动作库配置"""
    if not CONFIG_PATH.exists():
        print(f"[错误] 配置文件不存在: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_animations(config, category=None, names=None):
    """获取要下载的动作列表"""
    result = []
    for cat_key, cat in config.get('categories', {}).items():
        if category and cat_key != category:
            continue
        for anim in cat.get('animations', []):
            if names and anim['name'] not in names:
                continue
            result.append({
                **anim,
                'category': cat_key,
                'category_label': cat.get('label', cat_key),
            })
    return result


async def save_cookies(context):
    """保存浏览器 cookies"""
    cookies = await context.cookies()
    COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COOKIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(cookies, f, ensure_ascii=False, indent=2)
    print(f"[✓] Cookies 已保存到: {COOKIES_PATH}")


async def load_cookies(context):
    """加载 cookies 到浏览器上下文"""
    if not COOKIES_PATH.exists():
        return False
    with open(COOKIES_PATH, 'r', encoding='utf-8') as f:
        cookies = json.load(f)
    await context.add_cookies(cookies)
    return True


async def check_login(page) -> bool:
    """检查是否已登录 Mixamo"""
    try:
        await page.goto(MIXAMO_URL, wait_until="domcontentloaded", timeout=15000)
        # 等待页面加载
        await page.wait_for_timeout(3000)
        # 检查是否有用户头像/登出按钮（登录后会显示）
        # Mixamo 登录后右上角会显示用户信息
        try:
            # 尝试找"Sign In"按钮 — 存在说明未登录
            sign_in = await page.query_selector("text=Sign In")
            if sign_in:
                return False
        except:
            pass
        # 尝试找用户头像或菜单
        try:
            user_menu = await page.query_selector("[data-testid='user-menu'], .user-menu, .avatar")
            if user_menu:
                return True
        except:
            pass
        # 如果页面上没有 Sign In，认为已登录
        try:
            sign_in_btn = await page.get_by_text("Sign In", exact=False).count()
            if sign_in_btn > 0:
                return False
        except:
            pass
        return True
    except Exception as e:
        print(f"[!] 检查登录状态出错: {e}")
        return False


async def login_flow(proxy_server=None, headless=False):
    """交互式登录流程：打开浏览器让用户手动登录，成功后保存 cookies"""
    print("\n" + "=" * 60)
    print("  Mixamo 登录")
    print("=" * 60)
    print("\n  即将打开浏览器，请在打开的页面中：")
    print("  1. 点击右上角 Sign In 登录你的 Adobe 账号")
    print("  2. 登录成功后跳回 Mixamo 主页")
    print("  3. 确认右上角显示你的头像/用户名")
    print("  4. 回到终端按 Enter 继续\n")

    launch_kwargs = {"headless": headless}
    if proxy_server:
        launch_kwargs["proxy"] = {"server": proxy_server}

    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_kwargs)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        await page.goto(MIXAMO_URL, wait_until="domcontentloaded")
        print("  浏览器已打开，请完成登录...")
        input("  登录完成后，按 Enter 继续...")

        # 检查登录状态
        is_logged = await check_login(page)
        if is_logged:
            await save_cookies(context)
            print("\n[✓] 登录成功，Cookies 已保存！")
        else:
            print("\n[!] 似乎还没有登录成功，请确认右上角是否显示用户信息")
            print("[!] 你可以重新运行 --login 再试一次")

        await browser.close()


async def download_animation(page, anim, output_dir):
    """
    在 Mixamo 页面上搜索并下载单个动作
    返回 (成功?, 文件路径)
    """
    name = anim['name']
    search_term = Path(anim['file']).stem.replace('_', ' ')
    target_path = output_dir / anim['file']

    # 确保分类目录存在
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果已存在，跳过
    if target_path.exists():
        print(f"  ≈ {name}  已存在，跳过")
        return True, str(target_path)

    print(f"  → 下载: {name} (搜索: {search_term})")

    try:
        # 跳转到搜索页
        search_url = f"https://www.mixamo.com/#/?query={search_term.replace(' ', '+')}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(2000)

        # 点击第一个搜索结果（卡片）
        # Mixamo 的动作卡片选择器
        card_selectors = [
            ".item-card:first-child",
            ".search-results .item:first-child",
            "[data-testid*='card']:first-child",
            ".thumb-list-item:first-child",
        ]

        card_clicked = False
        for selector in card_selectors:
            try:
                card = await page.query_selector(selector)
                if card:
                    await card.click()
                    card_clicked = True
                    break
            except:
                continue

        if not card_clicked:
            # 尝试用文本搜索点击
            try:
                # 点击第一个缩略图/预览
                await page.locator(".item-card").first.click(timeout=5000)
                card_clicked = True
            except Exception as e:
                print(f"    ✗ 找不到动作卡片: {e}")
                return False, None

        await page.wait_for_timeout(1500)

        # 点击 Download 按钮
        download_selectors = [
            "button:has-text('Download')",
            ".download-button",
            "[data-testid='download-button']",
            "#download-btn",
        ]

        download_btn = None
        for selector in download_selectors:
            try:
                btn = await page.query_selector(selector)
                if btn and await btn.is_visible():
                    download_btn = btn
                    break
            except:
                continue

        if not download_btn:
            print(f"    ✗ 找不到 Download 按钮")
            return False, None

        # 开始监听下载
        async with page.expect_download(timeout=30000) as download_info:
            await download_btn.click()

        download = await download_info.value

        # 保存到目标路径
        await download.save_as(str(target_path))
        print(f"    ✓ 保存到: {anim['file']}")
        return True, str(target_path)

    except PlaywrightTimeoutError:
        print(f"    ✗ 超时")
        return False, None
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        return False, None


async def download_all(category=None, names=None, output_dir=None,
                        proxy_server=None, headless=False):
    """批量下载所有动作"""
    config = load_config()
    animations = get_animations(config, category, names)

    if not animations:
        print("[!] 没有找到要下载的动作")
        return

    output_dir = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  批量下载 Mixamo 动作")
    print(f"  数量: {len(animations)} 个")
    print(f"  输出目录: {output_dir}")
    if proxy_server:
        print(f"  代理: {proxy_server}")
    print("=" * 60 + "\n")

    # 检查 cookies
    if not COOKIES_PATH.exists():
        print("[!] 未找到 cookies，请先运行 --login 登录")
        sys.exit(1)

    success = 0
    failed = 0
    skipped = 0
    failed_list = []

    launch_kwargs = {"headless": headless}
    if proxy_server:
        launch_kwargs["proxy"] = {"server": proxy_server}

    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_kwargs)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            accept_downloads=True,
        )

        # 加载 cookies
        await load_cookies(context)
        page = await context.new_page()

        # 验证登录
        print("  验证登录状态...")
        is_logged = await check_login(page)
        if not is_logged:
            print("[!] Cookies 已失效，请重新运行 --login")
            await browser.close()
            sys.exit(1)
        print("  ✓ 登录有效\n")

        # 逐个下载
        start_time = time.time()
        for i, anim in enumerate(animations, 1):
            print(f"  [{i}/{len(animations)}]", end=" ")
            ok, path = await download_animation(page, anim, output_dir)
            if ok:
                if path and (output_dir / anim['file']).exists():
                    success += 1
                else:
                    skipped += 1
            else:
                failed += 1
                failed_list.append(anim['name'])

            # 每个动作之间稍作间隔，避免被限流
            if i < len(animations):
                await page.wait_for_timeout(1000)

        elapsed = time.time() - start_time
        # 保存最新 cookies
        await save_cookies(context)

        await browser.close()

    print(f"\n{'─' * 60}")
    print(f"  完成!  成功: {success}  跳过: {skipped}  失败: {failed}")
    print(f"  用时: {elapsed:.1f} 秒")
    if failed_list:
        print(f"\n  失败的动作:")
        for name in failed_list:
            print(f"    - {name}")
    print(f"{'─' * 60}\n")


async def check_status():
    """检查当前状态"""
    print("\n" + "=" * 60)
    print("  Mixamo 下载器状态")
    print("=" * 60 + "\n")

    # Cookies 状态
    if COOKIES_PATH.exists():
        with open(COOKIES_PATH, 'r', encoding='utf-8') as f:
            cookies = json.load(f)
        # 找 Mixamo 相关的 cookie
        mixamo_cookies = [c for c in cookies if 'mixamo' in c.get('domain', '').lower() or 'adobe' in c.get('domain', '').lower()]
        print(f"  ✓ Cookies 已保存 ({len(cookies)} 个, 其中 Mixamo/Adobe 相关 {len(mixamo_cookies)} 个)")
        print(f"    路径: {COOKIES_PATH}")
    else:
        print(f"  ✗ 未找到 Cookies，请先运行 --login")

    # 配置状态
    config = load_config()
    total = sum(len(cat['animations']) for cat in config['categories'].values())
    print(f"\n  ✓ 动作配置: {total} 个动作")

    # 已下载状态
    print(f"\n  下载目录: {DEFAULT_OUTPUT_DIR}")
    if DEFAULT_OUTPUT_DIR.exists():
        downloaded = list(DEFAULT_OUTPUT_DIR.rglob("*.fbx"))
        print(f"  已下载文件: {len(downloaded)} 个")
    else:
        print(f"  下载目录不存在")

    print(f"\n{'─' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Mixamo 自动化批量下载工具（Playwright + fq 代理加速）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 首次使用：打开浏览器登录并保存 cookies（自动检测代理）
  python mixamo_auto_downloader.py --login

  # 下载全部动作（自动检测 fq 代理）
  python mixamo_auto_downloader.py --download

  # 只下载 idle 分类
  python mixamo_auto_downloader.py --download --category idle

  # 只下载指定动作
  python mixamo_auto_downloader.py --download --names idle_normal,walk_happy,clap

  # 指定输出目录
  python mixamo_auto_downloader.py --download --output ./my_downloads

  # 指定代理协议 (clash/xray/hysteria/singbox/...)
  python mixamo_auto_downloader.py --download -p clash

  # 不自动启动代理，没代理直接报错
  python mixamo_auto_downloader.py --download --no-proxy-start

  # 查看状态
  python mixamo_auto_downloader.py --status
        """
    )
    parser.add_argument('--login', action='store_true', help='交互式登录并保存 cookies')
    parser.add_argument('--download', action='store_true', help='批量下载动作')
    parser.add_argument('--status', action='store_true', help='查看当前状态')
    parser.add_argument('--category', type=str, help='只下载指定分类 (idle/gesture/emotion/walk/dance/pose)')
    parser.add_argument('--names', type=str, help='只下载指定动作名，逗号分隔')
    parser.add_argument('--output', type=str, help='下载输出目录')
    parser.add_argument('-p', '--proto', default=None,
                        help=f'指定代理协议 ({"/".join(PROXIES)})，默认自动检测监听中的端口')
    parser.add_argument('--no-proxy', action='store_true',
                        help='不使用代理，直连 Mixamo（国内可能很慢）')
    parser.add_argument('--no-proxy-start', action='store_true',
                        help='没有代理时不自动 fq start，直接报错退出')
    parser.add_argument('--headless', action='store_true', help='无头模式运行浏览器')

    args = parser.parse_args()

    # 检查 playwright 是否已安装
    try:
        import playwright
    except ImportError:
        print("[错误] 未安装 playwright，请先安装:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)

    # 确定代理（只有登录和下载时才需要）
    proxy_server = None
    if not args.no_proxy and (args.login or args.download):
        proxy_server = ensure_proxy(args.proto, auto_start=not args.no_proxy_start)

    if args.login:
        asyncio.run(login_flow(proxy_server=proxy_server, headless=args.headless))
    elif args.download:
        names = [n.strip() for n in args.names.split(',')] if args.names else None
        asyncio.run(download_all(
            category=args.category,
            names=names,
            output_dir=args.output,
            proxy_server=proxy_server,
            headless=args.headless,
        ))
    elif args.status:
        asyncio.run(check_status())
    else:
        parser.print_help()
        print("\n[提示] 未指定操作，默认显示状态\n")
        asyncio.run(check_status())


if __name__ == '__main__':
    main()
