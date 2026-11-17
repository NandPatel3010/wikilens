import argparse
import re
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright


def record(base_url, output):
    frames = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 760}, reduced_motion="reduce")
        page.goto(base_url, wait_until="networkidle")
        query = page.get_by_role("textbox", name="Wikipedia search query")
        expect(query).to_be_visible()
        frames.append(page.screenshot())
        query.fill("Why is the sky blue?")
        frames.append(page.screenshot())
        page.locator("form").get_by_role("button", name="Search", exact=True).click()
        expect(page.locator(".result-head h2").first).to_be_visible(timeout=30000)
        frames.append(page.screenshot())
        page.get_by_role("radio", name=re.compile("^Semantic ")).check()
        expect(page.locator(".result-head h2").first).to_be_visible(timeout=30000)
        frames.append(page.screenshot())
        page.get_by_role("button", name="Benchmark").click()
        expect(page.get_by_role("table")).to_be_visible(timeout=30000)
        frames.append(page.screenshot())
        browser.close()

    images = [Image.open(BytesIO(frame)).convert("RGB") for frame in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], loop=0,
                   duration=[1100, 850, 2200, 1600, 2600])
    print(f"Recorded {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record and verify the WikiLens search flow.")
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--output", type=Path, default=Path("benchmark/demo.gif"))
    args = parser.parse_args()
    record(args.url, args.output)
