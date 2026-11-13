"""Opt-in actual Chromium smoke test for the M1 workstation controls."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request


@unittest.skipUnless(os.environ.get("FXB_BROWSER_TESTS")=="1", "opt-in Chromium UI test")
class BrowserTests(unittest.TestCase):
    def test_m1_controls_render_without_javascript_errors(self):
        from playwright.sync_api import sync_playwright
        with tempfile.TemporaryDirectory() as temp, socket.socket() as sock:
            sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
            with (Path(temp)/"server.log").open("w") as log:
                process=subprocess.Popen([sys.executable,"-m","fxb.ui","--preview","--port",str(port)],
                    cwd=temp,stdout=log,stderr=log)
                try:
                    url=f"http://127.0.0.1:{port}/"
                    for _ in range(100):
                        try:
                            with urllib.request.urlopen(url,timeout=.3) as response:
                                if response.status==200:break
                        except OSError:time.sleep(.1)
                    else:self.fail("Preview did not start: "+(Path(temp)/"server.log").read_text())
                    with sync_playwright() as p:
                        browser=p.chromium.launch(headless=True)
                        page=browser.new_page(viewport={"width":1440,"height":1000})
                        errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        page.goto(url,wait_until="networkidle")
                        page.locator('[data-page="training"]').click()
                        self.assertTrue(page.locator("#scalp-form").is_visible())
                        self.assertTrue(page.locator("#scalp-prepare-button").is_disabled())
                        self.assertEqual(page.locator('#scalp-form [name="symbol"]').input_value(),"XAUUSD.m")
                        page.locator('[data-page="execution"]').click()
                        self.assertTrue(page.locator("#scalp-status").is_visible())
                        self.assertEqual(errors,[])
                        browser.close()
                finally:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=="__main__":unittest.main()
