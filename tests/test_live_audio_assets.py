"""The actual application static mount must serve every live microphone module."""
import ast
import asyncio
from pathlib import Path
import re
from urllib.parse import urljoin

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
from jinja2 import Environment, FileSystemLoader


def test_rendered_live_pages_and_worklet_modules_resolve_through_application_mount():
    async def run():
        root = Path(__file__).resolve().parents[1]
        application = FastAPI()
        # Execute the real mount declaration, without starting the legacy server-side
        # microphone/TTS stack imported by main.py. A mount-path change breaks this test.
        parsed = ast.parse((root / "app/main.py").read_text())
        mounts = [node for node in parsed.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute)
                  and node.value.func.attr == "mount"]
        assert mounts, "Application static mount is missing"
        namespace = {"app": application, "StaticFiles": StaticFiles}
        for mount in mounts:
            exec(compile(ast.Module(body=[mount], type_ignores=[]), "app/main.py", "exec"), namespace)

        templates = Environment(loader=FileSystemLoader(root / "app/templates"), autoescape=True)

        @application.get("/call", response_class=HTMLResponse)
        async def call():
            return templates.get_template("elevenlabs_call.html").render(characters=[], voices=[])

        @application.get("/live-stt", response_class=HTMLResponse)
        async def live_stt():
            return (root / "app/templates/live_stt.html").read_text()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
            modules = set()
            for route in ("/call", "/live-stt"):
                response = await client.get(route)
                assert response.status_code == 200
                urls = re.findall(r'<script\s+src="([^"]+)"', response.text)
                assert any(url.endswith("/live_audio.js") for url in urls)
                for url in urls:
                    script = await client.get(url)
                    assert script.status_code == 200, f"{route} references unserved script {url}"
                    assert "javascript" in script.headers["content-type"]
                    modules.add(url)
                    for worklet in re.findall(r"addModule\(['\"]([^'\"]+)['\"]", script.text):
                        worker = await client.get(worklet)
                        assert worker.status_code == 200, f"Worklet is unserved: {worklet}"
                        for dependency in re.findall(r"import\s+['\"]([^'\"]+)['\"]", worker.text):
                            imported = await client.get(urljoin(worklet, dependency))
                            assert imported.status_code == 200, f"Worklet dependency is unserved: {dependency}"
            assert modules

    asyncio.run(run())
