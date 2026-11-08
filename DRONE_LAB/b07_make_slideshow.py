# b07_make_slideshow.py
# data/drone_cls 의 사진을 2초마다 랜덤으로 보여주는 HTML 을 만든다 (맥 데모용)
# 실행 : python b07_make_slideshow.py  → 브라우저가 열림
# 조작 : f 전체화면 / space 일시정지 / → 다음 사진 / l 정답 표시 켜기·끄기

import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "drone_cls"
HTML_FILE = SCRIPT_DIR / "b07_slideshow.html"

INTERVAL_MS = 2000

items = []
for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    for image_file in sorted(class_dir.glob("*.jpg")):
        items.append({
            "src": image_file.relative_to(SCRIPT_DIR).as_posix(),
            "label": class_dir.name,
        })

print("사진 수:", len(items))

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Drone Slideshow</title>
<style>
  html, body {{ margin:0; height:100%; background:#000; overflow:hidden; }}
  img {{ position:absolute; inset:0; margin:auto; height:90vh; width:auto;
        max-width:100vw; object-fit:contain; }}
  #label {{ position:fixed; left:12px; bottom:8px; color:#888;
           font:14px sans-serif; display:none; }}
  #hint {{ position:fixed; right:12px; bottom:8px; color:#444; font:12px sans-serif; }}
</style></head>
<body>
<img id="img">
<div id="label"></div>
<div id="hint">f 전체화면 · space 정지 · → 다음 · l 정답</div>
<script>
const items = {json.dumps(items)};
const img = document.getElementById("img");
const label = document.getElementById("label");
let paused = false, timer = null, count = 0;

function next() {{
  const it = items[Math.floor(Math.random() * items.length)];
  img.src = it.src;
  count += 1;
  label.textContent = count + " : " + it.label;
  console.log(count, it.label, it.src);
}}
function start() {{ clearInterval(timer); timer = setInterval(next, {INTERVAL_MS}); }}

document.addEventListener("keydown", e => {{
  if (e.key === "f") document.documentElement.requestFullscreen();
  if (e.key === " ") {{ paused = !paused; paused ? clearInterval(timer) : start(); }}
  if (e.key === "ArrowRight") {{ next(); if (!paused) start(); }}
  if (e.key === "l") label.style.display = label.style.display === "block" ? "none" : "block";
}});

next(); start();
</script></body></html>
"""

HTML_FILE.write_text(html, encoding="utf-8")
print("저장:", HTML_FILE)

subprocess.run(["open", str(HTML_FILE)])
