import base64, glob, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(BASE_DIR, 'output', 'embedded_images.html')
imgs = sorted(glob.glob(os.path.join(BASE_DIR, 'output', '*.png')))
html = ['<!doctype html>', '<html lang="en">', '<head>', '<meta charset="utf-8">', '<meta name="viewport" content="width=device-width, initial-scale=1">', '<title>Embedded Images - Supermart</title>', '<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} .card{border:1px solid #ddd;padding:12px;border-radius:6px;margin-bottom:16px;} img{max-width:100%;height:auto}</style>', '</head>', '<body>', '<h1>Embedded Plots Gallery</h1>']
for p in imgs:
    name = os.path.basename(p)
    with open(p, 'rb') as f:
        b = base64.b64encode(f.read()).decode('utf-8')
    html.append(f'<div class="card"><h3>{name}</h3>')
    html.append(f'<img src="data:image/png;base64,{b}" alt="{name}"></div>')
html.append('</body></html>')
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))
print('WROTE', OUT)