from pathlib import Path

ICON_DIR = Path("static/icons")

# Generated icons are checked into the project. Avoid importing Pillow (which can
# be slow on a freshly activated Windows/OneDrive environment) when no refresh is
# necessary. Delete either icon to force regeneration.
if all((ICON_DIR / f"icon-{size}.png").exists() for size in (192, 512)):
    print("PWA icons already exist; nothing to generate.")
    raise SystemExit(0)

from PIL import Image, ImageDraw
for size in (192,512):
    im=Image.new("RGB",(size,size)); p=im.load()
    for y in range(size):
        for x in range(size):
            t=(x+y)/(2*size); p[x,y]=(int(108+(162-108)*t),int(92+(155-92)*t),int(231+(254-231)*t))
    d=ImageDraw.Draw(im); m=size//5; d.rounded_rectangle((m,m,size-m,size-m),radius=size//7,outline="white",width=size//22)
    d.line((size*.34,size*.52,size*.46,size*.64,size*.69,size*.38),fill="white",width=size//18,joint="curve")
    ICON_DIR.mkdir(parents=True,exist_ok=True); im.save(ICON_DIR / f"icon-{size}.png")
