from pathlib import Path
from PIL import Image, ImageDraw
for size in (192,512):
    im=Image.new("RGB",(size,size)); p=im.load()
    for y in range(size):
        for x in range(size):
            t=(x+y)/(2*size); p[x,y]=(int(108+(162-108)*t),int(92+(155-92)*t),int(231+(254-231)*t))
    d=ImageDraw.Draw(im); m=size//5; d.rounded_rectangle((m,m,size-m,size-m),radius=size//7,outline="white",width=size//22)
    d.line((size*.34,size*.52,size*.46,size*.64,size*.69,size*.38),fill="white",width=size//18,joint="curve")
    Path("static/icons").mkdir(parents=True,exist_ok=True); im.save(f"static/icons/icon-{size}.png")
