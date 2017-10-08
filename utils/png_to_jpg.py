import os
import sys
import argparse
from PIL import Image, ImageMath

def convert0():
  for filename in os.listdir(args.dir):
    im = Image.open(args.dir+'/'+filename)
    rgb_im = im.convert('RGB')
    rgb_im.save(args.dir+'/'+filename.split('.')[0]+'.jpg')

    if (args.replace):
    	os.remove(args.dir+'/'+filename)


def convert1():
  for filename in os.listdir(args.dir):
    im = Image.open(args.dir+'/'+filename).convert('RGBA')
    #im.show()
    background = Image.new('RGBA', im.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, im)
    alpha_composite.save(args.dir+'/'+filename.split('.')[0]+'.jpg', 'JPEG', quality=95)

    if (args.replace):
    	os.remove(args.dir+'/'+filename)


def convert2():
  for filename in os.listdir(args.dir):
    im = Image.open(args.dir+'/'+filename)
    im.load()
    
    background = Image.new('RGB', im.size, (255, 255, 255))
    background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
    background.save(args.dir+'/'+filename.split('.')[0]+'.jpg', 'JPEG', quality=95)

    if (args.replace):
    	os.remove(args.dir+'/'+filename)


def convert3():
  for filename in os.listdir(args.dir):
    im = Image.open(args.dir+'/'+filename)
    im2 = ImageMath.eval('im/256', {'im':im}).convert('L')
    im2.save(args.dir+'/'+filename.split('.')[0]+'.jpg', 'JPEG', quality=95)
    
    if (args.replace):
    	os.remove(args.dir+'/'+filename)
    

if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--dir")
  a.add_argument("--replace")  
  
  args = a.parse_args()
  if args.dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.dir)):
    print("directories do not exist")
    sys.exit(1)
    
  convert3()



