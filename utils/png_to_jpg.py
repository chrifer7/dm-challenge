import os
import sys
import argparse
from PIL import Image


def convert():
  for filename in os.listdir(args.dir):
    im = Image.open(args.dir+'/'+filename)
    im.load() # required for png.split()

    background = Image.new("RGB", im.size, (255, 255, 255))
    print('Channels: ',size(im.split()))
    background.paste(im, mask=im.split()[3]) # 3 is the alpha channel

    background.save(args.dir+'/'+filename.split('.')[0]+'.jpg', 'JPEG', quality=90)
    #rgb_im = im.convert('RGB')
    #rgb_im.save(args.dir+'/'+filename.split('.')[0]+'.jpg')



if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--dir")
  
  args = a.parse_args()
  if args.dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.dir)):
    print("directories do not exist")
    sys.exit(1)
  
  convert()


