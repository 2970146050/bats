import os

# 设置你的 tiny-imagenet-200 路径
DATA_DIR = './data/tiny-imagenet-200/tiny-imagenet-200'
val_dir = os.path.join(DATA_DIR, 'val')
img_dir = os.path.join(val_dir, 'images')
annot_file = os.path.join(val_dir, 'val_annotations.txt')

# 1. 打开标签文件
with open(annot_file, 'r') as f:
    data = f.readlines()

# 2. 建立类别字典
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]

# 3. 创建子文件夹并移动图片
for img, folder in val_img_dict.items():
    newpath = os.path.join(val_dir, folder)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    if os.path.exists(os.path.join(img_dir, img)):
        os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

print("Validation set reorganized successfully!")