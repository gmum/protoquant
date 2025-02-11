import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def crop_dataset(raw_data_dir: Path, proc_data_dir: Path):
    images_file_path = raw_data_dir/'images.txt'
    split_file_path = raw_data_dir/'train_test_split.txt'
    bbox_file_path = raw_data_dir/'bounding_boxes.txt'
    
    train_save_path = proc_data_dir/'dataset/train'
    test_save_path = proc_data_dir/'dataset/test'

    with open(images_file_path, 'r') as f:
        images = [line.strip('\n').split(',') for line in f]

    with open(split_file_path, 'r') as f:
        split = [line.strip('\n').split(',') for line in f]

    with open(bbox_file_path, 'r') as f:
        bboxes = {int(id): (x, y, w, h) for id, x, y, w, h in (map(float, line.split()) for line in f)}

    for idx in tqdm(range(len(images))):
        id, fn = images[idx][0].split(' ')
        id = int(id)
        file_name = fn.split('/')[0]
        
        if int(split[idx][0][-1]) == 1:
            fold_save_path = train_save_path
        else:
            fold_save_path = test_save_path
            
        if not os.path.isdir(fold_save_path/file_name):
            os.makedirs(fold_save_path/file_name)
            
        # Load image;
        img_fname = images[idx][0].split(' ')[1]
        img = Image.open(raw_data_dir/'images'/img_fname).convert('RGB')
        img = img.convert('RGB')

        # Crop image;
        x, y, w, h = bboxes[id]
        crop = img.crop((x, y, x+w, y+h))
        crop_fname = img_fname.split('/')[1]
        crop.save(fold_save_path/file_name/crop_fname)

