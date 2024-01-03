import os
import random
from os import getcwd
from utils.utils import get_classes

# -------------------------------------------------------------------#
#   classes_path    指向model_data下的txt，与自己训练的数据集相关
#                   训练前一定要修改classes_path，使其对应自己的数据集
#                   txt文件中是自己所要去区分的种类
#                   与训练和预测所用的classes_path一致即可
# -------------------------------------------------------------------#
classes_path = 'model_data/cls_classes.txt'
# -------------------------------------------------------#
#   datasets_path   指向数据集所在的路径
# -------------------------------------------------------#
datasets_path = 'datasets'

sets = ["train", "test"]
classes, _ = get_classes(classes_path)

# 下面可以改动
intercept_path = "..\\features\\intercept"
slope_path = "..\\features\\slope"
pixel_path = "..\\features\\pixel"

if __name__ == "__main__":
    for se in sets:
        list_file = open('cls_' + se + '_fusion.txt', 'w')

        datasets_path_t = os.path.join(datasets_path, se)
        types_name = os.listdir(datasets_path_t)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)

            intercept_photos_path = os.path.join(intercept_path, type_name)
            slope_photos_path = os.path.join(slope_path, type_name)
            pixel_photos_path = os.path.join(pixel_path, type_name)

            intercept_photos = os.listdir(intercept_photos_path)
            slope_photos = os.listdir(slope_photos_path)
            pixel_photos = os.listdir(pixel_photos_path)

            for i in range(len(intercept_photos)):
                intercept_photo_name = intercept_photos[i]
                slope_photo_name = slope_photos[i]
                pixel_photo_name = pixel_photos[i]

                _, intercept_postfix = os.path.splitext(intercept_photo_name)
                _, slope_postfix = os.path.splitext(slope_photo_name)
                _, pixel_postfix = os.path.splitext(pixel_photo_name)

                if (
                    intercept_postfix not in ['.jpg', '.png', '.jpeg']
                    or slope_postfix not in ['.jpg', '.png', '.jpeg']
                    or pixel_postfix not in ['.jpg', '.png', '.jpeg']
                ):
                    continue

                intercept_photo_path = os.path.join(intercept_photos_path, intercept_photo_name)
                slope_photo_path = os.path.join(slope_photos_path, slope_photo_name)
                pixel_photo_path = os.path.join(pixel_photos_path, pixel_photo_name)

                list_file.write(
                    str(cls_id) + ";" + '%s;%s;%s' % (intercept_photo_path, slope_photo_path, pixel_photo_path)
                )
                list_file.write('\n')

        list_file.close()
