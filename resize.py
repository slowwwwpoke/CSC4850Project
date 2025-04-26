# resizing the images 256 x 256
import cv2 as cv
import os

dataset_images_paths = ['datasets/combined']

img_size = (160, 160)

def resize_image(datasets):
  resized_dataset_paths = []
  for dataset in datasets:
    # create dir for resized datasets
    resized_dataset = f"{dataset}-resized"
    os.makedirs(resized_dataset, exist_ok=True)
    resized_dataset_paths.append(resized_dataset)

    # iterate through the directories of each dataset
    for folder in os.listdir(dataset):
      # path for the curr folder
      folder_path = os.path.join(dataset, folder)
      # create new folder for resized images (keep in same category)
      resized_folder = f"{dataset}-resized/{folder}"
      os.makedirs(resized_folder, exist_ok=True)

      for img_file in os.listdir(folder_path):
        # path for curr img
        img_path = os.path.join(folder_path, img_file)
        img = cv.imread(img_path)

        if img is None:
          print(f"{img_file} could not be read.")
          continue

        resized_img = cv.resize(img, img_size, interpolation=cv.INTER_AREA)
        output_path = os.path.join(resized_folder, img_file)
        cv.imwrite(output_path, resized_img)

  return resized_dataset_paths

resized_dataset_paths = resize_image(dataset_images_paths)
