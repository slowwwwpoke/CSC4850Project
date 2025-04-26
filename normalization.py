# normalization (contrast enhancement)
import cv2 as cv
import os

def normalize_dataset_images(datasets):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    normalized_dataset_paths = []
    for dataset in datasets:
      if os.path.exists(dataset):
        print(f"Found dataset {dataset}")
      else:
        print(f"Could not find dataset {dataset}")
    for dataset in datasets:
       if os.path.exists(dataset):
              if not os.path.exists(f"{dataset}-normalized"):
                     os.makedirs(f"{dataset}-normalized")
                     normalized_dataset_paths.append(f"{dataset}-normalized")

              for folder in os.listdir(dataset):
                     if not os.path.exists(f"{dataset}-normalized/{folder}"):
                            os.makedirs(f"{dataset}-normalized/{folder}")

                     for img in os.listdir(f"{dataset}/{folder}"):
                            image = cv.imread(f"{dataset}/{folder}/{img}")
                            lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
                            l, a, b = cv.split(lab)
                            l = clahe.apply(l)
                            lab = cv.merge((l, a, b))
                            image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
                            cv.imwrite(f"{dataset}-normalized/{folder}/{img}", image)
              print(f"Dataset {dataset} normalized")
       else:
              print(f"Dataset {dataset} not found")
              return False
    print(f"Dataset paths: " + str(normalized_dataset_paths))
    return normalized_dataset_paths

# i changed the dataset_images_paths -> denoised_dataset_paths -suzuna
denoised_dataset_paths = ['datasets/combined-resized-denoised']
normalized_dataset_paths = normalize_dataset_images(denoised_dataset_paths)