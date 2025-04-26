# Denoising, Gaussian blur
import cv2 as cv
import os

def denoise_images(datasets):
    denoised_dataset_paths = []
    for dataset in datasets:
        # Output for denoised datasets
        denoised_dataset = f"{dataset}-denoised"
        os.makedirs(denoised_dataset)
        denoised_dataset_paths.append(denoised_dataset)

        # Iterate through the directories of each dataset
        for folder in os.listdir(dataset):

            folder_path = os.path.join(dataset, folder)
            denoised_folder = f"{dataset}-denoised/{folder}"
            os.makedirs(denoised_folder, exist_ok=True)

            for img_file in os.listdir(folder_path):
                # Path for current image
                img_path = os.path.join(folder_path, img_file)
                img = cv.imread(img_path)

                if img is None:
                    print(f"{img_file} could not be read.")
                    continue

                # Apply Gaussian blur
                blurred = cv.GaussianBlur(img, (3, 3), 0)

                # Apply bilateral denoising
                denoised_img = cv.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)

                # Save denoised image to the corresponding output folder
                output_path = os.path.join(denoised_folder, img_file)
                cv.imwrite(output_path, denoised_img)

        print("Finished denoising " + str(folder_path))

    return denoised_dataset_paths
resized_dataset_paths = ['datasets/combined-resized']

denoised_dataset_paths = denoise_images(resized_dataset_paths)
print(f"Denoising completed" + str(denoised_dataset_paths))
