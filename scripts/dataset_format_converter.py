import os
import glob
import json
import argparse


# Create a folder at a given path if it does not exist
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Generate a train and val dataset from the source path
def generate_train_and_val(source_path):
    # Get a list of all subfolders in the source path
    datasets = [f.path for f in os.scandir(source_path) if f.is_dir()]

    # For each subfolder
    for dataset in datasets:
        print(dataset)
        # Get the basename of the subfolder
        dataset_basename = os.path.basename(dataset)
        # Get the path of the grandparent folder of the subfolder
        parent_path = os.path.dirname(os.path.dirname(dataset))
        # Create a text file with the same name as the subfolder, in the parent folder
        with open(f'{parent_path}/{dataset_basename}.txt', 'w+') as f:
            # For each .png file in the subfolder, write the file path relative to the grandparent folder to the text file
            for jpg_file in sorted(glob.glob(f'{dataset}/**' + '/*.png')):
                f.write(jpg_file.replace(f'{parent_path}/', './') + '\n')

        print(f'Text file of dataset {dataset} generated successfully.')


# Export labels in the YOLO bbox format from the source path
def export_labels(source_path):
    # Get a list of all subfolders in the source path
    datasets = [f.path for f in os.scandir(source_path) if f.is_dir()]

    # For each subfolder
    for dataset in datasets:
        # Split the path of the subfolder into a list of directories
        dataset_path_split = dataset.split("/")
        try:
            # Try to open the COCO annotation (with VOC bbox format) file for the subfolder
            with open(f'{dataset}/_annotations.coco.json') as annotations:
                label_folder = dataset.replace("images", "labels")
                create_folder(label_folder)

                # Read the contents of the COCO annotation file and parse the JSON
                annotations_contents = annotations.read()
                annotations_json = json.loads(annotations_contents)

                # For each .png file in the subfolder
                for jpg_file in glob.glob(f'{dataset}/**' + '/*.png'):
                    # Split the path into its components
                    path_components = jpg_file.split("/")
                    # Get the location from the path
                    location = path_components[5]
                    # Create a folder with the same name as the location, in the label folder
                    create_folder(f'{label_folder}/{location}')
                    # Remove the first two folders from the path
                    jpg_file = '/'.join(path_components[4:])

                    # Find the image object in the annotations JSON that corresponds to the current .png file
                    pic = [img for img in annotations_json['images']
                           if img['file_name'] == f'leftImg8bit/{jpg_file}'][0]
                    pic_id, pic_height, pic_width = pic['id'], pic['height'], pic['width']
                    # Find the annotations (bounding boxes) for the current image
                    annotation = [bbox for bbox in annotations_json['annotations']
                                  if bbox['image_id'] == pic_id]
                    # Create a text file with the same name as the .png file, in the label folder
                    output_file = open(
                        f'{label_folder}/{location}/{os.path.basename(jpg_file)[:-4]}.txt', 'w+')
                    # For each bounding box, write the category ID and coordinates in the YOLO format to the text file
                    for item in annotation:
                        current_bbox = item['bbox']
                        x, y, w, h = current_bbox[0], current_bbox[1], current_bbox[2], current_bbox[3]

                        # Finding midpoints
                        x_centre = (x + (x+w))/2
                        y_centre = (y + (y+h))/2
                        
                        # Normalization
                        x_centre = x_centre / pic_width
                        y_centre = y_centre / pic_height
                        w = w / pic_width
                        h = h / pic_height
                        
                        # Limiting upto fix number of decimal places
                        x_centre = f'{x_centre:.6f}'
                        y_centre = f'{y_centre:.6f}'
                        w = f'{w:.6f}'
                        h = f'{h:.6f}'

                        output_file.write(
                            f'{item["category_id"]-1} {x_centre} {y_centre} {w} {h}\n')

                    output_file.close()

                print(f'Labels of dataset {dataset} generated successfully.')
        except:
            print(f'Dataset {dataset} has no label.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-folder-path', type=str, default='./yolov7/customdata/fog/images/',
                        help='Path to the images folder')
    args = parser.parse_args()

    generate_train_and_val(args.source_folder_path)
    export_labels(args.source_folder_path)
