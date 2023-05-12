import argparse
import sys
import json


def format_converter(args):
    # COCO: [x_min, y_min, width, height]
    # YOLO: [x_center, y_center, width, height]
    # VOC:  [x_min, y_min, x_max, y_max]

    # Load the original JSON file
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Create a new dictionary to store the converted data
    new_data = {}

    # Loop through each object in the original JSON file
    for obj in data:
        # Get the image ID
        image_id = obj['image_id']
        # Get the category ID, bounding box, and score
        category_id = obj['category_id'] + 1
        bbox = obj['bbox']
        score = obj['score']

        if args.convert_image_id:
            # Load the original JSON file
            with open(args.refer_path, 'r') as f:
                reference = json.load(f)

            # Iterate over the list of dictionaries
            for d in reference['images']:
                # check if the id matches the desired_id
                if d['id'] == image_id:
                    # retrieve the corresponding file_name
                    image_id = d['file_name']
                    break
        else:
            image_id = image_id + '.png'

        if args.input_format == 'COCO' and args.output_format == 'VOC':
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]

        # If the image ID is not already in the new dictionary, add it
        if image_id not in new_data:
            new_data[image_id] = {'boxes': [], 'labels': [], 'scores': []}

        # Add the category ID, bounding box, and score to the new dictionary
        new_data[image_id]['labels'].append(category_id)
        new_data[image_id]['boxes'].append(bbox)
        new_data[image_id]['scores'].append(score)

    # Save the converted data to a new JSON file
    with open(args.output, 'w') as f:
        json.dump(new_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='yolov7_results.json', help='input json filename')
    parser.add_argument('--output', type=str, default='yolov7_results_converted.json',
                        help='ouput json filename')
    parser.add_argument('--input-format', type=str,
                        default='COCO',
                        help='input bbox format')
    parser.add_argument('--output-format', type=str,
                        default='VOC',
                        help='ouput bbox format')
    parser.add_argument('--convert-image-id', type=bool, default=False,
                        help='[FOR DETR OUTPUT] convert the image id from numbers to filenames')
    parser.add_argument('--refer-path', type=str, default='valid/annotations.coco.json',
                        help='[FOR DETR OUTPUT] the annotations that contains the match image id and filenames')
    opt = parser.parse_args()

    format_converter(opt)
