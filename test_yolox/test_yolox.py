import os
import pixeltable as pxt
from pixeltable.ext.functions.yolox import yolox
from PIL import Image

def main():
    print("Testing YOLOX UDF functionality...")
    
    # Initialize Pixeltable
    pxt.init()
    
    # Create a table with an image column
    try:
        # Drop the table if it exists
        try:
            pxt.get_table('yolox_test')
            pxt.drop_table('yolox_test')
            print("Dropped existing table 'yolox_test'")
        except:
            pass
        
        # Create a new table
        table = pxt.create_table('yolox_test', {'image': pxt.Image})
        print("Created table 'yolox_test'")
        
        # Add computed columns with YOLOX object detections using different models
        print("Adding computed columns with YOLOX object detections...")
        table.add_computed_column(detect_yolox_tiny=yolox(table.image, model_id='yolox_tiny', threshold=0.25))
        table.add_computed_column(detect_yolox_nano=yolox(table.image, model_id='yolox_nano', threshold=0.2))
        
        # Access specific parts of the detection results
        print("Adding computed columns for specific parts of the detection results...")
        table.add_computed_column(yolox_tiny_bboxes=table.detect_yolox_tiny.bboxes)
        table.add_computed_column(yolox_tiny_labels=table.detect_yolox_tiny.labels)
        table.add_computed_column(yolox_tiny_scores=table.detect_yolox_tiny.scores)
        
        # Insert a test image
        # Note: You'll need to download a test image or use your own image
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_image.jpg')
        if os.path.exists(image_path):
            print(f"Inserting test image from {image_path}...")
            table.insert([{'image': image_path}])
            
            # Collect and display results
            print("\nCollecting results...")
            rows = table.collect()
            
            print("\nYOLOX Tiny Detection Results:")
            print(rows['detect_yolox_tiny'][0])
            
            print("\nYOLOX Nano Detection Results:")
            print(rows['detect_yolox_nano'][0])
            
            print("\nYOLOX Tiny Bounding Boxes:")
            print(rows['yolox_tiny_bboxes'][0])
            
            print("\nYOLOX Tiny Labels:")
            print(rows['yolox_tiny_labels'][0])
            
            # Convert labels to class names
            from yolox.data.datasets import COCO_CLASSES
            print("\nClass names for YOLOX Tiny Labels:")
            for label in rows['yolox_tiny_labels'][0]:
                print(f"Label {label}: {COCO_CLASSES[label]}")
        else:
            print(f"Test image not found at {image_path}")
            print("Please download a test image or use your own image.")
            print("You can download a test image with:")
            print("curl -o test_image.jpg https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/tests/data/000000000001.jpg")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        try:
            pxt.drop_table('yolox_test')
            print("Dropped table 'yolox_test'")
        except:
            pass

if __name__ == "__main__":
    main()
