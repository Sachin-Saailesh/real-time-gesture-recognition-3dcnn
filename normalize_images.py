import os
from PIL import Image

RAW_IMAGES_DIR = "raw_hand_images"  # Put your original photos here
OUTPUT_DIR = "my_test_images"  # Normalized images will be saved here
TARGET_SIZE = (128, 128)

def normalize_images():
    """Convert images to 128x128 grayscale PNG format"""

    # Create directories if they don't exist
    os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(RAW_IMAGES_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"No images found in '{RAW_IMAGES_DIR}'")
        print(f"Please add your hand gesture photos to '{RAW_IMAGES_DIR}' folder")
        return

    print(f"Found {len(image_files)} images to normalize\n")

    for idx, img_file in enumerate(image_files):
        try:
            # Load image
            img_path = os.path.join(RAW_IMAGES_DIR, img_file)
            img = Image.open(img_path)

            # Convert to grayscale
            img_gray = img.convert('L')

            # Resize to 128x128
            img_resized = img_gray.resize(TARGET_SIZE, Image.LANCZOS)

            # Save as PNG
            output_filename = f"normalized_{idx+1:02d}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            img_resized.save(output_path, 'PNG')

            print(f"✓ Processed: {img_file} -> {output_filename}")

        except Exception as e:
            print(f"✗ Error processing {img_file}: {str(e)}")

    print(f"\nNormalization complete! {len(image_files)} images saved to '{OUTPUT_DIR}'")
    print("\nNext steps:")
    print("1. Review the normalized images")
    print("2. Optionally rename them to indicate the gesture (e.g., '3_fingers_1.png')")
    print("3. Run '4_test_custom_images.py' to test your models")

if __name__ == "__main__":
    normalize_images()
