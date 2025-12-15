import cv2
import mediapipe as mp
import json
import os

mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks(image_path, output_json="target_landmarks.json", visualize=True):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ ERROR: Cannot read image '{image_path}'")
        return

    h, w = img.shape[:2]

    # Initialize MediaPipe
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False
    ) as face_mesh:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("❌ No face detected in this image.")
            return

        face_landmarks = results.multi_face_landmarks[0]

        # Extract 468 (x,y) points
        points = []
        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])

        # Save JSON
        data = {
            "image": os.path.basename(image_path),
            "width": w,
            "height": h,
            "points": points
        }

        # Save to same folder as image with same base name
        image_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(image_dir, f"{base_name}.json")
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved landmarks to {output_path}")

        # Optional visualization
        if visualize:
            for (x, y) in points:
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            cv2.imshow("Landmarks", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# -----------------------------
# EXAMPLE USAGE:
# extract_landmarks("lion.jpg", "lion_landmarks.json")
# extract_landmarks("elon_musk.jpg", "elon_musk_landmarks.json")
# extract_landmarks("cat.png", "cat_landmarks.json")
# -----------------------------

if __name__ == "__main__":
    # Specify the base folder containing subfolders with images
    base_folder = r"C:\Users\hp\Desktop\Anatomy tasks\Task 5\Targets"  # Change this to your folder path
    
    # Check if folder exists
    if not os.path.exists(base_folder):
        print(f"❌ Folder '{base_folder}' not found!")
        print("Please create the folder or update the path.")
    else:
        # Get all subdirectories in the base folder
        subdirs = [d for d in os.listdir(base_folder) 
                  if os.path.isdir(os.path.join(base_folder, d))]
        
        if not subdirs:
            print(f"❌ No subdirectories found in '{base_folder}'")
        else:
            print(f"Found {len(subdirs)} subdirectories: {', '.join(subdirs)}")
            
            total_processed = 0
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            
            # Process each subdirectory
            for subdir in subdirs:
                subdir_path = os.path.join(base_folder, subdir)
                print(f"\n📁 Processing folder: {subdir}")
                
                # Get all image files in this subdirectory
                image_files = [f for f in os.listdir(subdir_path) 
                              if os.path.splitext(f)[1] in image_extensions]
                
                if not image_files:
                    print(f"  ⚠️ No images found in '{subdir}'")
                    continue
                
                print(f"  Found {len(image_files)} images in {subdir}")
                
                for img_file in image_files:
                    img_path = os.path.join(subdir_path, img_file)
                    print(f"  📸 Processing: {img_file}")
                    extract_landmarks(img_path, visualize=False)
                    total_processed += 1
            
            print(f"\n✅ All done! Processed {total_processed} images from {len(subdirs)} folders.")
