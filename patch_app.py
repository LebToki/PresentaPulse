import sys

with open('app.py', 'r') as f:
    content = f.read()

search = """            selected_faces_list = [faces[int(idx.split()[1]) - 1] for idx in selected_face_indices
                                 if idx.startswith("Face")]

            for idx, face in enumerate(selected_faces_list):
                if progress:
                    progress(idx / len(selected_faces_list), f"Processing face {idx+1}/{len(selected_faces_list)}...")

                # Crop face from image
                if CV2_AVAILABLE:
                    cropped_face = detector.crop_face(image_path, face)
                    temp_face_path = live_portrait_output_dir / f'temp_face_{idx}.jpg'
                    cv2.imwrite(str(temp_face_path), cropped_face)"""

replace = """            selected_faces_list = [faces[int(idx.split()[1]) - 1] for idx in selected_face_indices
                                 if idx.startswith("Face")]

            # Load the image once to optimize repeated crops
            if CV2_AVAILABLE:
                loaded_img = cv2.imread(str(image_path))
            else:
                loaded_img = image_path

            for idx, face in enumerate(selected_faces_list):
                if progress:
                    progress(idx / len(selected_faces_list), f"Processing face {idx+1}/{len(selected_faces_list)}...")

                # Crop face from image using the pre-loaded image array
                if CV2_AVAILABLE:
                    cropped_face = detector.crop_face(loaded_img, face)
                    temp_face_path = live_portrait_output_dir / f'temp_face_{idx}.jpg'
                    cv2.imwrite(str(temp_face_path), cropped_face)"""

if search in content:
    content = content.replace(search, replace)
    with open('app.py', 'w') as f:
        f.write(content)
    print("Patch applied successfully.")
else:
    print("Search string not found in app.py!")
