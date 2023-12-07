import cv2
import os

# process the images (grayscale, crop, drop images wo faces, choose face from group photo)
def process_images(img_dir_path, df, isTraining):
    imgs = []
    ids_with_faces = []
    ids_wo_faces = []
    age_groups = []
    genders = []

    # Define the cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for i, row in df.iterrows():
        userid = row["user_ids"]
        img_path = os.path.join(img_dir_path, userid + ".jpg")
        image = cv2.imread(img_path)

        # Detect faces
        faces = face_cascade.detectMultiScale(image)

        cropped_faces = []
        if len(faces) > 0:
            ids_with_faces.append(userid)
            # Draw bounding boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = image[y:y + h, x:x + w]                
                gray_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.resize(gray_image, (128, 128))
                cropped_faces.append(gray_image)
            imgs.append(cropped_faces[0]) #select the first face
            if(isTraining):
                age_groups.append(row["age_group_label"])
                genders.append(row["gender"])
        else:
            ids_wo_faces.append(userid)
    return {'imgs': imgs, 'ids_with_faces': ids_with_faces, 'ids_wo_faces': ids_wo_faces, 'genders': genders, 'age_groups': age_groups}