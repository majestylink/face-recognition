import os
import shutil

import numpy as np
import cv2
import matplotlib
import pywt
from matplotlib import pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


class Classify:
    img = cv2.imread('./dataset/christiano_ronaldo/download (10).jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    path_to_data = "./dataset/"
    path_to_cr_data = "./dataset/cropped/"

    img_dirs = []

    cropped_image_dirs = []
    celebrity_file_names_dict = {}

    class_dict = {}

    best_clf = None

    def plot_colorful_image(self):
        print(os.getcwd())
        # print(self.img.shape)
        cv2.imshow("Ronaldo", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_black_and_white_image(self):
        # print(gray.shape)
        cv2.imshow("Ronaldo", self.gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_rectangle_on_face(self):
        faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        (x, y, w, h) = faces[0]
        face_img = cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("Cropped", face_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_rectangle_on_eyes(self):
        faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = self.gray[y:y + h, x:x + w]
            roi_color = face_img[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("Cropped", face_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return roi_color

    def get_cropped_image_if_2_eyes(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                return roi_color

    def create_dir_for_each_person(self):
        for entry in os.scandir(self.path_to_data):
            if entry.is_dir():
                self.img_dirs.append(entry.path)
        if os.path.exists(self.path_to_cr_data):
            shutil.rmtree(self.path_to_cr_data)
        os.mkdir(self.path_to_cr_data)
        # return img_dirs

    def crop_images(self):
        self.create_dir_for_each_person()

        for img_dir in self.img_dirs:
            count = 1
            celebrity_name = img_dir.split('/')[-1]
            print(celebrity_name)

            self.celebrity_file_names_dict[celebrity_name] = []

            for entry in os.scandir(img_dir):
                roi_color = self.get_cropped_image_if_2_eyes(entry.path)
                if roi_color is not None:
                    cropped_folder = self.path_to_cr_data + celebrity_name
                    if not os.path.exists(cropped_folder):
                        os.makedirs(cropped_folder)
                        self.cropped_image_dirs.append(cropped_folder)
                        print("Generating cropped images in folder: ", cropped_folder)

                    cropped_file_name = celebrity_name + str(count) + ".png"
                    cropped_file_path = cropped_folder + "/" + cropped_file_name

                    cv2.imwrite(cropped_file_path, roi_color)
                    self.celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                    count += 1

    def w2d(self, img, mode='haar', level=1):
        imArray = img
        # Datatype conversions
        # convert to grayscale
        imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
        # convert to float
        imArray = np.float32(imArray)
        imArray /= 255
        # compute coefficients
        coeffs = pywt.wavedec2(imArray, mode, level=level)

        # Process Coefficients
        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0

        # reconstruction
        imArray_H = pywt.waverec2(coeffs_H, mode)
        imArray_H *= 255;
        imArray_H = np.uint8(imArray_H)

        return imArray_H

    def map_persons_with_number(self):
        count = 0
        for celebrity_name in self.celebrity_file_names_dict.keys():
            self.class_dict[celebrity_name] = count
            count = count + 1
        return self.class_dict

    def prepare_x_and_y(self):
        X, y = [], []
        for celebrity_name, training_files in self.celebrity_file_names_dict.items():
            for training_image in training_files:
                img = cv2.imread(training_image)
                scalled_raw_img = cv2.resize(img, (32, 32))
                img_har = self.w2d(img, 'db1', 5)
                scalled_img_har = cv2.resize(img_har, (32, 32))
                combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
                X.append(combined_img)
                y.append(self.class_dict[celebrity_name])
        X = np.array(X).reshape(len(X), 4096).astype(float)
        return X, y

    def train_model(self):
        import pandas as pd

        sample = self.prepare_x_and_y()
        X_train, X_test, y_train, y_test = train_test_split(sample[0], sample[1], random_state=0)
        model_params = {
            'svm': {
                'model': svm.SVC(gamma='auto', probability=True),
                'params': {
                    'svc__C': [1, 10, 100, 1000],
                    'svc__kernel': ['rbf', 'linear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'randomforestclassifier__n_estimators': [1, 5, 10]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(solver='liblinear', multi_class='auto'),
                'params': {
                    'logisticregression__C': [1, 5, 10]
                }
            }
        }

        scores = []
        best_estimators = {}
        for algo, mp in model_params.items():
            pipe = make_pipeline(StandardScaler(), mp['model'])
            clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
            clf.fit(X_train, y_train)
            scores.append({
                'model': algo,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_
            })
            best_estimators[algo] = clf.best_estimator_

        df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
        print(df)
        print("SVM\t", best_estimators['svm'].score(X_test, y_test))
        print("Random Forest\t", best_estimators['random_forest'].score(X_test, y_test))
        print("Logistic Regression\t", best_estimators['svm'].score(X_test, y_test))

        self.best_clf = best_estimators['svm']

        return best_estimators['svm']

    def check_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        import seaborn as sn

        sample = self.prepare_x_and_y()
        X_train, X_test, y_train, y_test = train_test_split(sample[0], sample[1], random_state=0)
        cm = confusion_matrix(y_test, self.best_clf.predict(X_test))
        print(cm)
        # plt.figure(figsize=(10, 7))
        # sn.heatmap(cm, annot=True)
        # plt.xlabel('Predicted')
        # plt.ylabel('Truth')

    def save_trained_model_and_class_dict(self):
        import joblib
        import json
        # Save the model as a pickle in a file
        joblib.dump(self.best_clf, 'saved_model.pkl')
        with open("class_dictionary.json", "w") as f:
            f.write(json.dumps(self.class_dict))


c = Classify()
# c.plot_colorful_image()
# c.plot_black_and_white_image()
# c.draw_rectangle_on_face()
# c.draw_rectangle_on_eyes()
c.crop_images()
print(c.map_persons_with_number())
# c.prepare_x_and_y()
c.train_model()
c.save_trained_model_and_class_dict()
# c.check_confusion_matrix()
# roi = c.get_cropped_image_if_2_eyes()
# cv2.imshow("Cropped", roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
