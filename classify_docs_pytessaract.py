# -*- coding: utf-8 -*-

import cv2
import imutils
import pytesseract
import os
from fuzzywuzzy import fuzz
import pyzbar.pyzbar as pyzbar
import re

match_strc_docs = {
    'pan_f': {
        'basic': [{"INCOME": 3, "TAX": 3, "DEPARTMENT": 1, "GOVT": 1, "OF": 0, "INDIA": 1},
                  {"Permanent": 2, "Account": 2, "Number": 1},
                  {"Signature": 1}],
        'optional': {}
    },
    'aadhaar_f': {
        'basic': [{"GOVERNMENT": 2, "OF": 0, "INDIA": 1},
                  {'dob': {"/DOB:": 2, "/ DOB:": 2, "/ Year of Birth": 2, "/Year of Birth": 2, "/Yearof Birth": 2, "/Year ofBirth": 2, "/YearofBirth": 2}},
                  {'sex': {"/ MALE": 1, "/ FEMALE": 1, "/MALE": 1, "/FEMALE": 1}}],
        'optional': {}
    },
    'aadhaar_b': {
        'basic': [{'Unique': 3, 'Identification': 2, 'Authority': 2, "OF": 0, "INDIA": 1}],
        'optional': {'help@uidai.gov.in': 3, 'www.uidai.gov.in': 3, '1947': 2, '1800': 1, "Address": 3}
    },
}

match_unstrc_docs = {
    'income_proof_1': {
        1: ['signatory', 'deductions', 'payslip', 'earning', 'allowance', 'incentive',
            'esic', 'arrears', 'provident', 'fund', 'computer generated statement', 'joining date', 'resignation',
            'does not require any', 'bonus', 'basic', 'deputed', 'uan no', 'salary slip'],
        0.6: ['employee', 'only', 'total', 'of the month'],
        0.4: ['pay', 'slip', 'salary', 'net', 'gross', 'pay', 'name', 'desig', 'bank', 'payslip', 'acc', 'department', 'leave',
              'tax', 'tds', 'does not require any', 'bonus', 'basic', 'deputed', 'uan no'],
        -0.4: ['voter', 'govt', 'government', 'election' 'income tax department', 'unique identification',
               'authority of india', 'cash', 'pass', 'book', 'central board of excise']
    }
}

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def permutes(words_in_same_line):
    """outputs attached words
    eg., {"INCOME": 3, "TAX": 3, "DEPARTMENT": 1, "GOVT": 1, "OF": 0, "INDIA": 1} =>
         {'INCOME': 'INCOMETAX','TAX': 'TAXDEPARTMENT','DEPARTMENT': 'DEPARTMENTGOVT','GOVT': 'GOVTOF','OF': 'OFINDIA'}
    """
    new_dict = {}
    for i, k in zip(list(words_in_same_line.keys())[0::1], list(words_in_same_line.keys())[1::]):
        new_dict[i] = i+k
    return new_dict


def compute_struc_scores(text, qr_n_faces):
    """computes scores for the given document corresponding to all the defined document structures in 'match_strc_docs'"""
    scores_key = []
    scores_val = []
    pan_matches = len(re.findall("[A-Z]{5}[0-9]{4}[A-Z]{1}", text.replace(" ", "")))
    aadhar_matches = len(re.findall("[0-9]{12}", text.replace(" ", "")))
    # print("pan_matches, aadhar_matches: ", pan_matches, aadhar_matches, "qr_n_faces", qr_n_faces)
    for key, value in match_strc_docs.items():
        cumulative = 0
        norm_factor = 0
        for type, word_dict in value.items():
            if type == 'basic':
                for line in word_dict:
                    for word, points in line.items():
                        if isinstance(points, int):
                            combined_words = permutes(line)
                            similarity = fuzz.token_set_ratio(text, word)
                            try:
                                similarity1 = fuzz.token_set_ratio(text, combined_words[word])
                                # print(word, similarity, similarity1)
                                similarity = max(similarity, similarity1)
                            except:
                                x = 0
                                # print(word, similarity)
                            if similarity < 50:
                                similarity = 0
                            cumulative += points * similarity
                            norm_factor += points
                        elif isinstance(points, dict):
                            similarities = []
                            for word1, point in points.items():
                                similarities.append(fuzz.token_set_ratio(text, word1))
                            similarity = max(similarities)
                            # print(word, similarity)
                            if similarity < 50:
                                similarity = 0
                            cumulative += point * similarity
                            norm_factor += point
            elif type == 'optional':
                for word, points in word_dict.items():
                    similarity = fuzz.token_set_ratio(text, word)
                    # print(word, similarity)
                    if similarity > 60:
                        cumulative += points * similarity
                        norm_factor += points
        cumulative = cumulative/100
        # bonus scores
        if key == 'pan_f':
            if pan_matches > 0:
                cumulative += 3
                norm_factor += 3
            if qr_n_faces[1] == 1:
                cumulative += 2
                norm_factor += 2
        elif key == 'aadhaar_f':
            if aadhar_matches > 0:
                cumulative += 3
                norm_factor += 3
            if qr_n_faces[0] == 1:
                cumulative += 50
                norm_factor += 50
            if qr_n_faces[1] == 1:
                cumulative += 2
                norm_factor += 2
        elif key == 'aadhaar_b':
            if aadhar_matches > 0:
                cumulative += 3
                norm_factor += 3
            if qr_n_faces[0] == 1:
                cumulative += 50
                norm_factor += 50
        scores_key.append(key)
        scores_val.append((cumulative/norm_factor)*100)
    return scores_key, scores_val


def compute_unstruc_scores(text, qr_n_faces):
    scores_key, scores_val = [], []
    for key, value in match_unstrc_docs.items():
        cumulative = 0
        for val, words in value.items():
            similarities = []
            for word in words:
                similarity = fuzz.token_set_ratio(text.lower(), word)
                if similarity < 50:
                    similarity = 0
                # print(word, similarity)
                similarities.append(similarity)
            cumulative += val * sum(x for x in similarities)
        norm_factor = 5
        if qr_n_faces[0] > 0:
            cumulative -= 50
        if qr_n_faces[1] > 0:
            cumulative -= 15
        scores_key.append(key)
        # print("norm_factor ", norm_factor)
        # print("cumulative ", cumulative)
        scores_val.append(cumulative / norm_factor)
    return scores_key, scores_val


def show_img(img, title):
    """show smaller image - to fit screen"""
    try:
        height, width, _ = img.shape
    except:
        height, width = img.shape
    final_show = cv2.resize(img, (width // 2, height // 2))
    cv2.imshow(title, final_show)
    cv2.waitKey(0)


def doc_score_gen(gray):
    text_default = pytesseract.image_to_string(gray, lang='eng')
    flipped = imutils.rotate_bound(gray, 180)
    text_flipped = pytesseract.image_to_string(flipped)
    combined = text_default + text_flipped
    # print(combined)
    # print(20*"-")
    letter_count = len(combined.replace("\n", "").replace(" ", ""))
    qr_n_faces = [decode_qr(img), detect_face(img)]
    if letter_count > 200:
        scores_key, scores_val = compute_unstruc_scores(combined, qr_n_faces)
        if max(scores_val) < 55:
            if letter_count < 1300:
                scores_key, scores_val = compute_struc_scores(combined, qr_n_faces)
            else:
                scores_key, scores_val = ['None'], [0]
    else:
        if letter_count > 90 and letter_count < 1300:
            scores_key, scores_val = compute_struc_scores(combined, qr_n_faces)
        else:
            scores_key, scores_val = ['None'], [0]
    return scores_key, scores_val


def image_to_text(img_before, filter='normal'):
    """texual+face+qr analysis"""
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    pure_gray = gray.copy()
    if filter == 'inverted':  # this may help when document is captured on a light background
        ret, gray = cv2.threshold(gray, 150, 255, 0)
        gray = cv2.bitwise_not(gray)
    if filter == 'normal':
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)

    scores_keys = []
    scores_vals = []

    scores_key, scores_val = doc_score_gen(pure_gray)
    scores_keys.extend(scores_key)
    scores_vals.extend(scores_val)
    
    # show_img(gray, 'text1')
    scores_key, scores_val = doc_score_gen(gray)
    scores_keys.extend(scores_key)
    scores_vals.extend(scores_val)

    gray1 = cv2.erode(gray,kernel,iterations = 1)
    # show_img(gray1, 'text2')
    scores_key, scores_val = doc_score_gen(gray1)
    scores_keys.extend(scores_key)
    scores_vals.extend(scores_val)

    gray = cv2.dilate(gray,kernel,iterations = 1)
    # show_img(gray, 'text3')
    scores_key, scores_val = doc_score_gen(gray)
    scores_keys.extend(scores_key)
    scores_vals.extend(scores_val)
    print(scores_keys, scores_vals)
    max_val = max(scores_vals)
    if max_val > 55:
        return scores_keys[scores_vals.index(max_val)]
    else:
        return "Donno"


def decode_qr(im):
    # Find barcodes and QR codes
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    decodedObjects = pyzbar.decode(im)
    # for obj in decodedObjects:
    #     print('Type : ', obj.type)
    #     print('Data : ', obj.data, '\n')
    return len(decodedObjects)


def detect_face(original_image):
    # Convert color image to grayscale for Viola-Jones
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Load the classifier and create a cascade object for face detection
    face_cascade = cv2.CascadeClassifier(r'C:\Users\Mohit\PycharmProjects\untitled1\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
    return len(detected_faces)


# # try on images from specific directory

path = 'C:\\Users\\***' # edit path
files = []
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))
index = 0
for f in files:
    if ".jpg" in f:
        img = cv2.imread(f)
        result = image_to_text(img)
        print(result)
        new_name = "\\".join(f.split("\\")[:-1]) + "\\" + result + str(index) + ".jpg"
        try:
            os.replace(f, new_name)
            print(new_name)
        except:
            print(f)
        # show_img(img, 'org')
        # cv2.destroyAllWindows()
        index += 1
        print("\n", 5 * "***end of doc***", "\n")


# # try on individual image

# f = r'C:\Users\***' # edit path
# img = cv2.imread(f)
# result = image_to_text(img)
# print(result)
# show_img(img, 'org')