import cv2
import mediapipe as mp
import pyttsx3


engine = pyttsx3.init()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

aah_img = cv2.imread("aa.png", cv2.IMREAD_COLOR)
aah_img = cv2.resize(aah_img, (200, 180))  # Resize to match the dimensions of ROI

ah_img = cv2.imread("ah.png", cv2.IMREAD_COLOR)
ah_img = cv2.resize(ah_img, (200, 180))

akk_img = cv2.imread("akk.png", cv2.IMREAD_COLOR)
akk_img = cv2.resize(akk_img, (200, 180))

au_img = cv2.imread("au.png", cv2.IMREAD_COLOR)
au_img = cv2.resize(au_img, (200, 180))

m_img = cv2.imread("nn.png", cv2.IMREAD_COLOR)
m_img = cv2.resize(m_img, (200, 180))


kk_img = cv2.imread("ik.png", cv2.IMREAD_COLOR)
kk_img = cv2.resize(kk_img, (200, 180))

# Create a dictionary to map each sign to its corresponding image
sign_images = {
    "ஆ": aah_img,
    "அ": ah_img,
    "ஃ": akk_img,
    "ஔ": au_img,
    "ண்": m_img,
    "க்": kk_img,

}

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)
            finger_fold_status = []
            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                if lm_list[tip].x < lm_list[tip - 2].x:
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
            print(x, y)

            detected_sign = None
            if (lm_list[4].y < lm_list[8].y) and (lm_list[4].x < lm_list[8].x):
                detected_sign ="ண்"

            if lm_list[4].y < lm_list[2].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                0].x < \
                    lm_list[5].x:
                detected_sign = "ஃ"

            if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                    lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                detected_sign = "அ"


            if all(finger_fold_status[:4]) and not any(finger_fold_status[4:]):
                detected_sign = "க்"

            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y and \
                   lm_list[8].y < lm_list[7].y:  # Check if thumb and index fingers are raised
                    detected_sign = "ஆ"

                elif (lm_list[4].y < lm_list[11].y) and (lm_list[4].x < lm_list[11].x):
                    detected_sign = "ஃ"

            # Display "ஔ" when all fingers and thumb are raised
            if all(lm_list[tip].y < lm_list[thumb_tip].y for tip in finger_tips):
                detected_sign = "ஔ"

            if detected_sign:
                print(detected_sign)
                if detected_sign in sign_images:
                    sign_img = sign_images[detected_sign]
                    img[0:180, 0:200] = sign_img

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )

    cv2.imshow("Hand Sign Detection", img)
    cv2.waitKey(1)
