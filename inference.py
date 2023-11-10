import cv2
import mediapipe as mp
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from model import gesture_fc_type_vowel, gesture_fc_type_consonant, rockpaper
from PIL import ImageFont, ImageDraw, Image
from hangul_utils import join_jamos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_vowel = gesture_fc_type_vowel().to(device)
model_consonant = gesture_fc_type_consonant().to(device)

model_vowel.load_state_dict(torch.load('./FC_Model_vowel.pt'))
model_consonant.load_state_dict(torch.load('./FC_Model.pt'))

print('Model loaded Successfully')


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# font_path = './BMDOHYEON_ttf.ttf'
# font_path = './BMJUA_ttf.ttf'
# font_path = './NanumGothic.ttf'

font_path = './휴먼매직체.ttf'
font = ImageFont.truetype(font_path, 30)







Target = [ 
            {'key' : 'ㄱ', 'eng' : 'giyeok', 'label' : 0}, 
            {'key' : 'ㄴ', 'eng' : 'nieun', 'label' : 1}, 
            {'key' : 'ㄷ', 'eng' : 'digeut', 'label' : 2}, 
            {'key' : 'ㄹ', 'eng' : 'rieul', 'label' : 3}, 
            {'key' : 'ㅁ', 'eng' : 'mieum', 'label' : 4}, 
            {'key' : 'ㅂ', 'eng' : 'bieup', 'label' : 5}, 
            {'key' : 'ㅅ', 'eng' : 'siot', 'label' : 6}, 
            {'key' : 'ㅇ', 'eng' : 'ieung', 'label' : 7}, 
            {'key' : 'ㅈ', 'eng' : 'jieut', 'label' : 8}, 
            {'key' : 'ㅊ', 'eng' : 'chieut', 'label' : 9}, 
            {'key' : 'ㅋ', 'eng' : 'kieuk', 'label' : 10}, 
            {'key' : 'ㅌ', 'eng' : 'tieut', 'label' : 11}, 
            {'key' : 'ㅍ', 'eng' : 'pieup', 'label' : 12}, 
            {'key' : 'ㅎ', 'eng' : 'hieut', 'label' : 13}, 
            {'key' : 'ㅏ', 'eng' : 'a', 'label' : 14}, 
            {'key' : 'ㅑ', 'eng' : 'ya', 'label' : 15}, 
            {'key' : 'ㅓ', 'eng' : 'eo', 'label' : 16}, 
            {'key' : 'ㅕ', 'eng' : 'yeo', 'label' : 17}, 
            {'key' : 'ㅗ', 'eng' : 'o', 'label' : 18}, 
            {'key' : 'ㅛ', 'eng' : 'yo', 'label' : 19}, 
            {'key' : 'ㅜ', 'eng' : 'u', 'label' : 20}, 
            {'key' : 'ㅠ', 'eng' : 'yu', 'label' : 21}, 
            {'key' : 'ㅡ', 'eng' : 'eu', 'label' : 22}, 
            {'key' : 'ㅣ', 'eng' : 'i', 'label' : 23}, 
            {'key' : 'ㅔ', 'eng' : 'e', 'label' : 24}, 
            {'key' : 'ㅖ', 'eng' : 'ye', 'label' : 25}, 
            {'key' : 'ㅐ', 'eng' : 'ae', 'label' : 26}, 
            {'key' : 'ㅡ', 'eng' : 'yae', 'label' : 27}, 
            {'key' : 'ㅢ', 'eng' : 'ui', 'label' : 28}, 
            {'key' : 'ㅚ', 'eng' : 'oe', 'label' : 29}, 
            {'key' : 'ㅟ', 'eng' : 'wi', 'label' : 30}, 

        ]




def landmarks(data):
    return np.array([[landmark['x'], landmark['y'], landmark['z']] for landmark in data]).transpose()



# The normalization function
def normalize_landmarks(landmarks):
    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks], dtype=np.float32)
    norm_landmarks = landmarks_array - np.mean(landmarks_array, axis=0)
    max_dist = np.max(np.linalg.norm(norm_landmarks, axis=1))
    norm_landmarks = norm_landmarks / max_dist
    return norm_landmarks


def assign_hands_by_position(multi_hand_landmarks, image_width):
    # Assuming that the wrist is the first landmark in the hand landmarks list
    wrist_landmark_index = 0

    # Extract the x-coordinate of the wrist landmark for each hand
    hands_x_positions = [
        hand.landmark[wrist_landmark_index].x for hand in multi_hand_landmarks
    ]


    left_hand_index = np.argmin(hands_x_positions)
    right_hand_index = 1 - left_hand_index  # The other hand is the right hand

    # Assign hands based on left and right position
    if hands_x_positions[left_hand_index] * image_width < hands_x_positions[right_hand_index] * image_width:
        hand_landmarks_0 = multi_hand_landmarks[left_hand_index]
        hand_landmarks_1 = multi_hand_landmarks[right_hand_index]
    else:
        hand_landmarks_0 = multi_hand_landmarks[right_hand_index]
        hand_landmarks_1 = multi_hand_landmarks[left_hand_index]

    return hand_landmarks_0, hand_landmarks_1


# 엄지에 원그리기
def draw_hand_circle(image, hand_landmarks, touching, color, thickness=2, radius=30,THUMB_TIP_IDX=4):

    if touching:
        thumb_tip = hand_landmarks.landmark[THUMB_TIP_IDX]
        
        # 이미지의 크기에 맞춰서 실제 픽셀 좌표로 변환합니다.
        image_width, image_height = image.shape[1], image.shape[0]
        thumb_tip_x = int(thumb_tip.x * image_width)
        thumb_tip_y = int(thumb_tip.y * image_height)
        
        # 원을 그립니다.
        cv2.circle(image, (thumb_tip_x, thumb_tip_y), radius, color, thickness)
        
        return image






THUMB_INDEX_THRESHOLD = 0.22
THUMB_MIDDLE_THRESHOLD = 0.22

def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1[0] - landmark2[0])**2 + (landmark1[1] - landmark2[1])**2 + (landmark1[2] - landmark2[2])**2)

def check_fingers_touching(thumb_tip,index_finger_tip,middle_finger_tip, thumb_index_threshold=THUMB_INDEX_THRESHOLD, thumb_middle_threshold=THUMB_MIDDLE_THRESHOLD):


    thumb_index_distance = calculate_distance(thumb_tip, index_finger_tip)
    thumb_middle_distance = calculate_distance(thumb_tip, middle_finger_tip)

    thumb_index_touching = thumb_index_distance < thumb_index_threshold
    thumb_middle_touching = thumb_middle_distance < thumb_middle_threshold

    return thumb_index_touching, thumb_middle_touching


def make_double_consonant(consonant):
    if consonant == 'ㄱ':
        return 'ㄲ'
    elif consonant == 'ㄷ':
        return 'ㄸ'
    elif consonant == 'ㅂ':
        return 'ㅃ'
    elif consonant == 'ㅅ':
        return 'ㅆ'
    elif consonant == 'ㅈ':
        return 'ㅉ'
    else:
        return consonant




def draw_text(image, text, position, font=font, font_size=20, color=(0, 0, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # PIL 이미지에 텍스트 쓰기
    draw.text(position, text, font=font, fill=color)

    # PIL 이미지를 OpenCV 이미지로 다시 변환
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def inference_model(norm_landmarks, model):
    hand_data = []
    for norm_data in norm_landmarks:
        tmp= {
            "x": float(norm_data[0]),
            "y": float(norm_data[1]),
            "z": float(norm_data[2])
        }
        hand_data.append(tmp)
    
    data = torch.tensor(landmarks(hand_data),dtype=torch.float32).unsqueeze(0).to(device)
    output = model(data).squeeze(0).detach().cpu().numpy()
    return output






def vowel_compensation(recorded_letters):
    """
    ㅗㅐ = ㅙ
    ㅗㅏ = ㅘ
    ㅜㅔ = ㅞ
    ㅜㅓ = ㅝ
    """
    if recorded_letters[-2:] == 'ㅗㅐ':
        return recorded_letters[:-2] + 'ㅙ'
    elif recorded_letters[-2:] == 'ㅗㅏ':
        return recorded_letters[:-2] + 'ㅘ'
    elif recorded_letters[-2:] == 'ㅜㅔ':
        return recorded_letters[:-2] + 'ㅞ'
    elif recorded_letters[-2:] == 'ㅜㅓ':
        return recorded_letters[:-2] + 'ㅝ'
    else:
        return recorded_letters










landmark_drawing_spec_0 = mp.solutions.drawing_utils.DrawingSpec(color=(30, 250, 0), thickness=1, circle_radius=2)
connection_drawing_spec_0 = mp.solutions.drawing_utils.DrawingSpec(color=(250, 60, 60), thickness=2)

landmark_drawing_spec_1 = mp.solutions.drawing_utils.DrawingSpec(color=(0, 250, 30), thickness=1, circle_radius=2)
connection_drawing_spec_1 = mp.solutions.drawing_utils.DrawingSpec(color=(60, 60, 250), thickness=2)



thumb_index_touching = False
thumb_middle_touching = False
thumb_ring_touching = False
thumb_pinky_touching = False
rock = False

thumb_index_touching_list = [False, False, False, False]
thumb_middle_touching_list = [False, False, False, False]
thumb_ring_touching_list = [False, False, False, False]
thumb_pinky_touching_list = [False, False, False, False]
rock_list = [False, False, False, False]


recorded_letters = ''
hangeul = ''

cap = cv2.VideoCapture(0) 
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        
        image_width = frame.shape[1]
        color_image = cv2.flip(frame, 1) # 이미지 반전 : 필요없으면 삭제
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        



        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hand_landmarks_0, hand_landmarks_1 = assign_hands_by_position(results.multi_hand_landmarks, image_width)

            mp.solutions.drawing_utils.draw_landmarks(
                color_image, hand_landmarks_0, mp_hands.HAND_CONNECTIONS,landmark_drawing_spec_0,connection_drawing_spec_0)
            
            mp.solutions.drawing_utils.draw_landmarks(
                color_image, hand_landmarks_1, mp_hands.HAND_CONNECTIONS,landmark_drawing_spec_1,connection_drawing_spec_1)
                


            # Normalize landmarks
            norm_landmarks_0 = normalize_landmarks(hand_landmarks_0.landmark)
            norm_landmarks_1 = normalize_landmarks(hand_landmarks_1.landmark)



            ### 왼손으로 판단하는 영역
            # Voxel인지 Consonant인지 판단
            thumb_index_touching, thumb_middle_touching = check_fingers_touching(norm_landmarks_0[4],norm_landmarks_0[8],norm_landmarks_0[12])
            # 지우기, 띄어쓰기 기능 추가
            thumb_ring_touching, thumb_pinky_touching = check_fingers_touching(norm_landmarks_0[4],norm_landmarks_0[16],norm_landmarks_0[20])
            # 주먹을 줬는가? -> 왼손이 'ㅎ'일경우 -> 쌍자음 기능으로 활용
            rock_paper_idx = np.argmax(inference_model(norm_landmarks_0, model_consonant))
            rock_paper_results = Target[rock_paper_idx]['eng']
            

            # 엄지 검지 끝이 터치 : 자음
            if thumb_index_touching:
                # 검지랑 터치시 왼손에 초록색 원 표시
                draw_hand_circle(color_image, hand_landmarks_0, True, (0, 255, 0))

                # Hand Sign Recognition (오른손)
                output = inference_model(norm_landmarks_1, model_consonant)
                predict_idx = np.argmax(output)
                results = Target[predict_idx]['eng']
                cv2.putText(color_image,str(results) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (5, 0, 255), 3)


            # 엄지 중지 끝이 터치 : 모음
            if thumb_middle_touching:
                # 엄지랑 터치시 오른손에 빨간색 원 표시
                draw_hand_circle(color_image, hand_landmarks_0, True, (255, 0, 255))

                # Hand Sign Recognition (오른손)
                output = inference_model(norm_landmarks_1, model_vowel)
                predict_idx = np.argmax(output) + 14
                results = Target[predict_idx]['eng']
                cv2.putText(color_image,str(results) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (5, 0, 255), 3)
        

            # 왼손이 ㅎ이면 : 쌍자음
            if rock_paper_results == 'hieut':
                rock = True
                draw_hand_circle(color_image, hand_landmarks_0, True, (0, 0, 255),THUMB_TIP_IDX=0)
                
                # Hand Sign Recognition (오른손)
                output = inference_model(norm_landmarks_1, model_consonant)
                predict_idx = np.argmax(output)
                results = Target[predict_idx]['eng']
                cv2.putText(color_image,str(results) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (5, 0, 255), 3)
            else:
                rock = False




        if thumb_index_touching_list == [True, True, True, False]:
            print('consonant!')
            # 자음을 기록합니다.
            recorded_letters += Target[predict_idx]['key']

        if thumb_middle_touching_list == [True, True, True, False]:
            print('vowel!')
            # 모음을 기록합니다.
            recorded_letters += Target[predict_idx]['key']

        if thumb_ring_touching_list == [True, True, True, False]:
            print('space!')
            # 띄어쓰기 기능
            recorded_letters += ' '

        if thumb_pinky_touching_list == [True, True, True, False]:
            print('delete!')
            # 지우기 기능
            recorded_letters = recorded_letters[:-1]
            
        if rock_list == [True, True, True, False]:
            print('double consonant!')
            # 쌍자음을 기록합니다.
            recorded_letters += make_double_consonant(Target[predict_idx]['key'])


        thumb_index_touching_list.pop(0)
        thumb_middle_touching_list.pop(0)
        thumb_ring_touching_list.pop(0)
        thumb_pinky_touching_list.pop(0)
        rock_list.pop(0)

        thumb_index_touching_list.append(thumb_index_touching)
        thumb_middle_touching_list.append(thumb_middle_touching)
        thumb_ring_touching_list.append(thumb_ring_touching)
        thumb_pinky_touching_list.append(thumb_pinky_touching)
        rock_list.append(rock)
                

        white_background = np.ones_like(color_image) * 255








        if len(recorded_letters) == 0:
            text_img = draw_text(white_background, "Express!", (50, 50))
        else:
            if len(recorded_letters)>2:
                recorded_letters = vowel_compensation(recorded_letters)
            text_img = draw_text(white_background, join_jamos(recorded_letters), (50, 50))
            

        

        combined_image = np.hstack([color_image, text_img])



            

        cv2.imshow('Webcam Feed', combined_image)






        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): # q를 누르면 종류
            break

        
                


                



finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
