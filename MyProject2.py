import cv2
import face_recognition as frc
import numpy as np

while(True):
    while(True) :
        SelectImg = input("기준 이미지를 선택하세요. (1.감스트, 2.로다주, 3.젊은 로다주) >>")
        if(SelectImg == '1') :
            StandardImg = frc.load_image_file('faces/gamst.jpg'); break # Test 이미지 가져오기
        elif(SelectImg == '2') :
            StandardImg = frc.load_image_file('faces/Robert-Downey-Jr2.jpg'); break  # Test 이미지 가져오기
        elif(SelectImg == '3') :
            StandardImg = frc.load_image_file('faces/dong_jun.jpg'); break  # Test 이미지 가져오기
        else :
            print("잘못된 입력입니다.")
    while(True):
        print("선택한 기준 이미지가 맞습니까?"+"(현재 선택:"+SelectImg+")")
        answer = input("입력(y/n) >>")
        if(answer == 'y'):
            break
        elif(answer == 'n'):
            print("그럼 다시 선택하십시오.");break
        else:
            print("잘못된 입력입니다.")
    if (answer == 'y'):
        break

StandardImg = cv2.cvtColor(StandardImg, cv2.COLOR_BGR2RGB)  # RGB로 변환

# 얼굴의 위치 같게 만들기
faceLoc = frc.face_locations(StandardImg)[0]
encodeStandard = frc.face_encodings(StandardImg)[0]  # 감지할 얼굴 인코딩, 첫번째 요소만 가져오기
cv2.rectangle(StandardImg, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)  # 우리가 얼굴을 감지한 위치를 확인하기 위해 사각형을 이미지에 그림

cap = cv2.VideoCapture(0)  # 노트북 웹캠을 카메라로 사용
cap.set(3, 640)  # 너비
cap.set(4, 480)  # 높이
while cap.isOpened():
    ret, frame = cap.read()  # 사진 촬영
    frame = cv2.flip(frame, 1)  # 좌우 대칭

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) == ord("q"):
        cv2.imwrite('faces/self img.jpg', frame)  # 사진 저장
        break

cap.release()
cv2.destroyAllWindows()

TestImg = frc.load_image_file('faces/self img.jpg') # Test 이미지 가져오기
TestImg = cv2.cvtColor(TestImg, cv2.COLOR_BGR2RGB )#Tset 이미지 RGB로 변환


faceLocTest = frc.face_locations(TestImg)[0]
encodeTest = frc.face_encodings(TestImg)[0]  # Test이미지에 대한 첫번째 요소만 가져오기
cv2.rectangle(TestImg, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)  # Test 이미지에 사각형 이미지


results = frc.compare_faces([encodeStandard], encodeTest)  # 인코딩 이미지와 Test 이미지 간의 비교하기
error_rate = frc.face_distance([encodeStandard], encodeTest)  # 이미지 유사성 알기, (==> 얼굴 간의 오차 느낌인거 같음)
print("확인 결과 :", results, "오차율 :", error_rate)  # 두 개의 이미지가 서로 같으면 True, 다른 이미지일 경우 False를 출력한다

# 이미지에 대한 결과랑 유사성을 Test 이미지에 명시해주기, round()는 유사성을 소수점 둘째짜리로 반올림 한다는 뜻
cv2.putText(TestImg, f'{results} {round(error_rate[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

StandardImg = cv2.resize(StandardImg, (360, 360))
TestImg = cv2.resize(TestImg, (360, 360))
cv2.imshow('Standard Img', StandardImg)  # 인코딩 이미지 불러오기
cv2.imshow('Test Img', TestImg)  # Test 이미지 불러오기
cv2.waitKey(0)
