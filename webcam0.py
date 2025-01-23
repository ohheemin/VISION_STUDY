import cv2

def open_webcam():
    # 웹캠 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미합니다. 다른 장치를 사용하려면 1, 2로 변경.
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    print("웹캠을 실행합니다. 'q'를 눌러 종료하세요.")
    
    while True:
        ret, frame = cap.read()  # 프레임 읽기
        
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        cv2.imshow("Webcam", frame)  # 프레임 보여주기
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()  # 캡처 객체 해제
    cv2.destroyAllWindows()  # 모든 창 닫기

# 웹캠 실행
if __name__ == "__main__":
    open_webcam()
