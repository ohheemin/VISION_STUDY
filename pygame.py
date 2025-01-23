import pygame             # pygame라이브러리 불러옴
from math import *        # math 모듈에 포함된 모든 함수와 상수를 불러옴
from pygame.locals import * # pygame.locals 모듈에 포함된 모든 상수 불러옴(ex) QUIT)
import numpy as np        # numpy라이브러리(배열계산 및 선형대수연산에 사용, 여기서는 변환행렬을 사용해 로컬에서 글로벌 좌표로 변환할 때 사용)을 불러옴, 이를 np라는 별칭으로 사용
import time               # time 모듈을 불러옴(시간과 관련된 작업처리, 여기서는 목표지점에 도달한 후 1초정지하는 타이머에서 사용) ex) time.time()

WHITE = (255, 255, 255)   #(R,G,B)인데 0~255의 값이 있다. 
BLACK = (0, 0, 0)         
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init() #pygame 라이브러리를 초기화하는 함수(pygame을 사용하기전에 반드시 호출해야함)
# 화면의 범위는 가로 500, 세로 500
WIDTH = 500
HEIGHT = 500

screen = pygame.display.set_mode((WIDTH, HEIGHT))  #주어진 크기의 화면 생성
clock = pygame.time.Clock() #프레임속도를 제어하는 시계 객체 생성
rate = 100     #프레임:100

# 목표 좌표
GOAL = [[150, 300], [245, 400], [75, 75]]   # 마지막 목표는 출발점
        #(0.0),(0,1) (1,0),(1,1) (2,0),(2,1)

class Car:
    def __init__(self, initial_location):
        self.x, self.y = initial_location #자동차의 현재 위치 설정 차례대로 X좌표, Y좌표

        self.length = 15   #자동차의 길이
        self.width = 10    #자동차의 너비
        self.tread = 10    #자동차의 바퀴간의 간격
        self.wheel_radius = 1  #자동차 바퀴의 반지름

        self.heading = 0 #자동차가 현재 바라보는 방향 단위는 라디안이며 X축을 기준으로 반시계방향이 양수

        self.speed = 0   #자동차의 속도
        self.steer = 0   # 자동차의 조향각
        self.predict_time = 0.01  #이동시 예측 계산을 위한 시간간격

        # ↓ ↓ ↓ ↓ ↓ ↓ init 에서 활용하고 싶은 인스턴스 변수가 있을 시 이곳에 작성 ↓ ↓ ↓ ↓ ↓ ↓
        self.step = 0             #step: 현재목표지점을 추적 (0: 첫번째 목표지점(150,300)추적, 1: 두번째 목표지점(245,400) 추적, 2: 마지막 목표지점(75,75) 추적)
        self.stop_time = None     #stop_time: 목표지점에서 정지하는 시간을 기록(초기값은 0, 목표에 도달하면 시간을 기록하고 1초동안 정지상태 유지)
        self.end_flag = False     #end_flag:시뮬레이션 종료 상태를 나타내는 플래그(True면 차량이 모든 목표지점에 도달하고 정지한 상태, 모두 완료하면 True로 설정)      
        # ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑

    # noinspection PyMethodMayBeStatic
    def convert_coordinate_l2g(self, d_x, d_y, d_theta):                    # 차량의 local -> global 좌표 변환
        trans_matrix = np.array([[cos(d_theta), -sin(d_theta), 0],          # 변환 행렬을 이용해 local(d_x,d_y,d_theta)을 global로 변환
                                 [sin(d_theta), cos(d_theta), 0],
                                 [0, 0, 1]])
        return np.dot(trans_matrix, np.transpose([d_x, d_y, d_theta]))     #행렬곱(np.dot)를 통해 변환한 값 반환

    def generate_predict_pose(self):       # 차량의 다음 예상 위치를 계산
        self.steer = min(max(self.steer, -30), 30)  # 조향각을 -30도~+30도 제한
        tan_dis = self.speed * self.predict_time  # 접선 이동 거리 (= 호의 길이)  차량이 predict_time동안 이동한 거리를 계산
        R = self.length / tan(radians(-self.steer)) if self.steer != 0 else float('inf')  # 곡률 반경계산 조향각이 0이면 inf(무한대)로 설정
        d_theta = tan_dis / R   #d_theta 계산(방향 변화량)

        predict_pose = [tan_dis, 0.0, 0.0] if R == float('inf') else [R * sin(d_theta), R * (1 - cos(d_theta)), d_theta] # 예상 위치 계산
        d_x, d_y, heading = np.transpose(self.convert_coordinate_l2g(predict_pose[0], predict_pose[1], d_theta + self.heading)) #예상 위치를 글로벌 좌표로 변환
        return self.x + d_x, self.y + d_y, heading

    def move(self):          #차량의 실제 위치 업데이트,generate_predict_pose를 호출해서 계산한 값을 self.x, self.y, self.heading에 반영
        self.x, self.y, self.heading = self.generate_predict_pose()

    def GUI_display(self):     #pygame을 사용해 차량과 목표지점을 화면에 그림
        pygame.draw.circle(screen, GREEN, [GOAL[2][0], 500 - GOAL[2][1]], 10) #pygame의자표계(위가 y축 음의 방향)와 맞추기 위해 500-GOAL[2][1]
        pygame.draw.circle(screen, BLUE, [GOAL[0][0], 500 - GOAL[0][1]], 10)
        pygame.draw.circle(screen, BLUE, [GOAL[1][0], 500 - GOAL[1][1]], 10)

        a = atan2(self.width, self.length) #차량모양의 대각선과 방향(heading)의 각도
        b = sqrt(self.length ** 2 + self.width ** 2) / 2 #차량의 대각선 길이의 절반
        corner1 = [self.x + cos(self.heading - a) * b, 500 - (self.y + sin(self.heading - a) * b)]
        corner2 = [self.x + cos(self.heading + a) * b, 500 - (self.y + sin(self.heading + a) * b)]
        corner3 = [self.x + cos(self.heading + pi - a) * b, 500 - (self.y + sin(self.heading + pi - a) * b)]
        corner4 = [self.x + cos(self.heading + pi + a) * b, 500 - (self.y + sin(self.heading + pi + a) * b)]    # 차들의 모서리 좌표
        pygame.draw.polygon(screen, RED, [corner1, corner2, corner3, corner4])

    # set_motor_value 함수 내에서 자유롭게 작성
    def set_motor_value(self):           #차량의 속도와 조향각도를 계산하는 함수
        if self.end_flag:           
            return

        target_x, target_y = GOAL[self.step]  #현재 목표 지점의 좌표 가져오기
        dx = target_x - self.x                
        dy = target_y - self.y
        distance = sqrt(dx ** 2 + dy ** 2)  #차량과 목표지점 사이의 x,y거리를 계산

        if distance < 1:  # 목표 지점 도달 조건
            if self.stop_time is None:
                self.stop_time = time.time()   #처음도달시 현재 시간을 기록(self. stop_time)
            elif time.time() - self.stop_time >= 1:  # 1초 정지 후 다음 목표로
                self.stop_time = None
                self.step += 1
                if self.step >= len(GOAL):
                    self.end_flag = True     # 모든 목표를 완료하면 True로바뀌면서 동작종료
                    return
        else:
            self.stop_time = None

        # 목표 방향 계산
        target_heading = atan2(dy, dx)    
        heading_diff = target_heading - self.heading   #각도 차이

        # -pi ~ pi 범위로 각도 정규화   최종적으로 가장짧은이동으로 목표대상까지 가기위함
        if heading_diff > pi:          #ex)270도면 변환했을 때
            heading_diff -= 2 * pi     #270도-360도=-90도  
        elif heading_diff < -pi:
            heading_diff += 2 * pi

        # steering 값 설정 (-30 ~ 30 제한)
        self.steer = -degrees(heading_diff)    # heading_diff는
        self.steer = max(-30, min(30, self.steer))

        # 속도 설정 (정지 상태가 아닐 때만)
        if self.stop_time is None:                
            self.speed = 80    #움직일 때 속도
        else:
            self.speed = 0
        
        

def main():
    car = Car([GOAL[2][0], GOAL[2][1]])  # 시작위치 (75,75)에 차량 생성
    
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:    # 창을 닫는 이벤트(QUIT)가 발생하면 프로그램 종료
                pygame.quit()
                return 0

        car.set_motor_value()    #위에 있는 함수들 적용
        car.move()               #위에 있는 함수들 적용

        screen.fill(WHITE)      #배경 WHITE
        car.GUI_display()      #차량과 목표지점 표시

        pygame.display.flip()   #화면 갱신
        clock.tick(rate)      #프레임
        


if __name__ == '__main__':
    main()