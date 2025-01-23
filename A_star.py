import numpy as np
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
import time

# 맵 크기 설정
map_size = (10, 10)

# 장애물 생성 함수이며, 랜덤으로 설치
def generate_obstacles(grid, start, goal, num_obstacles=20):
    rows, cols = grid.shape
    for _ in range(num_obstacles):
        x = random.randint(0, rows-1)
        y = random.randint(0, cols-1)
        # 장애물 위치는 시작점, 목표점과 다르다는 것을 명시
        while grid[x, y] == 1 or (x, y) == start or (x, y) == goal:
            x = random.randint(0, rows-1)
            y = random.randint(0, cols-1)
        grid[x, y] = 1
    return grid

# 장애물 환경 설정 (0과 1은 빈공간, 장애물)
grid = np.zeros(map_size)

# 시작점과 목표점
start = (0, 0)
goal = (9, 9)

# 장애물 랜덤 배치
grid = generate_obstacles(grid, start, goal, num_obstacles=15)

# 8방향으로 이동 가능함
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# A* 알고리즘 함수
def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_list = PriorityQueue()
    open_list.put((0, start))  # (f, (x, y)) -> f는 cost + heuristic
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_list.empty():
        _, current = open_list.get()

        # 목표에 도달하면 종료
        if current == goal:
            return reconstruct_path(came_from, current)

        # 이웃 탐색
        for neighbor in neighbors:
            neighbor_x = current[0] + neighbor[0]
            neighbor_y = current[1] + neighbor[1]
            neighbor_pos = (neighbor_x, neighbor_y)

            if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols and grid[neighbor_x, neighbor_y] == 0:
                tentative_g_score = g_score[current] + 1  # 각 이동은 1의 비용을 가짐

                if neighbor_pos not in g_score or tentative_g_score < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g_score
                    f_score[neighbor_pos] = g_score[neighbor_pos] + heuristic(neighbor_pos, goal)
                    open_list.put((f_score[neighbor_pos], neighbor_pos))

    return None  # 목표에 도달할 수 없으면 None 반환

# 휴리스틱 함수 (맨해튼 거리)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 경로 재구성 함수
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # 역순으로 반환

# 경로 계산
path = a_star(grid, start, goal)

# 경로 시각화 함수 (로봇이 이동하면서 경로를 따라가도록)
def plot_grid(grid, path, start, goal):
    fig, ax = plt.subplots()
    img = ax.imshow(grid, cmap='binary', origin='upper')  # imshow 반환 객체
    fig.colorbar(img)  # colorbar는 imshow 객체에서 호출해야 함
    ax.scatter(start[1], start[0], color='green', label="Start", marker='o')
    ax.scatter(goal[1], goal[0], color='red', label="Goal", marker='x')
    ax.legend()

    # 로봇이 경로를 따라가는 애니메이션 효과
    for i in range(len(path)):
        ax.plot([path[i][1]], [path[i][0]], color='red', marker='o')
        plt.draw()
        plt.pause(0.3)  # 0.1초 동안 대기하여 로봇이 이동하는 효과를 나타냄
        if i != len(path)-1:
            ax.plot([path[i+1][1]], [path[i+1][0]], color='blue', marker='o')

    plt.show()

# 시각화 실행
plot_grid(grid, path, start, goal)
