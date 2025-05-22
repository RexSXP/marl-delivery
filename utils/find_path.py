import json
import os
from collections import deque


def find_path(path):
    matrix = []
    with open(path, 'r') as input_file:
        for line in input_file:
            row = list(map(int, line.strip().split()))
            matrix.append(row)

    map_position = []
    n, m = len(matrix), len(matrix[0])
    directions = [("U", (-1, 0)), ("L", (0, -1)), ("R", (0, 1)), ("D", (1, 0))]

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 0:
                map_position.append((i+1, j+1))

    # Create output directory if it doesn't exist
    os.makedirs("maps/paths", exist_ok=True)
    out_path = os.path.join("maps/paths", os.path.basename(path))
    
    with open(out_path, 'w') as out_file:
        for i in range(len(map_position)):
            dist = [[-1] * m for _ in range(n)]
            str_path = [[""] * m for _ in range(n)]
            queue = deque()

            start_i, start_j = map_position[i]
            queue.append((start_i, start_j))
            dist[start_i][start_j] = 0
            str_path[start_i][start_j] = ""

            while queue:
                x, y = queue.popleft()

                for (direc_move, (di, dj)) in directions:
                    pos_i = x + di
                    pos_j = y + dj
                    if 0 < pos_i < n and 0 < pos_j < m and dist[pos_i][pos_j] == -1 and matrix[pos_i][pos_j] == 0:
                        dist[pos_i][pos_j] = dist[x][y] + 1
                        str_path[pos_i][pos_j] = str_path[x][y] + direc_move
                        queue.append((pos_i, pos_j))
                        path_answer = {
                            "start": (start_i, start_j),
                            "target": (pos_i, pos_j),
                            "path": str_path[pos_i][pos_j]
                        }
                        out_file.write(json.dumps(path_answer) + '\n')

if __name__ == "__main__":
    path = "maps/map1.txt"
    find_path(path)