import math
import heapq

def dijkstra(graph, start):
    # 1. 初始化距离字典
    distances = {node: math.inf for node in graph}
    distances[start] = 0

    # 2. 初始化优先队列
    pq = [(0, start)]

    # 3. 主循环
    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # 4. 过滤过时信息
        if current_distance > distances[current_node]:
            continue

        # 5. 遍历邻居并松弛
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # 如果发现更短的路径
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
