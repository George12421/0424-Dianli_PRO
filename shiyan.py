import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


# ==========================================
# 1. 初始化单纯复形网络 (Random Simplicial Complex)
# ==========================================
def generate_simplicial_complex(N, k_edge, k_triangle):
    """
    生成一个包含成对连边和三角形的随机高阶网络
    """
    G = nx.erdos_renyi_graph(N, k_edge / N)
    triangles = set()

    # 随机添加三角形 (2-simplices)
    num_triangles_target = int((N * k_triangle) / 3)
    nodes = list(G.nodes())
    while len(triangles) < num_triangles_target:
        u, v, w = random.sample(nodes, 3)
        triplet = tuple(sorted([u, v, w]))
        if triplet not in triangles:
            triangles.add(triplet)
            G.add_edge(u, v)
            G.add_edge(v, w)
            G.add_edge(u, w)

    return G, list(triangles)


# ==========================================
# 2. 单步蒙特卡洛演化 (微观规则映射)
# ==========================================
def monte_carlo_step(G, triangles, states, params):
    dt, mu, beta1, beta_delta, omega1, omega_delta = params
    N = len(G.nodes())
    new_states = states.copy()

    # 预处理三角形状态，加速查询
    node_to_triangles = {i: [] for i in range(N)}
    for tidx, tri in enumerate(triangles):
        for node in tri:
            node_to_triangles[node].append(tidx)

    for i in list(G.nodes()):
        # ---- 恢复过程 ----
        if states[i] == 1:  # I节点
            if random.random() < mu * dt:
                new_states[i] = 0

        # ---- 感染与重连过程 ----
        elif states[i] == 0:  # S节点
            infected_neighbors = [neighbor for neighbor in G.neighbors(i) if states[neighbor] == 1]
            num_I_neighbors = len(infected_neighbors)

            # 统计高阶暴露状态
            in_low_risk = False  # {S, S, I}
            in_high_risk = False  # {S, I, I}

            for tidx in node_to_triangles[i]:
                tri = triangles[tidx]
                I_count = sum([states[n] for n in tri])
                if I_count == 1:
                    in_low_risk = True
                elif I_count == 2:
                    in_high_risk = True
                    # 高阶感染机制：暴露于 {S, I, I}
                    if random.random() < beta_delta * dt:
                        new_states[i] = 1

            # 低阶成对感染机制
            if new_states[i] == 0 and num_I_neighbors > 0:
                # 独立感染概率 1 - (1 - beta1*dt)^n
                prob_infect = 1 - (1 - beta1 * dt) ** num_I_neighbors
                if random.random() < prob_infect:
                    new_states[i] = 1

            # 状态依赖的适应性重连 (若没有被感染，则尝试逃避)
            if new_states[i] == 0 and num_I_neighbors > 0:
                # 判断当前S节点处于何种恐慌状态
                if in_high_risk:
                    rewire_prob = omega_delta * dt
                elif in_low_risk or num_I_neighbors > 0:
                    rewire_prob = omega1 * dt
                else:
                    rewire_prob = 0

                if random.random() < rewire_prob:
                    # 随机选择一个邻居I断开
                    target_I = random.choice(infected_neighbors)
                    G.remove_edge(i, target_I)
                    # 从全网寻找一个非邻居的S节点相连
                    S_nodes = [n for n in range(N) if states[n] == 0 and n != i and not G.has_edge(i, n)]
                    if S_nodes:
                        new_neighbor = random.choice(S_nodes)
                        G.add_edge(i, new_neighbor)

                    # 注意：为了简化仿真，此处代码处理了1-单纯形的重连，
                    # 严格的单纯形重连需要同步更新 triangles 列表中的节点。

    return new_states


# ==========================================
# 3. 绝热扫描主程序 (提取迟滞环散点)
# ==========================================
N = 1000  # 网络节点数，跑论文图时可以增大到 2000-5000
dt = 0.05
params_base = [dt, 1.0, 0.0, 18.0, 0.8, 0.2]  # [dt, mu, beta1, beta_delta, omega1, omega_delta]
beta1_range = np.linspace(0.01, 0.4, 40)  # beta1 扫描区间

print("初始化网络...")
G_init, triangles_init = generate_simplicial_complex(N, k_edge=4.0, k_triangle=2.0)

# ---- 正向扫描 (Invasion) ----
print("开始正向演化...")
states = np.zeros(N, dtype=int)
states[np.random.choice(N, int(N * 0.01), replace=False)] = 1  # 初始 1% 感染
P_forward = []

for b1 in beta1_range:
    params_base[2] = b1
    G = G_init.copy()

    # 弛豫时间 (达到稳态)
    for _ in range(300):
        states = monte_carlo_step(G, triangles_init, states, params_base)

    # 采样时间 (计算平均感染密度)
    rho = 0
    for _ in range(50):
        states = monte_carlo_step(G, triangles_init, states, params_base)
        rho += np.mean(states)
    P_forward.append(rho / 50)

# ---- 反向扫描 (Extinction) ----
print("开始反向演化...")
states = np.ones(N, dtype=int)  # 初始 100% 感染，强制处于高流行态
P_backward = []

for b1 in reversed(beta1_range):
    params_base[2] = b1
    G = G_init.copy()

    for _ in range(300):
        states = monte_carlo_step(G, triangles_init, states, params_base)

    rho = 0
    for _ in range(50):
        states = monte_carlo_step(G, triangles_init, states, params_base)
        rho += np.mean(states)
    P_backward.append(rho / 50)
P_backward.reverse()

# 可视化散点图
plt.figure(figsize=(8, 6))
plt.scatter(beta1_range, P_forward, color='red', marker='o', label='MC Forward')
plt.scatter(beta1_range, P_backward, color='blue', marker='s', label='MC Backward')
plt.xlabel(r'Transmission rate $\beta_1$', fontsize=14)
plt.ylabel('Steady-state Prevalence P', fontsize=14)
plt.title('Monte Carlo Simulation on Adaptive Simplicial Complex', fontsize=15)
plt.legend()
plt.grid(True)
plt.show()