# COMP424-Final-Project
Implemented AI agents to play a game, used the idea of Monte Carlo Tree Search.

MCTS Algorithm Overview:
From the current game state, decide how to proceed next
Nodes represent a game state (board layout)

1. Selection. Select a leaf node (starting from the root node, choose the child node with the highest UCB value each time until reaching a leaf node, balancing exploration and exploitation)
2. Expansion/Simulation. 
    1. Simulation: If no simulation has been performed at the leaf node, perform a simulation/playout/rollout: let two players make random moves until a winner is determined (game ends), then backpropagate
    2. Expansion: If a simulation has been performed at the leaf node before, expand the node by adding all possible next game states as child nodes, and then choose any one of them for simulation
3. Backpropagation: The simulation will generate a value v. Along the path back to the root node, the n value (number of rollouts) for each node will be incremented by 1, but only the v value of the nodes where the winner took turn will be incremented by v.

Result:
Continue the tree expansion until the time's up, then select the child node of the root node with the highest v value as the next move.

Note:
Store/remember the tree to continue building the tree in the limited time of each turn, and accumulate the calculation.

MCTS算法概览：
从当前的游戏状态，决定下一步应该怎么走
节点代表一个游戏状态（棋盘布局）
1. 选取一个叶子节点（从根节点，每次选取最大化UCB值的子节点，直到一个叶子节点，平衡发现合验证）
2. 如果在该叶子节点没有进行过预演，则直接进行预演：两个玩家随机落子直到分出胜负，进入第四步向上回溯预演结果
3. 如果在该叶子节点之前有过预演，对该节点进行扩张，即将所有可能的下一步游戏状态添加为子节点，然后选取任意一个进行预演
4. 向上回溯：预演会产生一个价值v，沿着回回到根节点的路径，所有路径中的节点n+1（预演回合数），但只有预演中获胜的一方玩家的落子节点的价值总数会+v

结果：
- 计算运行直到规定时间用完，从根节点选取v值最大的子节点作为下一步。
- 存储一个蒙特卡洛树，可以在每次落子回合规定时间内继续建树，累积计算结果
