# Student agent: Add your own agent here

from agents.agent import Agent
from store import register_agent
import sys

# additional imports
from copy import deepcopy
from numpy import random
import math
from time import time

def random_walk(chess_board, my_pos, adv_pos, max_step):
    ori_pos = deepcopy(my_pos)
    # Moves (Up, Right, Down, Left)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    steps = random.randint(0, max_step + 1)

    # Random Walk
    for _ in range(steps):
        r, c = my_pos
        direction = random.randint(0, 4)
        m_r, m_c = moves[direction]
        my_pos = (r + m_r, c + m_c)

        # Special Case enclosed by Adversary
        k = 0
        while chess_board[r, c, direction] or my_pos == adv_pos:
            k += 1
            if k > 300:
                break
            direction = random.randint(0, 4)
            m_r, m_c = moves[direction]
            my_pos = (r + m_r, c + m_c)

        if k > 300:
            my_pos = ori_pos
            break

    # Put Barrier
    direction = random.randint(0, 4)
    r, c = my_pos
    while chess_board[r, c, direction]:
        direction = random.randint(0, 4)

    return my_pos, direction


def rollout(chess_board, my_pos, adv_pos, max_step):
    temp_chessboard = deepcopy(chess_board)
    turn = 1
    endgame, my_score, adv_score = check_endgame(temp_chessboard, my_pos, adv_pos)
    while not endgame:
        # print("loop")
        if turn == 0:
            # print("turn0")
            my_pos, direction = random_walk(temp_chessboard, my_pos, adv_pos, max_step)
            x, y = my_pos
        else:
            # print("turn1")
            adv_pos, direction = random_walk(temp_chessboard, adv_pos, my_pos, max_step)
            x, y = adv_pos
        set_barrier(temp_chessboard, x, y, direction)
        # print((my_pos, temp_chessboard[my_pos[0]][my_pos[1]]))
        # print((adv_pos, temp_chessboard[adv_pos[0]][adv_pos[1]]))
        endgame, my_score, adv_score = check_endgame(temp_chessboard, my_pos, adv_pos)
        # print((endgame, my_score, adv_score))
        turn = 1 - turn
    # print("endgame")
    return my_score, adv_score

def set_barrier(chess_board, r, c, dir):
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    # Set the barrier to True
    chess_board[r, c, dir] = True
    # Set the opposite barrier to True
    move = moves[dir]
    chess_board[r + move[0], c + move[1], opposites[dir]] = True

def remove_barrier(chess_board, r, c, dir):
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    # Set the barrier to True
    chess_board[r, c, dir] = False
    # Set the opposite barrier to True
    move = moves[dir]
    chess_board[r + move[0], c + move[1], opposites[dir]] = False

def check_endgame(chess_board, p0_pos, p1_pos):
    board_size = len(chess_board)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    # Union-Find
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                moves[1:3]
            ):  # Only check down and right
                if chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))

    p0_r = find(tuple(p0_pos))
    p1_r = find(tuple(p1_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    return True, p0_score, p1_score


class BoardState():
    def __init__(self, chess_board, p0_pos, p1_pos, max_step):
        self.chess_board = chess_board
        self.p0_pos = p0_pos
        self.p1_pos = p1_pos
        self.max_step = max_step

    def equal(self, state):
        if (self.chess_board == state.chess_board).all() and self.p0_pos == state.p0_pos and self.p1_pos == state.p1_pos and self.max_step == state.max_step:
            return True


class MCTnode():
    def __init__(self, chess_board, my_pos, adv_pos, max_step, parent=None):
        self.chess_board = chess_board
        self.board_size = len(self.chess_board)
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.parent = parent # None for root node

        endgame, _, _ = check_endgame(self.chess_board, self.my_pos, self.adv_pos)
        if endgame:
            self.is_terminal = True
        else:
            self.is_terminal = False

        # to avoid division by zero error
        self.N = 1
        self.U = 1

        self.children = dict() # {move:MCTnode}

    # helper functions (selection policies)
    def surrounding_barriers(self, pos, range):
        r, c = pos
        rmin = min(0, r-range)
        rmax = max(r+range, self.board_size-1)
        cmin = min(0, c - range)
        cmax = max(c + range, self.board_size - 1)
        cnt = 0
        for i in range(rmin, rmax+1, 2):
            for j in range(cmin+1, cmax+1, 2):
                cnt += self.chess_board[i][j].sum()
        return cnt

    def expand(self):
        # with selection policy, branching factor = 10
        i = 0
        counter = 0
        while i < 15:
            counter += 1
            (r1, c1), dir = random_walk(self.chess_board, self.my_pos, self.adv_pos, self.max_step)
            set_barrier(self.chess_board, r1, c1, dir)
            if self.chess_board[r1][c1].sum() > 2 and counter < 100:
                remove_barrier(self.chess_board, r1, c1, dir)
                continue
            child = MCTnode(deepcopy(self.chess_board), (r1, c1), self.adv_pos, self.max_step)
            self.children[((r1, c1), dir)] = child
            remove_barrier(self.chess_board, r1, c1, dir)
            i += 1

    def init_simulations(self):
        # for each childnode, do 5 initial rollouts
        for move, childnode in self.children.items():
            (r1, c1), dir = move
            set_barrier(self.chess_board, r1, c1, dir)
            for _ in range(10):
                childnode.N += 1
                self.N += 1
                s0, s1 = rollout(self.chess_board, (r1, c1), self.adv_pos, self.max_step)
                if s0 > s1: childnode.U += (s0 - s1)
                # if s0 > s1: childnode.U += 2*(s0 - s1) / self.board_size
            remove_barrier(self.chess_board, r1, c1, dir)

    def select(self):
        # return the childnode for expansion
        # using UCB1 formula: U(n)/N(n) + sqrt(2)*sqrt(logN(parent(n))/N(n))
        l = []
        for childnode in self.children.values():
            UCB1 = childnode.U/childnode.N + math.sqrt(2) * math.sqrt(math.log(self.N)/childnode.N)
            l.append([UCB1, childnode])
        l.sort(key=lambda x:x[0], reverse=True)
        return l[0][1]

    def simulate(self):
        # number of simulation that can perform in limited time
        num_simulation = 2000//self.board_size
        for _ in range(num_simulation):
            node = self.select()
            s0, s1 = rollout(node.chess_board, node.my_pos, node.adv_pos, node.max_step)
            if s0 > s1: node.U += 1

    def final_move(self):
        self.expand()
        self.simulate()
        movelist = []
        for move, childnode in self.children.items():
            win_percentage = childnode.U/childnode.N
            movelist.append((win_percentage, move))
        movelist.sort(key=lambda x:x[0], reverse=True)
        return movelist[0][1]


@register_agent("student_agent2")
class StudentAgent2(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent2, self).__init__()
        self.name = "StudentAgent2"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True  # enable autoplay

    def step(self, chess_board, my_pos, adv_pos, max_step):
        start = time()
        curr_node = MCTnode(chess_board, my_pos, adv_pos, max_step)
        move = curr_node.final_move()
        print(time()-start)
        return move

