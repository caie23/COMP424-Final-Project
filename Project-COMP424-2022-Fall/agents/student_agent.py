# Student agent: Add your own agent here

from agents.agent import Agent
from store import register_agent
import sys

# additional imports
from copy import deepcopy
from numpy import random
import math
# from time import time

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
        if turn == 0:
            my_pos, direction = random_walk(temp_chessboard, my_pos, adv_pos, max_step)
            x, y = my_pos
        else:
            adv_pos, direction = random_walk(temp_chessboard, adv_pos, my_pos, max_step)
            x, y = adv_pos
        set_barrier(temp_chessboard, x, y, direction)
        endgame, my_score, adv_score = check_endgame(temp_chessboard, my_pos, adv_pos)
        turn = 1 - turn
    return my_score, adv_score

def set_barrier(chess_board, r, c, dir):
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    chess_board[r, c, dir] = True
    move = moves[dir]
    chess_board[r + move[0], c + move[1], opposites[dir]] = True

def remove_barrier(chess_board, r, c, dir):
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    chess_board[r, c, dir] = False
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

def surrounding_barriers(chess_board, pos):
    board_size = len(chess_board)
    size = board_size//4
    r, c = pos
    rmin = max(0, r-size)
    rmax = min(r+size, board_size-1)
    cmin = max(0, c - size)
    cmax = min(c + size, board_size - 1)
    cnt = 0
    for i in range(rmin, rmax+1, 2):
        for j in range(cmin+1, cmax+1, 2):
            cnt += chess_board[i][j].sum()
    return cnt

class MCTS():
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self.chess_board = chess_board
        self.board_size = len(self.chess_board)
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

        self.children = []

    def get_move(self):
        # greedy heuristics
        # 1. escape itself and trap the opponent
        moves = []
        (r, c), dir = random_walk(self.chess_board, self.my_pos, self.adv_pos, self.max_step)
        moves.append(((r, c), dir))
        for _ in range(50):
            (r, c), dir = random_walk(self.chess_board, self.my_pos, self.adv_pos, self.max_step)
            set_barrier(self.chess_board, r, c, dir)
            if self.chess_board[r, c].sum() > 2:
                remove_barrier(self.chess_board, r, c, dir)
                continue
            if self.chess_board[self.adv_pos[0], self.adv_pos[1]].sum() >= 2: return ((r, c), dir)
            moves.append(((r, c), dir))
            remove_barrier(self.chess_board, r, c, dir)
        # 2. maximize adv surrounding barriers and minimize my surrounding barriers
        l = []
        for move in moves:
            (r, c), dir = move
            set_barrier(self.chess_board, r, c, dir)
            l.append([surrounding_barriers(self.chess_board, self.adv_pos), surrounding_barriers(self.chess_board, self.my_pos), move])
            remove_barrier(self.chess_board, r, c, dir)
        l = sorted(l, key=lambda x: x[0], reverse=True)
        l = sorted(l, key=lambda x: x[1])
        return l[0][2]

    def expand_children(self):
        moves = set()
        i = 0
        # choose 10 children for expansion
        num_child = self.board_size*2
        num_simul = 200//num_child
        while i < num_child:
            move = self.get_move()
            (x, y), direction = move
            if move in moves: continue
            set_barrier(self.chess_board, x, y, direction)
            utility = 0
            # simulate 20 rollouts for a selected move
            for _ in range(num_simul):
                myscore, advscore = rollout(self.chess_board, (x, y), self.adv_pos, self.max_step)
                utility += (myscore-advscore)/self.board_size
            remove_barrier(self.chess_board, x, y, direction)
            self.children.append((utility, move))
            i += 1

    def find_move(self):
        # choose the move with the highest utility
        self.expand_children()
        self.children.sort(key=lambda x:x[0], reverse=True)
        return self.children[0][1]

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True  # enable autoplay

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # start = time()
        mcts = MCTS(chess_board, my_pos, adv_pos, max_step)
        move = mcts.find_move()
        # print(time()-start)
        return move

