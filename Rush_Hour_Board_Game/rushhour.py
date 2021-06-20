import sys
import copy
import string
import random

# ----------------------------------------------------------------------------------
# Board Class
# ----------------------------------------------------------------------------------
class Board:
    def __init__(self, board_state, cars=[]):
        self.board_state = []
        self.cars = cars
        self.id = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=6))

        if (isinstance(board_state, str)):
            self.parse_state(board_state)

        if (isinstance(board_state, list)):
            self.board_state = board_state.copy()

    # FUNCTION - Parse boardstate and cars
    def parse_state(self, board_state):
        board_state = board_state.split('|')
        for i in range(6):
            self.board_state.append(list(board_state[i]))
            for c in list(board_state[i]):
                self.cars.append(c)
        self.cars = list(set(self.cars))
        self.cars.remove(" ")
        self.cars.sort(reverse=True)

    # FUNCTION - Check if the board is in end state
    def done(self):
        if (self.board_state[2][4] + self.board_state[2][5] == "xx"):
            return True
        return False

    # FUNCTION - Returns the car position in the board
    def car_position(self, car):
        position = []
        for i in range(len(self.board_state)):
            for j in range(len(self.board_state[i])):
                if self.board_state[i][j] == car:
                    position.append((i, j))
        return position

    # FUNCTION - Returns true if car can move horitzonally
    def move_horizontal(self, car):
        position = self.car_position(car)
        car_head = position[0]
        for pos in position:
            if (pos[0] != car_head[0]):
                return False
        return True

    # FUNCTION - Returns all possible moves for a car
    def possible_moves(self, pos, horizontal):
        possible_moves = []

        if (horizontal):
            direction = pos[0][0]
            car_head = max(pos)[1] + 1
            car_tail = min(pos)[1] - 1

            while (car_tail > -1) and (self.board_state[direction][car_tail] == " "):
                possible_moves.append((direction, car_tail))
                car_tail -= 1

            while (car_head < 6) and (self.board_state[direction][car_head] == " "):
                possible_moves.append((direction, car_head))
                car_head += 1

        else:
            direction = pos[0][1]
            car_head = max(pos)[0] + 1
            car_tail = min(pos)[0] - 1

            while (car_tail > -1) and (self.board_state[car_tail][direction] == " "):
                possible_moves.append((car_tail, direction))
                car_tail -= 1

            while (car_head < 6) and (self.board_state[car_head][direction] == " "):
                possible_moves.append((car_head, direction))
                car_head += 1

        return possible_moves

    # FUNCTION - Returns board objects for all possible for a cars
    def next_for_car(self, car):
        next_moves = []
        pos = self.car_position(car)
        hr = self.move_horizontal(car)
        moves = self.possible_moves(pos, hr)

        for move in moves:
            B = self.clone()
            B.move(car, move)
            next_moves.append(B)

        return next_moves

    # FUNCTION - Returns board objects for all possible for all cars
    def next(self):
        possible_boards = []
        for car in self.cars:
            next_moves = self.next_for_car(car)
            for next_move in next_moves:
                possible_boards.append(next_move)
        if (len(possible_boards) > 0):
            # self.print(possible_boards)
            return possible_boards
        return None

    # FUNCTION - Returns distance between two cars
    def distance(self, p1, p2):
        P = (p2[0] - p1[0], p2[1] - p1[1])
        P = (P[0]) + (P[1])
        return P

    # FUNCTION - Moves a card in the board
    def move(self, car, move_to):
        new_pos = []
        current_pos = self.car_position(car)
        move_horizontal = self.move_horizontal(car)

        car_head = max(current_pos)
        car_tail = min(current_pos)

        L = self.distance(car_head, move_to)
        U = self.distance(car_tail, move_to)

        move_distance = U if (abs(U) < abs(L)) else L

        if (move_horizontal):
            for p in current_pos:
                new_pos.append((p[0], p[1]+move_distance))
                self.board_state[p[0]][p[1]] = " "
        else:
            for p in current_pos:
                new_pos.append((p[0] + move_distance, p[1]))
                self.board_state[p[0]][p[1]] = " "

        for p in new_pos:
            self.board_state[p[0]][p[1]] = car

    # FUNCTION - Returns the clone of the board
    def clone(self, board_state=None, cars=None):

        if board_state == None:
            board_state = self.board_state
        if cars == None:
            cars = self.cars

        state = copy.deepcopy(board_state)
        clone = Board(state, cars)
        return clone

    # FUNCTION - Prints the boards in rows
    def print(self, boards=[]):
        # SETUP
        if len(boards) == 0:
            boards.append(self)
        width = len(boards)
        height = 6
        # PRINT
        print(" ------ " * width)
        for i in range(height):
            for board in boards:
                bar = " " if (i == 2) else "|"
                print("|"+"".join(board.board_state[i]), end=bar)
            print("")
        print(" ------ " * width)

    # OVERRIDE - Equals operator
    def __eq__(self, obj):
        if not (isinstance(obj, Board)):
            return False

        if (self.board_state == obj.board_state):
            if (self.cars == obj.cars):
                return True

        return False

    # OVERRIDE - Hash function
    def __hash__(self):
        return hash((self.id))

    '''
    PATH ALGORITHMS
    Random Walk | BFS | A*
    '''

    # ----------------------------------------------------------------------------------
    # Random Walk
    # ----------------------------------------------------------------------------------
    def random(self, N=10):

        path = Path()
        pathAdd = [self.clone()]

        for _ in range(N):
            current = pathAdd[-1].clone()

            if (current.done()):
                break

            possible_moves = current.next()

            if (possible_moves != None):
                random_pick = random.randint(0, len(possible_moves)-1)
                random_pick = possible_moves[random_pick]
                pathAdd.append(random_pick)
            else:
                break

        path.addPath(pathAdd)
        path.last()

    # ----------------------------------------------------------------------------------
    # Breadth First Search - BFS
    # ----------------------------------------------------------------------------------
    def bfs(self):
        visited = []
        queue = [self.clone()]
        path = Path()

        while(len(queue) > 0):
            # Pop first node
            current = queue.pop(0)

            # Add Path
            path.add(current, True)
            
            # Path Found
            if current.done():
                break
            
            # Add to visited
            if current not in visited:
                visited.append(current)
            
            # Generate possible moves
            possible_moves = current.next()

            # Perform BFS
            for board in possible_moves:
                path.addChildren(current, board)
                if (board not in visited and board not in queue):
                    queue.append(board)

        print(len(visited))

    # ----------------------------------------------------------------------------------
    # A*
    # ----------------------------------------------------------------------------------
    def A_cost(self, parent):
        return parent["g"] + 1

    def A_heuristic(self, board):
        position = board.car_position("x")
        goal = [(2, 4), (2, 5)]
        obstacle = 0
    
        gap = int(goal[1][1] - position[1][1])/2
        if (gap > 0):
            if (board.board_state[2][position[1][1]+1] != " "):
                obstacle += 1

        A = abs(goal[1][1] - position[1][1])
        B = abs(goal[0][1] - position[0][1])
        heuristic = A + B + obstacle
        return(heuristic)

    def A_checkNodeInList(self, node, node_list):
        for N in node_list:
            if (node["board"] == N["board"]):
                if(node["f"] > N["f"]):
                    return True
        return False

    def Astar(self):
        # Start Node
        start = {
            "board": self.clone(),
            "parent": None,
            "f": 0,
            "g": 0,  # cost
            "h": self.A_heuristic(self)  # heuristic
        }

        open_list = [start]
        closed_list = []
        path = Path()

        while(len(open_list) > 0):
            # Pop the node with lowest F
            open_list = sorted(open_list, key=lambda B: B['f'])
            currentNode = open_list.pop(0)
            current = currentNode["board"]

            # Add to path
            path.add(current, True)

            # Path Found
            if (current.done()):
                break
            
            # Add node to closed list
            closed_boards = [B["board"] for B in closed_list]
            if current not in closed_boards:
                closed_list.append(currentNode)

            # Generate possible moves
            possible_moves = current.next()

            # Perform A*
            for board in possible_moves:
                path.addChildren(current, board)
                cost = self.A_cost(currentNode)
                heuristic = self.A_heuristic(board)
                node = {
                    "board": board,
                    "parent": current,
                    "f": cost + heuristic,
                    "g": cost,
                    "h": heuristic
                }
                # Add child to open list if doesn't exist f is lower
                if not self.A_checkNodeInList(node, closed_list):
                    if not self.A_checkNodeInList(node, open_list):
                        open_list.append(node)

        print(len(closed_list))

# ----------------------------------------------------------------------------------
# Path Class - Handles Creating Path
# ----------------------------------------------------------------------------------
class Path:
    def __init__(self):
        self.path = []
        self.tracker = {}

    def addPath(self, path):
        self.path.append(path)

    def add(self, node, print_mode=False):
        if (len(self.tracker) == 0):
            self.tracker[node] = None

        path = self.formPath(node)
        self.addPath(path)

        if print_mode:
            self.print(path)

    def addChildren(self, parent, child):
        self.tracker[child] = parent

    def formPath(self, node):
        path = []
        while node:
            path.insert(0, node)
            node = self.tracker[node]
        return path

    def last(self):
        if (len(self.path) > 0):
            self.print(self.path[-1])

    def print(self, path=None):
        # Process path to print
        while(len(path) > 0):
            boards = []
            for _ in range(6):
                if (len(path) == 0):
                    break
                boards.append(path.pop(0))
            self.print_row(boards)
        print()

    def print_row(self, boards):
        width = len(boards)
        height = 6
        # PRINT
        print(" ------ " * width)
        for i in range(height):
            for board in boards:
                bar = " " if (i == 2) else "|"
                print("|"+"".join(board.board_state[i]), end=bar)
            print("")
        print(" ------ " * width)


# ----------------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------------
if __name__ == "__main__":

    arguments_length = len(sys.argv)

    if(arguments_length < 2):
        print("\nInvalid Command.\nPlease enter atleast 1 argument.\n")
        sys.exit()

    if(arguments_length > 2):
        board_state = sys.argv[2]
    else:
        board_state = "  o aa|  o   |xxo   |ppp  q|     q|     q"

    command = sys.argv[1]
    board = Board(board_state)

    if (command == "print"):
        board.print()

    elif (command == "done"):
        print(board.done())

    elif (command == "next"):
        board.print(board.next())

    elif (command == "random"):
        board.random()

    elif (command == "bfs"):
        board.bfs()

    elif (command == "astar"):
        board.Astar()

    else:
        print("\nInvalid command.")
        print("Valid commands: print | done | next\n")