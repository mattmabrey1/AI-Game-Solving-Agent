import numpy
import queue

# A* Star Search algorithm that returns the path to the closest target each time and collects all
# targets to finish the level. Moving up is considered more expensive than going to the side so when picking a
# a target the y-coordinate is squared.


class AStarSearch:

    def __init__(self, move_grid, plat_grid, cur_r, cur_c, target_val):
        self.move_grid = move_grid
        self.plat_grid = plat_grid
        self.cur_r = cur_r
        self.cur_c = cur_c
        self.target_val = target_val
        self.target_c = 0
        self.target_r = 0
        self.p_que = queue.PriorityQueue()
        self.nodes_to_goal = []
        self.path_to_goal = []

        self.visited = numpy.zeros_like(self.move_grid)

    # Find_target_location method is used to get the row and column location of the target in the grid
    # so that we can use it in the heuristic method to get a heuristic value for any given node.
    # It also counts the number of cherries available to make sure it collects all bonus targets
    # before clearing the stage by eating the last cherry.
    def find_target_location(self):

        min_diff = 5000  # min straight line distance to a target
        num_of_cherry = 0  # number of cherries on the grid
        num_of_targets = 0  # number of total targets on the grid

        self.target_r = 0  # initialize to zero so we know if no target is found
        self.target_c = 0  # initialize to zero so we know if no target is found

        for row in self.move_grid:

            row_index = self.move_grid.index(row)

            for col in range(len(row)):

                if self.move_grid[row_index][col] in self.target_val and self.plat_grid[row_index][col].isActive:

                    num_of_targets += 1

                    if self.move_grid[row_index][col] == 8:
                        num_of_cherry += 1  # counting the number of cherries so we know not to get the last one
                                            # until all bonus target are eaten

        for row in self.move_grid:

            row_index = self.move_grid.index(row)

            for col in range(len(row)):

                # going up is more expensive so the row difference is squared
                diff = numpy.square(abs(self.cur_r - row_index)) + abs(self.cur_c - col)

                if self.move_grid[row_index][col] in self.target_val and diff < min_diff and self.plat_grid[row_index][
                    col].isActive:

                    if not (self.move_grid[row_index][col] == 8 and num_of_cherry == 1 and num_of_targets > 1):
                        min_diff = diff

                        self.target_r = row_index
                        self.target_c = col

        return

    # The heuristic value gives the smallest possible number of moves from the given node to the target node
    def heuristic_value(self, row, col):

        # Getting the absolute value of the (row distance difference) and the (column distance difference)
        h_val = abs(self.target_r - row) + abs(self.target_c - col)

        return h_val

    def find_path(self):

        self.find_target_location()
        # if the find_target_location left target_r and target_c at their initial 0 values, then exit
        if self.target_r == 0 and self.target_c == 0:
            print("No target found in the grid!")
        else:
            target_node = self.search_move_grid()

            # retrace the best path by going back to every parent node and store it
            while target_node.parent_node is not None:
                self.nodes_to_goal.insert(0, target_node)
                self.path_to_goal.insert(0, (target_node.row, target_node.col))
                target_node = target_node.parent_node

            # print all of the (rows, columns) of the path_to_goal
            for n in self.nodes_to_goal:
                # print("(", n.row, ",", n.col, ")")
                self.visited[n.row][n.col] = 3

            # print the grid of the best path and the searched path
            print("Final Search Grid (H = here, X = best path, O = searched path)")
            for row in self.visited:

                for col in row:
                    if col == 0:
                        print(".", end=" ")
                    elif col == 1:
                        print("O", end=" ")
                    elif col == 2:
                        print("H", end=" ")
                    elif col == 3:
                        print("X", end=" ")
                print("|")

        return self.path_to_goal

    # Search_move_grid uses A* Search to find the path of optimal length by using a priority queue (p_que)
    # to select which node gets expanded next. Each while loop adds all possible next nodes to the p_que and
    # then selects the least cost node (the head node) from the p_que and makes that the next parent_node
    # The priority queue values are equal to the total path cost to get to that node so far, plus the heuristic
    # value of that node. The node_count is the second value compared if two heuristic values are equal.
    def search_move_grid(self):

        # target -> found the goal!
        # 0 -> always move left
        # 1 -> move only left and right (cannot go to 0)
        # 6 -> move only up or down (cannot go to 4)
        # See homework description for details.

        # boolean variable to stop searching if once the target is reached
        target_found = False
        # initializing parent_node to current space with no parent
        parent_node = Node(self, None, 0, self.cur_r, self.cur_c)
        self.visited[self.cur_r][self.cur_c] = 2
        # node count to use in priority queue (p_que) for comparisons of nodes with equal f_vals
        node_count = 0

        while not target_found:

            if 0 <= self.cur_r <= 11 and 0 <= self.cur_c <= 19:

                if self.move_grid[self.cur_r][self.cur_c] in self.target_val and \
                        self.plat_grid[self.cur_r][self.cur_c].isActive:
                    # print("TARGET REACHED")
                    # print("Goal Path:")
                    target_found = True
                    return parent_node
                elif self.move_grid[self.cur_r][self.cur_c] == 0:
                    # if currently on a 0 space, always move left
                    node = Node(self, parent_node, 0, self.cur_r, self.cur_c - 1)
                    self.p_que.put((node.f_val, node_count, node))
                    node_count = node_count + 1

                elif self.move_grid[self.cur_r][self.cur_c] == 1 or (
                        self.move_grid[self.cur_r][self.cur_c] in self.target_val and not self.plat_grid[self.cur_r][
                    self.cur_c].isActive):
                    # if the space to the left is unvisited, then move left
                    if self.visited[self.cur_r][self.cur_c - 1] == 0:
                        node = Node(self, parent_node, parent_node.path_cost + 1, self.cur_r, self.cur_c - 1)
                        self.p_que.put((node.f_val, node_count, node))
                        node_count = node_count + 1

                    # if space to the right is unvisited, not equal to 0, and target is yet to be found, then move right
                    if self.visited[self.cur_r][self.cur_c + 1] == 0 and self.move_grid[self.cur_r][
                        self.cur_c + 1] != 0 and not target_found:
                        node = Node(self, parent_node, parent_node.path_cost + 1, self.cur_r, self.cur_c + 1)
                        self.p_que.put((node.f_val, node_count, node))
                        node_count = node_count + 1

                    # if space below is unvisited, equal to 6, and target is yet to be found, then move down
                    if self.visited[self.cur_r + 1][self.cur_c] == 0 and self.move_grid[self.cur_r + 1][
                        self.cur_c] == 6 and not target_found:
                        node = Node(self, parent_node, parent_node.path_cost + 1, self.cur_r + 1, self.cur_c)
                        self.p_que.put((node.f_val, node_count, node))
                        node_count = node_count + 1

                elif self.move_grid[self.cur_r][self.cur_c] == 6:

                    # if space above is unvisited and we're currently on a 6 space, then move up
                    if self.visited[self.cur_r - 1][self.cur_c] == 0:
                        node = Node(self, parent_node, parent_node.path_cost + 1, self.cur_r - 1, self.cur_c)
                        self.p_que.put((node.f_val, node_count, node))
                        node_count = node_count + 1

                    # if space below is unvisited, not equal to 4, and target is yet to be found, then move down
                    if self.visited[self.cur_r + 1][self.cur_c] == 0 and self.move_grid[self.cur_r + 1][
                        self.cur_c] != 4 and not target_found:
                        node = Node(self, parent_node, parent_node.path_cost + 1, self.cur_r + 1, self.cur_c)
                        self.p_que.put((node.f_val, node_count, node))
                        node_count = node_count + 1

                    # if space to the left is unvisited, above a 4 space, and target is yet to be found, the move left
                    if self.visited[self.cur_r][self.cur_c - 1] == 0 and self.move_grid[self.cur_r + 1][
                        self.cur_c - 1] in (4, 6) and not target_found:
                        node = Node(self, parent_node, parent_node.path_cost + 1, self.cur_r, self.cur_c - 1)
                        self.p_que.put((node.f_val, node_count, node))
                        node_count = node_count + 1

                    # if space to the right is unvisited, above a 4 space, and the target isn't found, then move right
                    if self.visited[self.cur_r][self.cur_c + 1] == 0 and self.move_grid[self.cur_r + 1][
                        self.cur_c + 1] in (4, 6) \
                            and self.move_grid[self.cur_r][self.cur_c + 1] != 0 and not target_found:
                        node = Node(self, parent_node, parent_node.path_cost + 1, self.cur_r, self.cur_c + 1)
                        self.p_que.put((node.f_val, node_count, node))
                        node_count = node_count + 1

            # if the priority queue is empty then no possible next steps were found, so return None
            if self.p_que.empty():
                print("No target found!")
                return None
            else:
                # get the least cost node from the p_que (the head node) and make that the parent_node for the next nodes
                parent_node = self.p_que.get()[2]

                # set the row and col to this new parent nodes row and col
                # so we "move" to that nodes location for the next while loop
                self.cur_r = parent_node.row
                self.cur_c = parent_node.col
                self.visited[self.cur_r][self.cur_c] = 1


# Class Node used to store parent_node, the path cost to get to this node so far, and the row/column of this node
# to back track when it's f_val is the lowest in the priority queue
class Node():
    def __init__(self, a_star_search, parent_node, path_cost, row, col):
        self.a_star_search = a_star_search
        self.parent_node = parent_node
        self.path_cost = path_cost
        self.row = row
        self.col = col
        # f_val is equal to the current path cost so far plus the heuristic value for this node
        self.f_val = path_cost + self.a_star_search.heuristic_value(self.row, self.col)
