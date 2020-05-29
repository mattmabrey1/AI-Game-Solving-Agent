import numpy as np
import copy

# Matthew Mabrey
# Artificial Intelligence CSC-380
# Dr. Yoon
# May 5th 2020
# Final Project
# Policy Iteration class which is used by AI function in main to determine a policy and action for each update
# given the current fruit targets, enemy locations, move grid, and time limit

class MDP:

    def __init__(self, init_row, init_col, move_grid, plat_grid, kill_grid, all_enemy_sprites_list, is_enemy_disable,
                 time_limit):

        np.set_printoptions(linewidth=350, precision=0)

        # Over the first 10 seconds increase reward decay up to 0.9 maximum
        if time_limit > 90:
            reward_decay_time = time_limit - 90
        else:
            reward_decay_time = 0

        # Upper bound for iterations
        self.T = 100

        # Modified value iterations to perform before policy iteration
        self.K = 6

        # MDP conditions
        self.living_expense = -3
        self.reward_decay = 0.9 - (reward_decay_time * 0.03)
        self.fruit_reward = 250 + ((99 - time_limit) * 30)
        self.bonus_reward = 500
        self.enemy_reward = -500
        self.spike_reward = -100

        # convergence threshold, smaller values slow draw time but improve accuracy
        self.epsilon = 0.01

        # Convergence boolean to check every policy improvement
        self.unchanged = False

        # Rows of this map
        self.rows = len(move_grid)
        # Columns of this map
        self.cols = len(move_grid[0])

        # Current Row for Tanuki
        self.init_row = init_row
        # Current Column for Tanuki
        self.init_col = init_col

        # kill grid for the next update
        self.next_kill_grid = np.copy(kill_grid)

        # calculate where all enemies will be for the next update
        if not is_enemy_disable:

            for enemy in all_enemy_sprites_list:

                if enemy.isActive:
                    enemy_row, enemy_col = enemy.get_gridRC()
                    enemy_going_left = enemy.isGoingLeft

                    # if enemy is going left and is not a col = 0 (e.g. about to turn around) and not about to collide
                    # with Tanuki (since we want the jump space over the enemy if it's directly in front of Tanuki)
                    if enemy_going_left and enemy_col > 0 and not (enemy_row == init_row and (enemy_col - 1) == init_col):

                        self.next_kill_grid[enemy_row][enemy_col - 1] = True
                        if move_grid[enemy_row][enemy_col] != 7:
                            self.next_kill_grid[enemy_row][enemy_col] = False

                    # if enemy is going right and not at col = 17 (e.g. about to turn around) and not about to collide
                    # with Tanuki (since we want the jump space over the enemy if it's directly in front of Tanuki)
                    elif not enemy_going_left and enemy_col < 18 and not (enemy_row == init_row and (enemy_col + 1) == init_col):

                        self.next_kill_grid[enemy_row][enemy_col + 1] = True
                        if move_grid[enemy_row][enemy_col] != 7:
                            self.next_kill_grid[enemy_row][enemy_col] = False

                    # else just keep it in the same spot, e.g. spikes and enemies turning around
                    else:
                        self.next_kill_grid[enemy_row][enemy_col] = True


        # MDP Movability Grid, 0 = Unmovable space, 1 = Movable space, 2 = Terminal space, 3 = Gap space
        self.G = np.zeros_like(move_grid)

        for row in range(self.rows):
            for col in range(self.cols):

                # if this space is a kill space (enemy/spike) make it a terminal space
                if self.next_kill_grid[row][col]:

                    self.G[row][col] = 2

                    # if the enemy or spike is not on the edge of the map
                    if 0 < col < 19:

                        # get distance between Tanuki and enemy
                        enemy_dist = abs(col - init_col) + abs(row - init_row)

                        # if the enemy is far enough away or it's not adjacent to another enemy or spike, make
                        # the spaces directly up, up left, and up right movable
                        if (not (self.next_kill_grid[row][col - 1] or self.next_kill_grid[row][col + 1])) or enemy_dist > 3:

                            # if the movable spaces are above ground (so that Tanuki doesn't jump over enemy to death)
                            if (move_grid[row + 1][col + 1] in (2, 3, 4, 5, 6) and move_grid[row + 1][col - 1] in (
                            2, 3, 4, 5, 6)):
                                self.G[row - 1][col] = 1
                                self.G[row - 1][col + 1] = 1
                                self.G[row - 1][col - 1] = 1

                # if this space is a target and it's still active, then make it a terminal space
                elif move_grid[row][col] in (8, 9, 10, 11) and plat_grid[row][col].isActive:
                    self.G[row][col] = 2

                # if this space is a ladder make it movable
                elif move_grid[row][col] == 6:
                    self.G[row][col] = 1

                # if this space is not a platform and not in the rightmost unmovable column
                elif move_grid[row][col] not in (2, 3, 4, 5, 6) and col < 19:

                    # if this space is above a platform make it movable
                    if move_grid[row + 1][col] in (2, 3, 4, 5, 6):
                        self.G[row][col] = 1

                    # if it's not above a platform, see if it's a space between a movable gap, assign it 3 for gap space
                    elif move_grid[row + 1][col + 1] in (2, 3, 4, 5, 6) and move_grid[row + 1][col - 1] in (2, 3, 4, 5, 6) \
                      and not (self.next_kill_grid[row][col - 1] or self.next_kill_grid[row][col + 1]):
                        self.G[row][col] = 3



        self.G[10][19] = 1  # make sure the first spot is movable even though its on the far right side of the map

        # Reward grid, rewards for every type of space at the top of this class def
        self.R = np.copy(move_grid)

        for row in range(self.rows):
            for col in range(self.cols):

                if self.next_kill_grid[row][col]:

                    # if this space is spike assign it the spike reward value
                    if move_grid[row][col] == 7:
                        self.R[row][col] = self.spike_reward
                    # else, it must be an enemy, so assign it the enemy reward value
                    else:
                        self.R[row][col] = self.enemy_reward

                # if this space is a regular fruit and it's active, assign it a fruit reward value
                elif move_grid[row][col] == 8 and plat_grid[row][col].isActive:

                    self.R[row][col] = self.fruit_reward

                    # calculate how many rows away the player is from this reward to see if enemies should decrease
                    # it's rewards because there's no need to calculate
                    reward_row_dist = abs(row - init_row)
                    reward_total_dist = abs(col - init_col) + reward_row_dist

                    # if this reward is several rows up or down then don't calculate a decrease for it since the
                    # enemy will be far away by the time Tanuki actually reaches that reward
                    if reward_row_dist < 5 or reward_total_dist < 12:
                        for enemy in all_enemy_sprites_list:

                            if enemy.isActive:
                                enemy_row, enemy_col = enemy.get_gridRC()
                                enemy_going_left = enemy.isGoingLeft


                                # reduce reward based on how many enemies are moving towards this target and how
                                # far away they are from it
                                if enemy_row == row and ((enemy_col > col and enemy_going_left) or \
                                                         (enemy_col < col and not enemy_going_left)):
                                    enemy_dist = abs(col - enemy_col)
                                    self.R[row][col] = self.R[row][col] * min((enemy_dist / 10), 1)

                # if this space is one of the bonus targets and it's active, assign it the bonus target reward
                elif move_grid[row][col] in (9, 10, 11) and plat_grid[row][col].isActive:

                    self.R[row][col] = self.bonus_reward

                    # calculate how many rows and cols away the player is from this reward to see if enemies should
                    # decrease it's rewards
                    reward_row_dist = abs(row - init_row)
                    reward_total_dist = abs(col - init_col) + reward_row_dist

                    # if this reward is far away then don't calculate a decrease for it since the
                    # enemy will be far away by the time Tanuki actually reaches that reward
                    if reward_row_dist < 5 or reward_total_dist < 12:
                        for enemy in all_enemy_sprites_list:

                            if enemy.isActive:
                                enemy_row, enemy_col = enemy.get_gridRC()
                                enemy_going_left = enemy.isGoingLeft

                                # bonus rewards are more dangerous when enemies are nearby since a secret enemy can
                                # spawn, so reduce value more if an enemy is close to it
                                if enemy_row == row and ((enemy_col > col and enemy_going_left) or \
                                                         (enemy_col < col and not enemy_going_left)):
                                    enemy_dist = abs(col - enemy_col)
                                    self.R[row][col] = self.R[row][col] * min((enemy_dist / 15), 1)

                # if this space is just a blank space, give it the negative living expense as it's reward
                else:
                    self.R[row][col] = self.living_expense

        # V grid (= max(Q))
        self.V = np.zeros_like(move_grid, dtype=float)

        # I left this array hardcoded instead of "self.Q = np.zeros((self.rows, self.cols, 4))" because hardcoding it
        # seemed to speed up performance
        # Q-value grid (= sum of (reward decay) * V(s'), (s, a, s') with transition probability equal to 1)
        self.Q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]


        # Policy array, 0 = North, 1 = East, 2 = South, 3 = West
        self.Pi = np.zeros_like(move_grid)


    def show_G(self):
        print("G = ")
        for i in range(self.rows):
            print(self.G[i])

    def show_R(self):
        print("R = ")
        for i in range(self.rows):
            print(self.R[i])

    def show_V(self):
        print("V = ")
        for i in range(self.rows):
            print(self.V[i])

    def show_Q(self):
        print("Q = ")
        for i in range(self.rows):
            print(self.Q[i])

    def show_Pi(self):
        print("Pi = ")
        Pi = np.chararray(shape=(len(self.Pi), len(self.Pi[0])), )
        for row in range(self.rows):
            for col in range(self.cols):
                if self.Pi[row][col] == 0:
                    Pi[row][col] = '^'
                elif self.Pi[row][col] == 1:
                    Pi[row][col] = '>'
                elif self.Pi[row][col] == 2:
                    Pi[row][col] = 'v'
                elif self.Pi[row][col] == 3:
                    Pi[row][col] = '<'
                else:
                    Pi[row][col] = '.'
            print(Pi[row])

    def calc_V(self):
        # simply pick the best Q
        for r in range(self.rows):
            for c in range(self.cols):
                self.V[r][c] = max(self.Q[r][c])



    def calc_Q(self):
        for r in range(self.rows):
            for c in range(self.cols):

                if self.G[r][c] == 0:
                    # If this cell is not movable, then no need to compute Q (and V)
                    self.Q[r][c] = [0, 0, 0, 0]
                    continue
                elif self.G[r][c] == 2:
                    # If this cell is not movable, then just copy reward to Q (and V)
                    self.Q[r][c][0] = self.R[r][c]
                    self.Q[r][c][1] = self.R[r][c]
                    self.Q[r][c][2] = self.R[r][c]
                    self.Q[r][c][3] = self.R[r][c]
                    continue
                else:
                    # --------------------------------------------------------------------------------------------------
                    # 0 = NORTH
                    self.Q[r][c][0] = 0.0
                    # Movable this direction?
                    if r > 0 and self.G[r-1][c] != 0:
                        self.Q[r][c][0] += (self.R[r][c] + self.reward_decay * self.V[r-1][c])
                    # else this direction isn't movable, so just take the current space's value
                    else:
                        self.Q[r][c][0] += (self.R[r][c] + self.reward_decay * self.V[r][c])

                    # --------------------------------------------------------------------------------------------------
                    # 1 = EAST

                    self.Q[r][c][1] = 0.0
                    # Movable this direction? (ok_prob)
                    if c < self.cols-1 and self.G[r][c+1] != 0:
                        self.Q[r][c][1] += (self.R[r][c] + self.reward_decay * self.V[r][c+1])
                    # else this direction isn't movable, so just take the current space's value
                    else:
                        self.Q[r][c][1] += (self.R[r][c] + self.reward_decay * self.V[r][c])

                    # --------------------------------------------------------------------------------------------------
                    # 2 = SOUTH

                    self.Q[r][c][2] = 0.0
                    # Movable this direction? (ok_prob)
                    if r < self.rows-1 and self.G[r+1][c] != 0:
                        self.Q[r][c][2] += (self.R[r][c] + self.reward_decay * self.V[r+1][c])
                    # else this direction isn't movable, so just take the current space's value
                    else:
                        self.Q[r][c][2] += (self.R[r][c] + self.reward_decay * self.V[r][c])

                    # --------------------------------------------------------------------------------------------------
                    # 3 = WEST

                    self.Q[r][c][3] = 0.0
                    # Movable this direction? (ok_prob)
                    if c > 0 and self.G[r][c-1] != 0:
                        self.Q[r][c][3] += (self.R[r][c] + self.reward_decay * self.V[r][c-1])
                    # else this direction isn't movable, so just take the current space's value
                    else:
                        self.Q[r][c][3] += (self.R[r][c] + self.reward_decay * self.V[r][c])


    def calc_PolicyEvaluation(self):

        # calculate the utility (value) for each state given current policy
        for r in range(self.rows):
            for c in range(self.cols):
                policy = self.Pi[r][c]
                self.V[r][c] = self.Q[r][c][policy]


    def calc_PolicyImprovement(self):

        # calculate the current Q values for all states
        self.calc_Q()

        self.unchanged = True

        # if the Q value max of any state is higher than the current state value from the policy action,
        # make that Q value max direction the new policy of that state and set unchanged to False so
        # we know that the values haven't converged
        for row in range(self.rows):
            for col in range(self.cols):

                if self.G[row][col] in (1, 3):
                    new_policy = 0
                    q_max = self.Q[row][col][0]


                    for i in range(0, 4):

                        if self.Q[row][col][i] > q_max:
                            q_max = self.Q[row][col][i]
                            new_policy = i

                    value_difference = abs(q_max - self.V[row][col])

                    # if values haven't converged, choose the new best policy
                    if value_difference > self.epsilon:
                        self.Pi[row][col] = new_policy
                        self.unchanged = False




    def doPolicyIteration(self):

        # modified policy iteration that does a K simplified value iterations before starting policy iteration
        for k in range(self.K + 1):
            self.calc_Q()
            self.calc_V()

        # begin policy iteration for T iterations
        for t in range(self.T + 1):

            self.calc_PolicyEvaluation()
            self.calc_PolicyImprovement()

            # if values converged, then break out of FOR loop early
            if self.unchanged:
                break


        # Uncomment to print the grid data
        # self.show_Q()
        # self.show_V()
        # self.G[self.init_row][self.init_col] = 5
        # print("MDP grid movability \n")
        # self.show_G()
        # self.show_Pi()
        # print("\n Reward Array\n", self.R)
