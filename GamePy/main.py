import arcade
import game_core
import threading
import time
import os
import pygame
import MDP as mdp
import numpy as np

class Agent(threading.Thread):

    def __init__(self, threadID, name, counter, show_grid_info=True):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.show_grid_info = show_grid_info

        self.game = []
        self.move_grid = []
        self.kill_grid = []
        self.isGameClear = False
        self.isGameOver = False
        self.current_stage = 0
        self.time_limit = 0
        self.total_score = 0
        self.total_time = 0
        self.total_life = 0
        self.tanuki_r = 0
        self.tanuki_c = 0

        # save last direction for checking if we just went up
        self.last_direction = 0

    #############################################################
    #           Mabrey                                          #
    #           Artificial Intelligence CSC-380                 #
    #           Dr. Yoon                                        #
    #           May 5th 2020                                    #
    #           Final Project                                   #
    #           AI Function to control the agent                #
    #           'Tanuki' to beat all 10 levels                  #
    #############################################################
    def ai_function(self):

        # if Tanuki is still alive, perform AI function
        if not (self.game.tanuki.isDead or self.game.tanuki.isDying):

            myMDP = mdp.MDP(self.tanuki_r, self.tanuki_c, self.move_grid, self.game.plat_grid, self.kill_grid,
                            self.game.all_enemy_sprites_list, self.game.isDisableEnemy, self.time_limit)
            myMDP.doPolicyIteration()

            curr_row = self.tanuki_r
            curr_col = self.tanuki_c

            direction = myMDP.Pi[curr_row][curr_col]

            row_change = 0
            col_change = 0

            arcade_key = None


            if direction == 0:  # check for moving up

                above_policy = myMDP.Pi[self.tanuki_r - 1][self.tanuki_c]

                # if the above policy isn't go up again (e.g. on a ladder)
                if above_policy != 0:

                    # check if we at the top space of the ladder
                    if self.move_grid[curr_row][curr_col + 1] in (2, 3, 4, 5) or self.move_grid[curr_row][curr_col - 1] in (2, 3, 4, 5) and self.move_grid[curr_row][curr_col] == 6:
                        arcade_key = arcade.key.UP
                        row_change = -1

                    # check to jump right
                    elif above_policy == 1:
                        arcade_key = arcade.key.RIGHT

                        # if we're correctly pointing right and not on a ladder then jump
                        if not self.game.tanuki.isGoingLeft and not self.game.tanuki.isGoingUpDown:
                            arcade_key = arcade.key.SPACE
                            col_change = 1

                    # check to jump left
                    elif above_policy == 3:
                        arcade_key = arcade.key.LEFT

                        # if we're correctly pointing left and not on a ladder then jump
                        if self.game.tanuki.isGoingLeft and not self.game.tanuki.isGoingUpDown:
                            arcade_key = arcade.key.SPACE
                            col_change = -1
                else:
                    arcade_key = arcade.key.UP

                    # if we're already in the going up state then we will change rows
                    if self.game.tanuki.isGoingUpDown:
                        row_change = -1

            elif direction == 2:  # check for moving down

                arcade_key = arcade.key.DOWN

                # if we're already in the going up state then we will change rows
                if self.game.tanuki.isGoingUpDown:
                    row_change = 1

            elif direction == 1:  # check for moving right

                arcade_key = arcade.key.RIGHT

                # if we're already pointing right then we're going to change columns
                if not self.game.tanuki.isGoingLeft and not self.game.tanuki.isGoingUpDown:
                    col_change = 1

            elif direction == 3:  # check for moving left

                arcade_key = arcade.key.LEFT

                # if we're already pointing left then we're going to change columns
                if self.game.tanuki.isGoingLeft and not self.game.tanuki.isGoingUpDown:
                    col_change = -1
            else:
                print("stuck")  # :))


            # check for if we're on the middle ladder space so that we don't try to go left or right when we can't
            if self.move_grid[curr_row + 1][curr_col] == 6 and self.move_grid[curr_row - 1][curr_col] == 6:

                # if on the ladder space and left or right spaces are pointing away from ladder and we didn't just go up
                if (myMDP.Pi[curr_row][curr_col + 1] == 1 or myMDP.Pi[curr_row][curr_col - 1] == 3) and self.last_direction != 0:

                    # then go down because we need to go down before we can jump
                    arcade_key = arcade.key.DOWN
                    row_change = 1
                    col_change = 0

            # if the next space is a jump gap space and we're currently facing the right direction then jump
            if myMDP.G[curr_row][curr_col + col_change] == 3 and col_change != 0:
                arcade_key = arcade.key.SPACE

            # the safety of our current agent so we know what actions to take
            next_space_safe = True
            jump_space_safe = False


            # check whether the next space is deadly currently or in the next time step and that their is ground
            # beneath it. If we're just turning around and not moving then don't check
            if not(row_change == 0 and col_change == 0):
                if self.kill_grid[curr_row + row_change][curr_col + col_change] or myMDP.next_kill_grid[curr_row + row_change][curr_col + col_change] or \
                        self.move_grid[curr_row + row_change + 1][curr_col + col_change] not in (2, 3, 4, 5, 6):
                    next_space_safe = False


            # inside this statement is where we check if we should jump, don't jump if it would be the left edge of
            # the map
            if arcade_key == arcade.key.SPACE and (curr_col + (2 * col_change) >= 0):

                # if the space we're skipping is 'not' (a reward, that is active, and not a current enemy)
                # and it's 'not' (a space with policy {north or south}, on/above a ladder, and not a current enemy)
                # -------------------------------------------
                # and if the space we're jumping to is 'not' (a current enemy or next update enemy or not above ground)
                if not (self.move_grid[curr_row][curr_col + col_change] in (8, 9, 10, 11) \
                        and self.game.plat_grid[curr_row][curr_col + col_change].isActive \
                        and not self.kill_grid[curr_row][curr_col + col_change]) \
                        and not(myMDP.Pi[curr_row][curr_col + col_change] in (0, 2) \
                        and (self.move_grid[curr_row][curr_col + col_change] == 6
                            or self.move_grid[curr_row + 1][curr_col + col_change] == 6) \
                        and not self.kill_grid[curr_row][curr_col + col_change]) \
                        and not (self.kill_grid[curr_row][curr_col + (2 * col_change)] \
                            or myMDP.next_kill_grid[curr_row][curr_col + (2 * col_change)] \
                                 or self.move_grid[curr_row + 1][curr_col + (2 * col_change)] not in (2, 3, 4, 5, 6)):
                    jump_space_safe = True

                # if the jump space isn't safe then check if we should instead just move one space left/right
                elif next_space_safe:
                    if col_change > 0:
                        arcade_key = arcade.key.RIGHT
                    else:
                        arcade_key = arcade.key.LEFT

                # else check if we should actually just try to go up a ladder
                else:
                    arcade_key = arcade.key.UP
                    next_space_safe = True



            if next_space_safe or jump_space_safe:
                self.game.on_key_press(arcade_key, 0)


            # Uncomment to print the movement data
            # print("Row change: ", row_change,  " Col change: ", col_change)
            # print("Arcade key code: ", arcade_key)
            # print("IsGoingUpDown: ", self.game.tanuki.isGoingUpDown)
            # print("Next Space safe: ", next_space_safe, "     Jump space safe: ", jump_space_safe ,"\n\n\n")

            self.last_direction = direction



    def run(self):
        print("Starting " + self.name)

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50+320, 50)
        if self.show_grid_info:
            pygame.init()
        else:
            pygame = []

        # Prepare grid information display (can be turned off if performance issue exists)
        if self.show_grid_info:
            screen_size = [200, 120]
            backscreen_size = [40, 12]

            screen = pygame.display.set_mode(screen_size)
            backscreen = pygame.Surface(backscreen_size)
            arr = pygame.PixelArray(backscreen)
        else:
            time.sleep(0.5)  # wait briefly so that main game can get ready

        # roughly every 50 milliseconds, retrieve game state (average human response time for visual stimuli = 25 ms)
        go = True
        while go and (self.game is not []):
            # Dispatch events from pygame window event queue
            if self.show_grid_info:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        go = False
                        break

            # RETRIEVE CURRENT GAME STATE
            self.move_grid, self.kill_grid, \
                self.isGameClear, self.isGameOver, self.current_stage, self.time_limit, \
                self.total_score, self.total_time, self.total_life, self.tanuki_r, self.tanuki_c \
                = self.game.get_game_state()

            self.ai_function()

            # Display grid information (can be turned off if performance issue exists)
            if self.show_grid_info:
                for row in range(12):
                    for col in range(20):
                        c = self.move_grid[row][col] * 255 / 12
                        arr[col, row] = (c, c, c)
                    for col in range(20, 40):
                        if self.kill_grid[row][col-20]:
                            arr[col, row] = (255, 0, 0)
                        else:
                            arr[col, row] = (255, 255, 255)

                pygame.transform.scale(backscreen, screen_size, screen)
                pygame.display.flip()

            # We must allow enough CPU time for the main game application
            # Polling interval can be reduced if you don't display the grid information
            time.sleep(0.05)

        print("Exiting " + self.name)


def main():
    ag = Agent(1, "My Agent", 1, False)

    ag.start()

    ag.game = game_core.GameMain()
    ag.game.set_location(50, 50)

    # Uncomment below for recording
    #ag.game.isRecording = True
    #ag.game.replay('replay.rpy')  # You can specify replay file name or it will be generated using timestamp

    # Uncomment below to replay recorded play
    #ag.game.isReplaying = True
    #ag.game.replay('replay_clear.rpy')

    ag.game.reset()
    arcade.run()
    print("Total life", ag.total_life)
if __name__ == "__main__":
    main()