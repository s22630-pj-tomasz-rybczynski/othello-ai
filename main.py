from easyAI import TwoPlayerGame, AI_Player, Negamax, Human_Player

class Board:
    """
    A class representing the game board.
    """
    def __init__(self):
        self.grid = [[' ' for _ in range(8)] for _ in range(8)]
        self.grid[3][3] = 'W'
        self.grid[3][4] = 'B'
        self.grid[4][3] = 'B'
        self.grid[4][4] = 'W'

    def display(self):
        """
        Display the current state of the board.
        """
        print('  0 1 2 3 4 5 6 7')
        print(' +-+-+-+-+-+-+-+-+')
        for i in range(8):
            row = str(i) + '|'
            for j in range(8):
                row += self.grid[i][j] + '|'
            print(row)
            print(' +-+-+-+-+-+-+-+-+')


class Player:
    """
    A class representing the player in the game - either human or AI.
    """
    def __init__(self, color, is_human=False):
        self.color = color
        self.is_human = is_human

    def get_move(self, game):
        if self.is_human:
            return Human_Player().ask_move(game)
        else:
            ai = AI_Player(Negamax(6))
            move = ai.ask_move(game)
            return (move[0], move[1])
        
    def ask_move(self, game):
        return self.get_move(game)


class Game(TwoPlayerGame):
    """
    A class representing the game itself.
    """
    def __init__(self, players):
        """
        Initialize the game.

        Args:
            players (list): A list of two Player objects.
        """
        self.players = players
        self.board = Board()
        self.current_player = 2

    def possible_moves(self):
        """
        Get a list of all possible moves for the current player.

        Returns:
            list: A list of (x, y) tuples representing possible moves.
        """
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board.grid[i][j] == ' ':
                    if self.is_valid_move(i, j):
                        moves.append((i, j))
        return moves

    def make_move(self, move):
        """
        Make a move on the board.

        Args:
            move (tuple): An (x, y) tuple representing the move to make.
        """
        x, y = move
        self.board.grid[x][y] = self.get_current_player().color
        self.flip_pieces(x, y)

    def flip_pieces(self, x, y):
        """
        Flip pieces on the board around the piece after a move is made.

        Args:
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.
        """
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            self.flip_pieces_in_direction(x, y, dx, dy)

    def flip_pieces_in_direction(self, x, y, dx, dy):
        """
        Flip pieces in a specific direction after a move is made.

        Args:
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.
            dx (int): The x-direction to flip pieces in.
            dy (int): The y-direction to flip pieces in.
        """
        other_color = 'B' if self.get_current_player().color == 'W' else 'W'
        pieces_to_flip = []
        while True:
            x += dx
            y += dy
            if x < 0 or x >= 8 or y < 0 or y >= 8:
                break
            if self.board.grid[x][y] == ' ':
                break
            if self.board.grid[x][y] == self.get_current_player().color:
                for i, j in pieces_to_flip:
                    self.board.grid[i][j] = self.get_current_player().color
                break
            if self.board.grid[x][y] == other_color:
                pieces_to_flip.append((x, y))

    def is_valid_move(self, x, y):
        """
        Check if a move is valid.

        Args:
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        if self.board.grid[x][y] != ' ':
            return False
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            if self.is_valid_direction(x, y, dx, dy):
                return True
        return False

    def is_valid_direction(self, x, y, dx, dy):
        """
        Check if a move is valid in a specific direction.

        Args:
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.
            dx (int): The x-direction to check.
            dy (int): The y-direction to check.

        Returns:
            bool: True if the move is valid in the given direction, False otherwise.
        """
        other_color = 'B' if self.get_current_player().color == 'W' else 'W'
        found_other_color = False
        while True:
            x += dx
            y += dy
            if x < 0 or x >= 8 or y < 0 or y >= 8:
                return False
            if self.board.grid[x][y] == ' ':
                return False
            if self.board.grid[x][y] == other_color:
                found_other_color = True
            if self.board.grid[x][y] == self.get_current_player().color:
                return found_other_color
            
    def show(self):
        """
        Display the current state of the game.
        """
        self.board.display()
            
    def is_over(self):
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(i, j):
                    return False

        return True
    
    def scoring(self):
        """
        Calculate the scoring for AI.

        Returns:
            int: Score.
        """
        # Initialize scores
        scores = {self.players[0].color: 0, self.players[1].color: 0}

        # Calculate piece count and mobility
        for i in range(8):
            for j in range(8):
                if self.board.grid[i][j] != ' ':
                    scores[self.board.grid[i][j]] += 1
                else:
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        if self.is_valid_direction(i, j, dx, dy):
                            scores[self.get_current_player().color] += 1

        # Calculate final score
        return scores[self.players[0].color] - scores[self.players[1].color]
    
    def get_current_player(self):
        """
        Get the current player.

        Returns:
            Player: The current player.
        """
        return self.players[self.current_player - 1]

if __name__ == '__main__':
    game = Game([Player('W'), Player('B', True)])
    game.play()

    points = {game.players[0].color: 0, game.players[1].color: 0}
    for i in range(8):
        for j in range(8):
            if game.board.grid[i][j] != ' ':
                points[game.board.grid[i][j]] += 1
    print('Game over! Final score: W:%d - B:%d' % (points['W'], points['B']))
