import torch

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Board:
    def __init__(self) -> None:
        self.current_player = 1
        self.board = torch.zeros(9).to(device)
        self.game_history = []

    def reset_board(self):
        self.current_player = 1
        self.board = torch.zeros(9).to(device)
        self.game_history = []

    def get_game_state(self) -> tuple:
        curr_player = torch.Tensor([self.current_player]).to(device)
        board = self.board

        legality_matrix = torch.zeros_like(self.board).to(device)
        for i, e in enumerate(self.board):
            if torch.round(torch.abs(e)).item() == 0:
                legality_matrix[i] = 1
            else:
                legality_matrix[i] = 0

        return torch.cat([curr_player, board]), legality_matrix

    def check_if_game_over(self) -> tuple:
        game_over = False

        vis_board = self.board.reshape(3, 3)

        row_sums = torch.round(torch.sum(vis_board, dim=1))
        col_sums = torch.round(torch.sum(vis_board, dim=0))
        diag1_sum = torch.round(torch.sum(torch.diag(vis_board)))
        diag2_sum = torch.round(torch.sum(torch.diag(torch.flip(vis_board, dims=[1]))))

        all_sums = torch.cat((row_sums, col_sums, diag1_sum.unsqueeze(0), diag2_sum.unsqueeze(0)), dim=0)

        for i, element in enumerate(all_sums):
            if abs(element.item()) == 3:
                game_over = True
                return True, element.item() / 3
        if not game_over:
            draw = True
            for i, element in enumerate(self.board):
                if torch.round(element).item() == 0:
                    draw = False
            if draw:
                return True, 0
            else:
                return False, 0

    def player_make_turn(self, where_player_went: int) -> None:
        g_state, l_matrix = self.get_game_state()
        self.game_history.append((g_state, where_player_went, l_matrix))
        self.board[where_player_went] = self.current_player
        self.current_player = -self.current_player

    def set_winner(self, winner: int) -> tuple:
        good_moves, bad_moves, neutral_moves = [], [], []
        if winner == 0:
            neutral_moves = self.game_history.copy()
        else:
            for i, (g_state, p_choice, l_matrix) in enumerate(self.game_history):
                if winner == g_state[0]:
                    good_moves.append((g_state, p_choice, l_matrix))
                else:
                    bad_moves.append((g_state, p_choice, l_matrix))

        return good_moves, bad_moves, neutral_moves


TTT = Board()

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

        self.layers = nn.Sequential(
            nn.Linear(in_features=10, out_features=32),
            self.tanh,
            nn.Linear(in_features=32, out_features=32),
            self.tanh,
            nn.Linear(in_features=32, out_features=32),
            self.tanh,
            nn.Linear(in_features=32, out_features=9),
        )

    def forward(self, x: torch.Tensor, leg_matrix: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = (self.tanh(x) + 1) / 2
        x = leg_matrix * x
        return x


model = Model().to(device)


def play_machine_only():
    TTT.reset_board()
    gameover = False
    while not gameover:
        g_state, l_matrix = TTT.get_game_state()
        raw_logits = model(g_state, l_matrix)
        softmax_logits = model.softmax(raw_logits)
        choice = torch.argmax(softmax_logits).item()
        TTT.player_make_turn(choice)
        is_over, player = TTT.check_if_game_over()
        if is_over:
            g_moves, b_moves, n_moves = TTT.set_winner(player)
            # print(TTT.board.reshape(3, 3))
            # gameover = True
            return g_moves, b_moves, n_moves


def play_with_player(playerturn: bool = True):
    TTT.reset_board()
    gameover = False
    print(TTT.board.reshape(3, 3), end="\n\n")
    while not gameover:
        if playerturn:
            player_index = int(input("Where to go? "))
            TTT.player_make_turn(player_index)
            print(TTT.board.reshape(3, 3), end="\n\n")
            is_over, player = TTT.check_if_game_over()
            if is_over:
                # g_moves, b_moves, n_moves = TTT.set_winner(player)
                print("Player won")
                gameover = True
                # return g_moves, b_moves, n_moves
            playerturn = False
        else:
            g_state, l_matrix = TTT.get_game_state()
            raw_logits = model(g_state, l_matrix)
            softmax_logits = model.softmax(raw_logits)
            choice = torch.argmax(softmax_logits).item()
            TTT.player_make_turn(choice)
            print(TTT.board.reshape(3, 3), end="\n\n")
            is_over, player = TTT.check_if_game_over()
            if is_over:
                # g_moves, b_moves, n_moves = TTT.set_winner(player)
                print("Machine won")
                gameover = True
                # return g_moves, b_moves, n_moves
            playerturn = True


import torch

EPOCHS = 2000
LEARNING_RATE = 0.01

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"E {epoch + 1:,} - {((epoch + 1) / EPOCHS) * 100:.2f}%")
    model.eval()

    with torch.no_grad():
        g, b, n = play_machine_only()
        good, bad, neutral = [], [], []
        for index, (g_state, p_choice, l_matrix) in enumerate(g):
            onehot = torch.zeros(9).to(device)
            onehot[p_choice] = 1.0
            good.append((g_state, onehot, l_matrix))
        for index, (g_state, p_choice, l_matrix) in enumerate(b):
            onehot = torch.ones(9).to(device)
            onehot[p_choice] = 0.0
            bad.append((g_state, onehot, l_matrix))
        for index, (g_state, p_choice, l_matrix) in enumerate(n):
            onehot = torch.full(size=(9,), fill_value=0.5).to(device)
            onehot[p_choice] = 0.0
            neutral.append((g_state, onehot, l_matrix))
        game_positions = good + bad + neutral

    model.train()
    optimizer.zero_grad()

    for index, (g, p, l) in enumerate(game_positions):
        outputs = model(g, l)
        loss = loss_fn(outputs, p)
        loss.backward()

    optimizer.step()

play_with_player()
