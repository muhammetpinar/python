import chess
import chess.pgn
import pandas as pd
import chess.polyglot

player_name = input("Enter the name of the chess player: ")
year = input("Enter the year: ")


results = []
hasher = chess.polyglot.ZobristHasher([0] * 781)
pgn_file = open("cambridgeop23.pgn")
for i in pgn_file:
    game = chess.pgn.read_game(pgn_file)
    game.headers['Date']
    if player_name in game.headers['White'] or player_name in game.headers['Black']:
        if game.headers['Date'].startswith(year):
            board = game.board()
            opening = True
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number <= 6 and chess.Move.from_uci(str(move)) not in board.legal_moves:
                    legal = True
                    opening = False
                    break
            if opening:
                results.append('Opening preparation')
            else:
                results.append('Not an opening preparation')

print(results)