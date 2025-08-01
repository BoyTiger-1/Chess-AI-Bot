import chess
import pandas as pd
import pickle
import os
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Dataset processing
def build_move_database():
    if os.path.exists('move_db.pkl'):
        with open('move_db.pkl', 'rb') as f:
            return pickle.load(f)
    
    df = pd.read_csv('games.csv.gz')
    move_db = {}
    
    for _, row in df.iterrows():
        try:
            moves = row['moves'].split()
            board = chess.Board()
            result = 1 if row['winner'] == 'white' else 0 if row['winner'] == 'black' else 0.5
            
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if not board.is_legal(move):
                    continue
                    
                fen = board.fen()
                if fen not in move_db:
                    move_db[fen] = {}
                    
                if move_uci not in move_db[fen]:
                    move_db[fen][move_uci] = {'wins': 0, 'total': 0}
                
                # Update from perspective of player making the move
                move_db[fen][move_uci]['wins'] += result if board.turn == chess.WHITE else 1 - result
                move_db[fen][move_uci]['total'] += 1
                board.push(move)
        except:
            continue
    
    with open('move_db.pkl', 'wb') as f:
        pickle.dump(move_db, f)
    return move_db

move_db = build_move_database()

def get_ai_move(fen):
    board = chess.Board(fen)
    
    # Get moves from database
    if fen in move_db:
        best_score = -1
        best_moves = []
        
        for move_uci, stats in move_db[fen].items():
            if stats['total'] > 0:
                score = stats['wins'] / stats['total']
                if score > best_score:
                    best_score = score
                    best_moves = [move_uci]
                elif score == best_score:
                    best_moves.append(move_uci)
        
        if best_moves:
            return random.choice(best_moves)
    
    # Fallback to random move
    return random.choice([m.uci() for m in board.legal_moves])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    move = get_ai_move(data['fen'])
    return jsonify({'move': move})

if __name__ == '__main__':
    app.run(debug=True)
