import chess
import pandas as pd
import pickle
import os
import random
import time
import psutil
from flask import Flask, request, render_template, jsonify
from datetime import datetime

app = Flask(__name__, static_folder='chess-pieces')

# Application metadata
APP_VERSION = os.environ.get('APP_VERSION', '1.0.0')
APP_START_TIME = time.time()

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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': APP_VERSION
    }), 200

@app.route('/health/ready', methods=['GET'])
def readiness():
    try:
        # Check if move database is loaded
        if move_db is None or len(move_db) == 0:
            return jsonify({
                'status': 'not ready',
                'reason': 'move database not loaded'
            }), 503
        
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'not ready',
            'reason': str(e)
        }), 503

@app.route('/health/live', methods=['GET'])
def liveness():
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        uptime = time.time() - APP_START_TIME
        
        return jsonify({
            'uptime_seconds': uptime,
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': cpu_percent,
            'move_db_positions': len(move_db) if move_db else 0,
            'version': APP_VERSION,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
