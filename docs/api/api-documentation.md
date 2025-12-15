# API Documentation

## Overview

The Chess AI Bot API provides intelligent chess move suggestions based on historical game data analysis. All endpoints are RESTful and return JSON responses.

**Base URL**: `https://chess-ai.example.com`

**API Version**: 1.0.0

## Authentication

Currently, the API is publicly accessible. Future versions will implement JWT-based authentication.

### Future Authentication Flow

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "securepassword"
}

Response:
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

## Rate Limiting

API requests are rate-limited to prevent abuse:

- **Per IP**: 100 requests per 5 minutes
- **Per User** (authenticated): 1000 requests per hour

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640000000
```

## Endpoints

### 1. Get Chess Move Suggestion

Suggests the best chess move for a given position based on historical game analysis.

**Endpoint**: `POST /move`

**Request Headers**:
```http
Content-Type: application/json
```

**Request Body**:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| fen | string | Yes | Chess position in FEN notation |

**Response** (200 OK):
```json
{
  "move": "e2e4",
  "notation": "uci"
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| move | string | Suggested move in UCI notation |
| notation | string | Notation format (always "uci") |

**Error Responses**:

```json
// 400 Bad Request - Invalid FEN
{
  "error": "Invalid FEN position",
  "code": "INVALID_FEN"
}

// 400 Bad Request - Missing FEN
{
  "error": "FEN position required",
  "code": "MISSING_FEN"
}

// 429 Too Many Requests
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60
}

// 500 Internal Server Error
{
  "error": "Internal server error",
  "code": "INTERNAL_ERROR"
}
```

**Example Usage**:

```bash
curl -X POST https://chess-ai.example.com/move \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

```python
import requests

response = requests.post(
    'https://chess-ai.example.com/move',
    json={'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'}
)

if response.status_code == 200:
    move = response.json()['move']
    print(f"Suggested move: {move}")
```

```javascript
fetch('https://chess-ai.example.com/move', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
  })
})
.then(response => response.json())
.then(data => console.log('Suggested move:', data.move));
```

---

### 2. Health Check

Basic health check endpoint to verify service availability.

**Endpoint**: `GET /health`

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.0.0"
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| status | string | Service health status |
| timestamp | string | Current UTC timestamp (ISO 8601) |
| version | string | Application version |

**Example Usage**:

```bash
curl https://chess-ai.example.com/health
```

---

### 3. Readiness Check

Kubernetes readiness probe to check if the application is ready to serve traffic.

**Endpoint**: `GET /health/ready`

**Response** (200 OK):
```json
{
  "status": "ready",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "not ready",
  "reason": "move database not loaded"
}
```

**Use Case**: Load balancers use this endpoint to determine if a pod should receive traffic.

---

### 4. Liveness Check

Kubernetes liveness probe to check if the application is alive.

**Endpoint**: `GET /health/live`

**Response** (200 OK):
```json
{
  "status": "alive",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Use Case**: Kubernetes uses this endpoint to determine if a pod should be restarted.

---

### 5. Metrics

Prometheus-compatible metrics endpoint for monitoring.

**Endpoint**: `GET /metrics`

**Response** (200 OK):
```json
{
  "uptime_seconds": 86400,
  "memory_usage_mb": 512.5,
  "cpu_percent": 15.3,
  "move_db_positions": 150000,
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| uptime_seconds | number | Application uptime in seconds |
| memory_usage_mb | number | Memory usage in megabytes |
| cpu_percent | number | CPU utilization percentage |
| move_db_positions | number | Number of positions in move database |
| version | string | Application version |
| timestamp | string | Current UTC timestamp |

**Example Usage**:

```bash
curl https://chess-ai.example.com/metrics
```

---

### 6. Home Page

Main application interface for playing chess.

**Endpoint**: `GET /`

**Response**: HTML page with interactive chess board

**Features**:
- Color selection (White/Black)
- Interactive chessboard
- AI move suggestions
- Move history
- Undo functionality
- Reset board

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_FEN | 400 | The provided FEN position is invalid |
| MISSING_FEN | 400 | FEN position is required but not provided |
| RATE_LIMIT_EXCEEDED | 429 | API rate limit exceeded |
| INTERNAL_ERROR | 500 | Unexpected server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

## FEN Notation

FEN (Forsyth-Edwards Notation) is the standard notation for describing chess positions.

**Format**: `<piece placement> <active color> <castling> <en passant> <halfmove> <fullmove>`

**Example**:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

**Components**:
1. **Piece placement**: Board position from rank 8 to rank 1
   - Uppercase = White pieces (K, Q, R, B, N, P)
   - Lowercase = Black pieces (k, q, r, b, n, p)
   - Numbers = Empty squares
   - `/` = Rank separator

2. **Active color**: `w` (white) or `b` (black)

3. **Castling rights**: K (white kingside), Q (white queenside), k (black kingside), q (black queenside), `-` (none)

4. **En passant**: Target square in algebraic notation or `-`

5. **Halfmove clock**: Moves since last capture or pawn advance

6. **Fullmove number**: Starting at 1, incremented after black's move

**Valid Examples**:
```
# Starting position
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

# After 1.e4
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1

# After 1.e4 e5
rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2
```

## UCI Notation

UCI (Universal Chess Interface) notation represents moves using source and destination squares.

**Format**: `<from square><to square>[promotion piece]`

**Examples**:
- `e2e4`: Move from e2 to e4
- `e7e5`: Move from e7 to e5
- `e1g1`: Kingside castling (white)
- `e7e8q`: Pawn promotion to queen

**Square Notation**:
- Files: a-h (left to right)
- Ranks: 1-8 (white to black)
- Example: e2 = e-file, 2nd rank

## WebSocket API (Future)

Real-time move streaming and game updates via WebSocket connection.

**Endpoint**: `wss://chess-ai.example.com/ws`

**Connection**:
```javascript
const ws = new WebSocket('wss://chess-ai.example.com/ws');

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({
    type: 'subscribe',
    game_id: '12345'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

**Message Types**:

```json
// Subscribe to game
{
  "type": "subscribe",
  "game_id": "12345"
}

// Request move
{
  "type": "get_move",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}

// Move response
{
  "type": "move",
  "move": "e2e4",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## SDK Examples

### Python SDK

```python
import requests
from typing import Optional

class ChessAIClient:
    def __init__(self, base_url: str = "https://chess-ai.example.com"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_move(self, fen: str) -> Optional[str]:
        """Get move suggestion for a position."""
        try:
            response = self.session.post(
                f"{self.base_url}/move",
                json={"fen": fen},
                timeout=10
            )
            response.raise_for_status()
            return response.json()["move"]
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if service is healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

# Usage
client = ChessAIClient()
if client.health_check():
    move = client.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(f"Suggested move: {move}")
```

### JavaScript SDK

```javascript
class ChessAIClient {
  constructor(baseUrl = 'https://chess-ai.example.com') {
    this.baseUrl = baseUrl;
  }
  
  async getMove(fen) {
    try {
      const response = await fetch(`${this.baseUrl}/move`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.move;
    } catch (error) {
      console.error('Error:', error);
      return null;
    }
  }
  
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }
}

// Usage
const client = new ChessAIClient();
const healthy = await client.healthCheck();
if (healthy) {
  const move = await client.getMove('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  console.log('Suggested move:', move);
}
```

## Performance

### Response Times

| Endpoint | Average | P95 | P99 |
|----------|---------|-----|-----|
| POST /move | 50ms | 100ms | 200ms |
| GET /health | 5ms | 10ms | 20ms |
| GET /metrics | 10ms | 20ms | 50ms |

### Throughput

- **Sustained**: 1000 requests/second
- **Peak**: 2000 requests/second
- **Concurrent connections**: 10000

### Caching

- Move database is loaded into memory on startup
- No database queries during move suggestions
- Typical memory footprint: 500MB - 1GB

## Best Practices

### 1. Error Handling

Always implement proper error handling:

```python
try:
    response = requests.post(url, json=data, timeout=10)
    response.raise_for_status()
    return response.json()
except requests.exceptions.Timeout:
    # Handle timeout
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    # Handle HTTP errors
    if e.response.status_code == 429:
        print("Rate limited, retry after", e.response.headers.get('Retry-After'))
    else:
        print(f"HTTP error: {e}")
except requests.exceptions.RequestException as e:
    # Handle other errors
    print(f"Request failed: {e}")
```

### 2. Retry Logic

Implement exponential backoff for retries:

```python
import time
from typing import Optional

def get_move_with_retry(fen: str, max_retries: int = 3) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={"fen": fen})
            response.raise_for_status()
            return response.json()["move"]
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    return None
```

### 3. Connection Pooling

Use session objects for connection pooling:

```python
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3
)
session.mount('https://', adapter)
```

### 4. Rate Limit Handling

Respect rate limits:

```python
import time

def rate_limited_request(url, data):
    while True:
        response = requests.post(url, json=data)
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited, waiting {retry_after} seconds")
            time.sleep(retry_after)
        else:
            return response
```

## Security

### HTTPS Only

Always use HTTPS for API requests:
```python
# ✅ GOOD
requests.post('https://chess-ai.example.com/move', ...)

# ❌ BAD
requests.post('http://chess-ai.example.com/move', ...)
```

### Input Validation

Validate FEN positions before sending:
```python
import chess

def is_valid_fen(fen: str) -> bool:
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False
```

### Rate Limiting

Implement client-side rate limiting to avoid hitting server limits.

## Changelog

### Version 1.0.0 (2024-01-15)
- Initial API release
- POST /move endpoint
- Health check endpoints
- Metrics endpoint

### Upcoming Features

**Version 1.1.0** (Planned)
- JWT authentication
- User accounts and profiles
- Game history tracking
- Move analysis endpoint
- Opening book endpoint

**Version 1.2.0** (Planned)
- WebSocket support for real-time updates
- Multi-variant support (Chess960, etc.)
- Puzzle generation endpoint
- Advanced analytics

**Version 2.0.0** (Future)
- GraphQL API
- Batch move analysis
- Tournament management
- Social features

## Support

- **Documentation**: https://docs.chess-ai.example.com
- **Status Page**: https://status.chess-ai.example.com
- **Support Email**: support@chess-ai.example.com
- **GitHub Issues**: https://github.com/yourorg/chess-ai-bot/issues

## Legal

### Terms of Service

By using this API, you agree to:
- Use the API for lawful purposes only
- Not abuse or overload the service
- Not attempt to reverse engineer the algorithms
- Respect rate limits and usage quotas

### Privacy Policy

- We do not log FEN positions or moves
- We collect minimal analytics for service improvement
- We do not sell or share user data
- See full privacy policy at https://chess-ai.example.com/privacy

### License

The API is provided under the MIT License. See LICENSE file for details.
