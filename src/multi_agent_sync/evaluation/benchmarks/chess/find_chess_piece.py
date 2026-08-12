from __future__ import annotations

import chess


def get_final_piece(move_string: str, square_name: str = "b2") -> str:
    """Return the piece type on ``square_name`` after the supplied moves.

    Chess benchmark inputs end with an incomplete source-square fragment.  Tokens
    that are not legal UCI or SAN moves are therefore ignored deliberately.
    """

    board = chess.Board()
    for token in move_string.strip().split():
        if token.lower() in chess.SQUARE_NAMES:
            continue

        try:
            move = chess.Move.from_uci(token)
            if move in board.legal_moves:
                board.push(move)
                continue
        except ValueError:
            pass

        try:
            board.push_san(token)
        except ValueError:
            continue

    piece = board.piece_at(chess.parse_square(square_name))
    return chess.piece_name(piece.piece_type) if piece else "empty"
