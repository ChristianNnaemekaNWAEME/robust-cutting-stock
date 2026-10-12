"""
cutting_stock.py

A real (if simplified) 1D cutting-stock solver using the First-Fit-Decreasing
(FFD) heuristic, a standard, widely-used approach to the cutting-stock /
bin-packing problem. Given a stock roll length and a list of required piece
lengths, it assigns pieces to stock rolls to minimize the number of rolls
used (and therefore material waste), subject to a kerf (blade width) loss
per cut.

This module is deliberately simple and transparent rather than a full
Gilmore-Gomory column-generation solver, so that the effect of upstream
prediction error (robust vs. non-robust demand forecasting) on downstream
material waste can be measured directly and clearly.
"""

import numpy as np


def first_fit_decreasing(piece_lengths, stock_length, kerf=0.5):
    """
    Assign pieces to stock rolls using First-Fit-Decreasing.

    Returns:
        rolls: list of lists, each inner list the piece lengths on one roll
        waste_per_roll: list of leftover (waste) length per roll
    """
    pieces = sorted(piece_lengths, reverse=True)
    rolls = []          # list of piece-lists
    remaining = []       # remaining usable length per roll

    for piece in pieces:
        if piece > stock_length:
            # Piece doesn't fit on a single stock roll at all; still place
            # it alone (it will register as negative-waste / an infeasible
            # order in reporting) rather than silently dropping it.
            rolls.append([piece])
            remaining.append(stock_length - piece)
            continue

        placed = False
        for i, cap in enumerate(remaining):
            needed = piece + (kerf if rolls[i] else 0.0)
            if needed <= cap:
                rolls[i].append(piece)
                remaining[i] -= needed
                placed = True
                break
        if not placed:
            rolls.append([piece])
            remaining.append(stock_length - piece)

    waste_per_roll = remaining
    return rolls, waste_per_roll


def plan_summary(piece_lengths, stock_length, kerf=0.5):
    rolls, waste = first_fit_decreasing(piece_lengths, stock_length, kerf=kerf)
    total_stock_used = len(rolls) * stock_length
    total_piece_length = float(np.sum(piece_lengths))
    total_waste = float(np.sum(waste))
    utilization = total_piece_length / total_stock_used if total_stock_used else 0.0
    return {
        "n_rolls": len(rolls),
        "total_stock_used": total_stock_used,
        "total_piece_length": total_piece_length,
        "total_waste": total_waste,
        "utilization": utilization,
    }
