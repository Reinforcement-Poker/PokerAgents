from treys import Card, Evaluator


def get_hand_score(hand: list[str], board: list[str]) -> float:
    if len(board) > 0:
        N_HAND_RANKS = 7462
        evaluator = Evaluator()
        to_card = lambda card: Card.new(f"{card[1]}{card[0].lower()}")
        eval_hand = list(map(to_card, hand))
        eval_board = list(map(to_card, board))
        return 1 - evaluator.evaluate(eval_board, eval_hand) / N_HAND_RANKS

    card_number = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

    max_score = 318
    card_number = sum(card_number.index(card[1]) ** 2.5 for card in hand)
    is_suited = hand[0][0] == hand[1][0]
    is_pair = hand[0][1] == hand[1][1]

    if is_pair or is_suited:
        return 1 - card_number / max_score

    return 1 - card_number * 1.2 / max_score
