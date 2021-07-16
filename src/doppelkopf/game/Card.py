import numpy as np

class Card():
    suits = ["Clubs (\u2667)", "Spades (\u2664)", "Hearts (\u2661)", "Diamonds (\u2662)"]
    NUM_CARDTYPES = 24
    QUEEN_OF_CLUBS_TYPE = 1
    DUMMY_FLAT = np.zeros(NUM_CARDTYPES, dtype=np.float32)

    # suit = Clubs, Spades, Hearts or Diamonds
    # number = The number (or letter) shown on the card (9-10, J, B, Q, K, A)
    # rank = The rank within the hierarchy (who beats who)
    # value = How much this card is worth when counting points
    def __init__(self, cardType: int, suit: str, number: str, rank: int, value: int, isTrump: bool):
        self.cardType = cardType
        self.suit = suit
        self.number = number
        self.numberName = Card.NumberToName(self.number)
        self.rank = rank
        self.value = value
        self.isTrump = isTrump
        # Produce one-hot encoding of relevant card information (suit, number, value and has_been_played)
        self.flat = self.flatten()

    @staticmethod
    def createDeck() -> np.ndarray:
        cardDeck = []
        for card in Card.CARDTYPES:
            cardDeck.append(card.clone())
            cardDeck.append(card.clone())
        return np.array(cardDeck)

    @staticmethod
    def FromFlat(flat: np.ndarray):
        if flat.sum() == 0:
            return None
        cardType = flat.argmax(axis=0)
        return Card.CARDTYPES[cardType].clone() # Create a clone of the requested card type

    def flatten(self) -> np.ndarray:
        one_hot = np.zeros(Card.NUM_CARDTYPES, dtype=np.float32)
        one_hot[self.cardType] = 1
        return one_hot

    def Flat(self):
        return self.flat

    nameDict = {
        11: "Ace",
        12: "Jack",
        13: "Queen",
        14: "King"
    }
    @staticmethod
    def NumberToName(number):
        return Card.nameDict.get(number, str(number))

    @staticmethod
    def AreIdentcial(a, b) -> bool:
        return a.cardType == b.cardType and a.suit == b.suit and a.number == b.number and a.rank == b.rank and a.value == b.value and a.isTrump == b.isTrump

    def clone(self):
        return Card(self.cardType, self.suit, self.number, self.rank, self.value, self.isTrump)

    def __str__(self):
        return "%s of %s" % (self.numberName, self.suit)

    def __repr__(self):
        return self.__str__()

Card.CARDTYPES = [
    Card(0, Card.suits[2], 10, 1, 10, True), #"10 of Hearts"
    Card(1, Card.suits[0], 13, 2, 3, True), #"Queen of Clubs"
    Card(2, Card.suits[1], 13, 3, 3, True), #"Queen of Spades"
    Card(3, Card.suits[2], 13, 4, 3, True), #"Queen of Hearts"
    Card(4, Card.suits[3], 13, 5, 3, True), #"Queen of Diamonds"
    Card(5, Card.suits[0], 12, 6, 2, True), #"Jack of Clubs"
    Card(6, Card.suits[1], 12, 7, 2, True), #"Jack of Spades"
    Card(7, Card.suits[2], 12, 8, 2, True), #"Jack of Hearts"
    Card(8, Card.suits[3], 12, 9, 2, True), #"Jack of Diamonds"
    Card(9, Card.suits[3], 11, 10, 11, True), #"Ace of Diamonds"
    Card(10, Card.suits[3], 10, 11, 10, True), #"10 of Diamonds"
    Card(11, Card.suits[3], 14, 12, 4, True), #"King of Diamonds"
    Card(12, Card.suits[3], 9, 13, 0, True), #"9 of Diamonds"
    Card(13, Card.suits[0], 11, 1, 11, False), #"Ace of Clubs"
    Card(14, Card.suits[0], 10, 2, 10, False), #"10 of Clubs"
    Card(15, Card.suits[0], 14, 3, 4, False), #"King of Clubs"
    Card(16, Card.suits[0], 9, 4, 0, False), #"9 of Clubs"
    Card(17, Card.suits[1], 11, 1, 11, False), #"Ace of Spades"
    Card(18, Card.suits[1], 10, 2, 10, False), #"10 of Spades"
    Card(19, Card.suits[1], 14, 3, 4, False), #"King of Spades"
    Card(20, Card.suits[1], 9, 4, 0, False), #"9 of Spades"
    Card(21, Card.suits[2], 11, 1, 11, False), #"Ace of Hearts"
    Card(22, Card.suits[2], 14, 2, 4, False), #"King of Hearts"
    Card(23, Card.suits[2], 9, 3, 0, False), #"9 of Hearts"
]

Card.HIGHEST_CARD_VALUE = max(card.value for card in Card.CARDTYPES)