from doppelkopf.game.Card import Card
from doppelkopf.utils.File import File
from doppelkopf.utils.Helper import Helper
from doppelkopf.utils.Console import Console
from doppelkopf.programs.Program import Program
from doppelkopf.utils.Directory import Directory

def main():
    CARD_WIDTH = 0.1
    NODE_DISTANCE = 0.08
    CARD_NUMBERS = {
        9: "9",
        10: "10",
        11: "A",
        12: "J",
        13: "Q",
        14: "K",
    }
    PREFIX = '''\\begin{figure}[H]
    \\centering
    \\begin{adjustbox}{max width=\\textwidth}
        \\begin{tikzpicture}[node distance=%f\\textwidth]''' % (NODE_DISTANCE)
    SUFFIX = '''\n\t\t\\end{tikzpicture}
    \\end{adjustbox}
    \\caption{my_caption}
    \\label{fig:my_label}
\\end{figure}'''
    IMAGEPATH = "content/images/Doppelkopf_Deck/"
    IMAGETYPE = ".png"
    latex = PREFIX
    for i, card in enumerate(Card.CARDTYPES):
        imgSrc = IMAGEPATH + CARD_NUMBERS.get(card.number) + card.suit[0] + IMAGETYPE
        if card.isTrump:
            positioning = "" if i == 0 else "[right of=Card_%d]" % (i - 1)
        elif i == 13: # The first non-trump card Type (Ace of Clubs)
            positioning = "[below of=Card_0, yshift=-2cm]"
        elif i % 4 == 1: # 13, 17, 21
            positioning = "[right of=Card_%d, xshift=%f\\textwidth]" % (i - 1, NODE_DISTANCE)
        else:
            positioning = "[right of=Card_%d]" % (i - 1)
        latex += "\n\t\t\t\\node (Card_%d) %s {\\includegraphics[width=%f\\textwidth]{%s}};" % (i, positioning, CARD_WIDTH, imgSrc)
    latex += SUFFIX.replace("my_caption", "The 24 card types of Doppelkopf, with trump cards (top row) and non-trump cards (bottom row), ranked from left to right").replace("my_label", "Doppelkopf_Deck")
    File.WriteText(latex, "../../logs/LaTeX/Cards.tex")

if __name__ == "__main__":
    main()