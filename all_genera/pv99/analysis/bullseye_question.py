"""
The bullseye question for the pv99 top segments.

One multiple-choice question, asked of the context crop around each photo's
most confident SAM3 mask. Same mechanics as morph_questions.py: the options
are shown as letters, the model's next-token probabilities over those letters
are the answer, and the question is asked in both option orders and averaged.

"Bullseye" here means what can be seen in an ordinary (visible-light)
photograph: concentric zones of contrasting colour, a centre or a ring of
petal bases that stands out sharply from the outer part of the flower, so the
face reads like a target. Ultraviolet bullseyes that are invisible to us are
out of reach of these photos. `partial` collects faint or patchy contrast
(streaks, a few spots, a slightly darker centre) so that it does not have to
be forced into yes or no; `unclear` is the abstain option.
"""

PREAMBLE = ("You are an expert botanist. The photograph is cropped around one "
            "flower-like structure of a wild plant, with some surrounding context.")

QUESTION = dict(
    key="bullseye",
    label="Bullseye pattern",
    text=("Does the flower (or flower head) have a bullseye pattern? A bullseye is "
          "a target-like face: a central eye or a ring around the centre whose "
          "colour contrasts sharply with the outer part of the petals, as in "
          "black-eyed Susan (dark cone against yellow rays), blanket flower "
          "(red ray bases, yellow tips), plains coreopsis, flower-of-an-hour "
          "(Hibiscus trionum) or a poppy with dark blotches at the petal bases."),
    options=[
        ("bullseye", "Yes: a clear bullseye, the centre or the petal bases form an eye "
                     "or ring that contrasts sharply with the rest of the flower"),
        ("partial", "Partly: some contrast toward the centre, but faint, patchy, or "
                    "only streaks or scattered spots, not a clear ring or eye"),
        ("none", "No: the flower face is about one colour and the centre does not "
                 "stand out as a ring or eye"),
        ("unclear", "Can't tell: no flower face is visible, or the flowers are too "
                    "small or hidden"),
    ],
    abstain={"unclear"},
)

LETTERS = "ABCDEFGHIJ"


def option_keys(question=QUESTION):
    return [key for key, _ in question["options"]]


def prompt_text(order, question=QUESTION):
    """The user turn: preamble, question, options lettered in `order`."""
    lines = [PREAMBLE, "", question["text"], ""]
    for letter, i in zip(LETTERS, order):
        lines.append(f"{letter}. {question['options'][i][1]}")
    lines += ["", "Answer with the letter of the best option only."]
    return "\n".join(lines)
