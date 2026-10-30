"""
The morphology questions, one multiple-choice question per trait. Edit freely:
classify.py reads nothing else, and every output column is named from the
`key`s here.

Each option is (key, text). The model is shown the options as letters A, B, C...
and answers with one letter; its probability over those letters is the answer.
`abstain` lists the options that mean "can't tell from this photo": they are
recorded, but a genus's call is made among the other options, so ten photos of
tiny flowers don't all vote for "can't tell".

Keep each question to at most 8 options (A-H) and keep the wording free of the
letters' own meaning (no "option A is best if...").
"""

PREAMBLE = ("You are an expert botanist. Look at the flowers in this "
            "photograph of a wild plant.")

QUESTIONS = [
    dict(
        key="unit",
        text="What is the flowering structure in the photo?",
        options=[
            ("solitary", "A single flower, or flowers that each stand on their own"),
            ("inflorescence", "An inflorescence: a cluster of several distinct flowers "
                              "(raceme, spike, umbel, panicle, cyme)"),
            ("head", "A flower head (capitulum): many tiny florets packed together so "
                     "the whole looks like one flower, as in a daisy, sunflower or thistle"),
            ("spadix", "A spadix: a fleshy spike of tiny flowers, usually with a "
                       "hood- or leaf-like spathe, as in Arum, Anthurium or calla"),
            ("spikelet", "Grass- or sedge-like spikelets of chaffy scales "
                         "(glumes and lemmas), with no petals"),
            ("catkin", "A catkin: a dangling or upright spike of tiny petal-less "
                       "flowers, as in willow, birch or alder"),
            ("other", "Some other structure"),
            ("none", "No flower is visible"),
        ],
        abstain={"other", "none"},
    ),
    dict(
        key="symmetry",
        text="What is the symmetry of an individual flower, seen face on?",
        options=[
            ("actinomorphic", "Radial (actinomorphic): it can be cut into matching "
                              "halves along several lines, like a buttercup or lily"),
            ("zygomorphic", "Bilateral (zygomorphic): only one line cuts it into "
                            "mirror halves, like a pea, orchid, violet or snapdragon"),
            ("asymmetric", "Asymmetric: no line cuts it into mirror halves"),
            ("unclear", "Can't tell: the flowers are too small, reduced or hidden"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="display",
        text="What makes the flowers conspicuous?",
        options=[
            ("petals", "Showy, colourful petals or tepals"),
            ("bracts", "Showy, colourful bracts or sepals rather than petals, as in "
                       "bougainvillea, poinsettia or flowering dogwood"),
            ("stamens", "The stamens: a brush or powder-puff of long filaments, as in "
                        "bottlebrush or mimosa"),
            ("reduced", "Nothing: the flowers are small, greenish or brownish and "
                        "reduced, typical of wind pollination"),
            ("unclear", "Can't tell"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="fusion",
        text="Are the petals separate or fused?",
        options=[
            ("free", "Separate: each petal can be pulled off on its own (polypetalous)"),
            ("fused", "Fused into a tube, bell, funnel or cup (sympetalous)"),
            ("no_petals", "There are no petals"),
            ("unclear", "Can't tell"),
        ],
        abstain={"no_petals", "unclear"},
    ),
    dict(
        key="form",
        text="Which best describes the shape of an individual flower?",
        options=[
            ("open", "Open: a flat dish, bowl, saucer or star"),
            ("tubular", "Tubular, trumpet- or funnel-shaped"),
            ("bell", "Bell- or urn-shaped"),
            ("lipped", "Two-lipped (bilabiate), as in mints, sage or snapdragon"),
            ("papilionaceous", "Pea-like: a banner petal above two wings and a keel"),
            ("orchid", "Orchid-like, with one petal modified into a lip (labellum)"),
            ("spurred", "With a prominent spur, as in columbine, larkspur or violet"),
            ("unclear", "Something else, or can't tell"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="merosity",
        text="How many petals or tepals does one flower have?",
        options=[
            ("3_6", "3 or 6"),
            ("4_8", "4 or 8"),
            ("5_10", "5 or 10"),
            ("many", "More than 10"),
            ("unclear", "Can't tell, or there are none"),
        ],
        abstain={"unclear"},
    ),
]

LETTERS = "ABCDEFGH"


def prompt_text(question):
    """The user turn for one question: preamble, question, lettered options."""
    lines = [PREAMBLE, "", question["text"], ""]
    for letter, (_, text) in zip(LETTERS, question["options"]):
        lines.append(f"{letter}. {text}")
    lines += ["", "Answer with the letter of the best option only."]
    return "\n".join(lines)


def option_keys(question):
    return [key for key, _ in question["options"]]


def validate():
    """Fail early on a malformed edit rather than an hour into a GPU job."""
    seen = set()
    for q in QUESTIONS:
        keys = option_keys(q)
        if q["key"] in seen:
            raise ValueError(f"duplicate question key {q['key']!r}")
        seen.add(q["key"])
        if not 2 <= len(keys) <= len(LETTERS):
            raise ValueError(f"{q['key']}: needs 2-{len(LETTERS)} options")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{q['key']}: duplicate option keys")
        if not q["abstain"] <= set(keys):
            raise ValueError(f"{q['key']}: abstain options not among its options")
        if len(set(keys) - q["abstain"]) < 2:
            raise ValueError(f"{q['key']}: needs at least 2 non-abstain options")


validate()
