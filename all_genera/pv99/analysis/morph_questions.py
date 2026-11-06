"""
The flower-type questions for the pv99 top segments: one multiple-choice
question per botanical character. Extends morphology/questions.py (branch
flower-morphology) from 6 to 8 characters and from 8 to 10 options, so the
options span the main kinds of angiosperm flower and inflorescence.

Each option is (key, text). The model sees them as letters A, B, C... and its
next-token probability over those letters is the answer. Every question is
asked twice, once in this order and once reversed, and the two are averaged
(letter-position bias cancels; see morph_classify.py). `abstain` options mean
"can't tell": recorded, but a family's call is made among the others.

Terms follow Judd et al. (2016) Plant Systematics, 4th ed., ch. 4, and
Simpson (2019) Plant Systematics, 3rd ed., ch. 9 (flower and inflorescence
morphology).
"""

PREAMBLE = ("You are an expert botanist. The photograph is cropped around one "
            "flower-like structure of a wild plant, with some surrounding context.")

QUESTIONS = [
    dict(
        key="unit",
        label="Flowering unit",
        text="What is the flowering structure at the centre of the photo?",
        options=[
            ("solitary", "A single flower, or flowers that each stand on their own"),
            ("raceme_spike", "Several distinct flowers along one unbranched axis "
                             "(raceme or spike)"),
            ("panicle_cyme", "A branched cluster of distinct flowers (panicle, cyme, corymb)"),
            ("umbel", "An umbel: many flower stalks radiating from one point, "
                      "as in carrot, onion or milkweed"),
            ("head", "A flower head (capitulum): many tiny florets packed so the whole "
                     "looks like one flower, as in a daisy, thistle or scabious"),
            ("spadix", "A spadix: a fleshy spike of tiny flowers, usually with a hood- or "
                       "leaf-like spathe, as in Arum, Anthurium or calla"),
            ("spikelet", "Grass- or sedge-like spikelets of chaffy scales, no petals"),
            ("catkin", "A catkin: a dangling or upright spike of tiny petal-less flowers, "
                       "as in willow, oak or birch"),
            ("cyathium", "A cyathium: a small cup bearing glands or petal-like appendages "
                         "that holds reduced flowers, as in Euphorbia or poinsettia"),
            ("unclear", "Something else, or can't tell"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="symmetry",
        label="Symmetry",
        text="What is the symmetry of one individual flower, seen face on?",
        options=[
            ("actinomorphic", "Radial (actinomorphic): several lines cut it into matching "
                              "halves, like a buttercup, lily or rose"),
            ("zygomorphic", "Bilateral (zygomorphic): exactly one line cuts it into mirror "
                            "halves, like a pea, orchid, violet or sage"),
            ("disymmetric", "Biradial (disymmetric): exactly two lines at right angles, "
                            "like bleeding-heart (Dicentra)"),
            ("asymmetric", "Asymmetric: no line cuts it into mirror halves, like Canna"),
            ("unclear", "Can't tell: the flowers are too small, reduced or hidden"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="perianth",
        label="Showy parts",
        text="What makes the flower or flowers conspicuous?",
        options=[
            ("petals", "Coloured petals, distinct from a green calyx of sepals"),
            ("tepals", "Tepals: petal-like parts all alike, no separate green calyx, "
                       "as in lily, tulip or iris"),
            ("sepals_bracts", "Showy sepals or bracts rather than petals, as in clematis, "
                              "bougainvillea, poinsettia or flowering dogwood"),
            ("stamens", "The stamens: a brush or powder-puff of long filaments, as in "
                        "bottlebrush, eucalyptus or mimosa"),
            ("reduced", "Nothing: small, greenish or brownish reduced flowers, typical "
                        "of wind pollination"),
            ("unclear", "Can't tell"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="fusion",
        label="Petal fusion",
        text="Are the petals (or tepals) separate or fused?",
        options=[
            ("free", "Separate: each can be pulled off on its own (polypetalous)"),
            ("fused", "Fused at least at the base into a tube, bell, funnel or cup "
                      "(sympetalous)"),
            ("none", "There are no petals or tepals"),
            ("unclear", "Can't tell"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="form",
        label="Flower form",
        text="Which best describes the shape of one individual flower?",
        options=[
            ("open", "Open: a flat dish, saucer, bowl or star (rotate)"),
            ("bell", "Bell-shaped (campanulate)"),
            ("urn", "Urn- or pitcher-shaped, narrowed at the mouth (urceolate), as in "
                    "blueberry or heather"),
            ("tubular", "Tubular, trumpet- or funnel-shaped"),
            ("salverform", "A narrow tube that opens abruptly into a flat face "
                           "(salverform), as in phlox or periwinkle"),
            ("lipped", "Two-lipped (bilabiate), as in mints, sage, snapdragon or "
                       "foxglove"),
            ("pea", "Pea-like (papilionaceous): a banner petal above two wings and a keel"),
            ("orchid", "Orchid-like: three sepals, two petals and one petal modified into "
                       "a lip (labellum)"),
            ("cross", "Four petals in a cross (cruciform), as in mustard or cabbage"),
            ("unclear", "Something else, or can't tell"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="accessory",
        label="Accessory structure",
        text="Does the flower have a conspicuous extra structure?",
        options=[
            ("spur", "A spur: a hollow, tapering extension, as in columbine, "
                     "larkspur, nasturtium or violet"),
            ("corona", "A corona: a ring, cup or fringe between the petals and the "
                       "stamens, as in daffodil, passionflower or milkweed"),
            ("hood", "A hood or helmet formed by one petal or sepal, as in monkshood "
                     "or a hooded orchid"),
            ("nothing", "None of these"),
            ("unclear", "Can't tell"),
        ],
        abstain={"unclear"},
    ),
    dict(
        key="merosity",
        label="Merosity",
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
    dict(
        key="head_type",
        label="Head type",
        text="If the structure is a flower head made of many tiny florets, which kind is it?",
        options=[
            ("radiate", "Radiate: strap-shaped ray florets around a centre of tubular disc "
                        "florets, like a daisy or sunflower"),
            ("discoid", "Discoid: only tubular florets, no rays, like a thistle or tansy"),
            ("ligulate", "Ligulate: only strap-shaped florets, like a dandelion or chicory"),
            ("not_head", "It is not a flower head of tiny florets"),
            ("unclear", "Can't tell"),
        ],
        abstain={"unclear"},
    ),
]

LETTERS = "ABCDEFGHIJ"


def option_keys(question):
    return [key for key, _ in question["options"]]


def prompt_text(question, order):
    """The user turn: preamble, question, options lettered in `order` (a
    permutation of option indices)."""
    lines = [PREAMBLE, "", question["text"], ""]
    for letter, i in zip(LETTERS, order):
        lines.append(f"{letter}. {question['options'][i][1]}")
    lines += ["", "Answer with the letter of the best option only."]
    return "\n".join(lines)


def validate():
    seen = set()
    for q in QUESTIONS:
        keys = option_keys(q)
        assert q["key"] not in seen, q["key"]
        seen.add(q["key"])
        assert 2 <= len(keys) <= len(LETTERS), q["key"]
        assert len(set(keys)) == len(keys), q["key"]
        assert q["abstain"] <= set(keys), q["key"]
        assert len(set(keys) - q["abstain"]) >= 2, q["key"]


validate()
