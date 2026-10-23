import pandas as pd

from tkgat.kg.linking import match_ml100k_to_ml1m, norm_title, split_title


def test_split_and_norm_title():
    assert split_title("Toy Story (1995)") == ("Toy Story", 1995)
    assert norm_title("Ice Storm, The") == norm_title("The Ice Storm")
    assert norm_title("Boot, Das") == "dasboot"


def test_match_rules():
    a = pd.DataFrame({"item": [1, 2, 3], "title": ["Star Wars (1977)", "Boot, Das (1981)", "Nothing Like It (1990)"]})
    b = pd.DataFrame(
        {
            "item": [260, 2028, 9],
            "title": ["Star Wars: Episode IV - A New Hope (1977)", "Boat, The (Das Boot) (1981)", "Other (1990)"],
        }
    )
    m = match_ml100k_to_ml1m(a, b).set_index("item_100k")
    assert m.loc[1, "item_ml1m"] == 260 and m.loc[1, "rule"] == "contains+year"
    assert m.loc[2, "item_ml1m"] == 2028 and m.loc[2, "rule"] == "aka_title+year"
    assert pd.isna(m.loc[3, "item_ml1m"])
