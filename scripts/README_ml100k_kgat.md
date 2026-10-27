# Build MovieLens-100K data for KGAT

## Requirements

- Python 3.10 or newer.
- These local source files:

```text
data-for-Phon (1)/data/raw/
|-- ml-100k/{u.data,u.item,u.user,u.genre}
|-- ml-1m/movies.dat
|-- kb4rec/ml2fb.txt
`-- wikidata/facts.tsv
```

The source data is ignored by Git and must be downloaded or copied into these paths before running the scripts.
No additional Python packages are required.

## Run

Clone the original KGAT source first if `knowledge_graph_attention_network/` does not exist:

```powershell
git clone https://github.com/xiangwang1223/knowledge_graph_attention_network.git knowledge_graph_attention_network
```

The directory is ignored by this repository, so every new user must clone it locally. Then run from the repository
root:

```powershell
python scripts\prepare_ml100k_kgat.py
python scripts\enrich_ml100k_wikidata.py
```

If the base datasets already exist, only run the second command. Add `--overwrite` to regenerate existing output.

Final datasets are written to:

```text
knowledge_graph_attention_network/Data/ml-100k-chrono-wikidata/
knowledge_graph_attention_network/Data/ml-100k-chrono-timestamp-wikidata/
```

## How the KGAT data is built

The scripts remap users, movies, entities and relations to contiguous integer IDs. They generate KGAT interaction
files (`train.txt`, `val.txt`, `test.txt`), mapping files and `kg_final.txt` triples in `head relation tail` format.
Ratings and timestamps are stored separately for future interaction-edge weighting. The two dataset variants are
identical except that the timestamp variant includes timestamp labels.

## How the 10,000 interactions are selected

All MovieLens-100K interactions are sorted globally by ascending timestamp. The earliest 10,000 are selected and
split chronologically into 8,000 training, 1,000 validation and 1,000 test interactions. Training retains every
rating; ratings 4-5 are used as positive ground truth for validation and testing.

## How relations are collected

MovieLens relations are extracted from `u.item` and `u.user`, including genre, release information, IMDb ID and
user attributes.

Wikidata relations are obtained by:

1. matching MovieLens-100K movies to MovieLens-1M using title and year;
2. mapping MovieLens IDs to Freebase IDs with KB4Rec;
3. retrieving matching facts from the local Wikidata snapshot;
4. filtering future facts using the validation/test time cut-offs;
5. appending the new entities, relations and triples to the MovieLens graph.

The enrichment includes director, cast, producer, screenwriter, composer, genre, country, language, production
company, awards and other available movie relations. `kg_final.txt` contains MovieLens triples plus training-safe
Wikidata triples.

The Wikidata file is a 2026 snapshot; undated facts may still contain knowledge added after the 1997 interactions.
