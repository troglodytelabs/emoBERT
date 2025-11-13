from datasets import load_dataset

go = load_dataset("go_emotions", "simplified")
xed = load_dataset(
    "csv",
    data_files={
        "train": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/AnnotatedData/en-annotated.tsv",
    },
    delimiter="\t",
    column_names=["sentence", "labels"],
)

# print(go)
# print(xed)

xed = xed.rename_column("sentence", "text")
