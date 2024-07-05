# Cell 1
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Manual dictionary for model names
manual_model_name_map = {
    # base SQuAD
    "Llama3-ChatQA-1.5-8B": "llama3-8b\n(8B)",
    "falcon-7b-instruct": "falcon-7b\n(7B)",
    # tuned: SQuAD
    "bert": "bert\n(?)",
    "bert-large-cased-whole-word-masking-finetuned-squad": "largeBERT\n(336M)",
    "distilbert-base-cased-distilled-squad": "distilBERT\n(65M)",
    "mdeberta-v3-base-squad2": "mdeBERTa\n(278M)",
    "roberta-base-squad2": "roBERTa_base\n(124M)",
    "roberta-large-squad2": "roBERTa_large\n(354M)",
    "xlm-roberta-base-squad2": "xlm_roBERTa_base\n(277M)",
    # Gtuned: SQuAD
    "bert-multi-english-german-squad2": "multilang_BERT\n(177M)",
    "gelectra-base-germanquad-distilled": "GElectra_distil\n(109M)",
    "gelectra-base-germanquad": "GElectra_base\n(109M)",
    "gelectra-large-germanquad": "GElectra_large\n(335M)",
}


def plot_answer_length_distribution(base_dir):
    all_answer_lengths = []

    # Iterate through all JSON files in the directory
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                data = json.load(file)
                # Extract answer lengths
                answer_lengths = [
                    len(str(answer))
                    for answer in data.values()
                    if isinstance(answer, (str, int, float))
                ]
                all_answer_lengths.extend(answer_lengths)

    # Plot the distribution of answer lengths
    plt.figure(figsize=(10, 6))
    plt.hist(all_answer_lengths, bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel("Answer Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Answer Lengths Across All Files")
    plt.grid(True)
    plt.show()


# Function to plot bar chart with line graph for overall_score
def plot_bar_chart_with_line_graph(df, dataset_name, figure_root):
    df_sorted = df[df["dataset"] == dataset_name].sort_values(by="overall_score")
    plt.figure(figsize=(23, 6))
    bar_width = 0.25
    index = np.arange(len(df_sorted))

    # Bar chart for F1 Score, Exact Match, and Eval Other with subtle colors
    plt.bar(
        index, df_sorted["has_ans_f1"], bar_width, label="F1 Score", color="lightblue"
    )
    plt.bar(
        index + bar_width,
        df_sorted["has_ans_exact"],
        bar_width,
        label="Exact Match",
        color="lightgreen",
    )
    plt.bar(
        index + 2 * bar_width,
        df_sorted["eval_other"],
        bar_width,
        label="BBR Score",
        color="lightcoral",
    )

    # Line graph for Overall Score with a subtle color
    plt.plot(
        index + bar_width,
        df_sorted["overall_score"],
        color="blue",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=5,
        label="Overall Score",
    )

    plt.xlabel("Models")
    plt.ylabel("Scores")
    plt.title(f"Model Comparison (ascending order) - {dataset_name}")
    plt.xticks(
        index + bar_width,
        [manual_model_name_map.get(name, name) for name in df_sorted["short_name"]],
        rotation=0,
    )
    plt.legend()

    plt.tight_layout()

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"overall-bar-chart-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot Gardner Quadrants Style graph
def plot_gardner_quadrant(df, dataset_name, figure_root):
    df_sorted = df[df["dataset"] == dataset_name].sort_values(by="overall_score")
    plt.figure(figsize=(14, 6))
    plt.scatter(
        df_sorted["eval_other"], df_sorted["eval_v2_score_hasAns"], c="b", alpha=0.5
    )
    for i, txt in enumerate(df_sorted["short_name"]):
        plt.annotate(
            manual_model_name_map.get(txt, txt),
            (df_sorted["eval_other"].iat[i], df_sorted["eval_v2_score_hasAns"].iat[i]),
        )

    plt.xlabel("BBR Score (BLEU, BERT, ROUGE)")
    plt.ylabel("Eval V2 Score (F1, Exact Match)")
    plt.title(f"Model Performance Gardner Quadrant - {dataset_name}")

    plt.grid(True)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"overall-gardner-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot heat map
def plot_heat_map(df, dataset_name, figure_root):
    df_sorted = df[df["dataset"] == dataset_name].sort_values(by="overall_score")
    plt.figure(figsize=(12, 8))

    # Select and rename the columns for the heatmap
    heat_data = df_sorted[
        [
            "short_name",
            "has_ans_f1",
            "has_ans_exact",
            "bleu_score",
            "rouge_score",
            "bert_score",
            "overall_score",
        ]
    ].set_index("short_name")
    heat_data.columns = [
        (
            "F1"
            if col == "has_ans_f1"
            else "EXACT" if col == "has_ans_exact" else col.split("_")[0].upper()
        )
        for col in heat_data.columns
    ]
    heat_data.index = [
        manual_model_name_map.get(name, name) for name in heat_data.index
    ]

    # Create the heatmap
    sns.heatmap(heat_data, annot=True, cmap="Greens")

    plt.title(f"Model Metrics Heatmap - {dataset_name}")

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"overall-heatmap-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot bar chart to compare overall exact and f1 with HasAns exact and f1
def plot_bar_chart_comparison(df, dataset_name, figure_root):
    df_sorted = df[df["dataset"] == dataset_name].sort_values("overall_score")
    plt.figure(figsize=(20, 6))
    bar_width = 0.2
    index = np.arange(len(df_sorted))

    # Bar chart for Overall F1 Score, Overall Exact Match, HasAns F1 Score, and HasAns Exact Match with subtle colors
    plt.bar(
        index, df_sorted["f1"], bar_width, label="Overall F1 Score", color="darkblue"
    )
    plt.bar(
        index + bar_width,
        df_sorted["exact"],
        bar_width,
        label="Overall Exact Match",
        color="lightblue",
    )
    plt.bar(
        index + 2 * bar_width,
        df_sorted["has_ans_f1"],
        bar_width,
        label="HasAns F1 Score",
        color="darkgreen",
    )
    plt.bar(
        index + 3 * bar_width,
        df_sorted["has_ans_exact"],
        bar_width,
        label="HasAns Exact Match",
        color="lightgreen",
    )

    # Calculate eval_v2_score and eval_v2_score_hasAns
    df_sorted["eval_v2_score"] = 0.5 * (df_sorted["exact"] + df_sorted["f1"])
    df_sorted["eval_v2_score_hasAns"] = 0.5 * (
        df_sorted["has_ans_exact"] + df_sorted["has_ans_f1"]
    )

    # Plot lines for eval_v2_score and eval_v2_score_hasAns
    plt.plot(
        index + 1.5 * bar_width,
        df_sorted["eval_v2_score"],
        color="blue",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=5,
        label="Overall Average F1 & Exact",
    )
    plt.plot(
        index + 1.5 * bar_width,
        df_sorted["eval_v2_score_hasAns"],
        color="green",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=5,
        label="HasAns Average F1 & Exact",
    )

    plt.xlabel("Models")
    plt.ylabel("Scores")
    plt.title(
        f"Comparison of F1 & Exact Match Scores Overall and for Questions with Answers - {dataset_name}"
    )
    plt.xticks(
        index + 1.5 * bar_width,
        [manual_model_name_map.get(name, name) for name in df_sorted["short_name"]],
        rotation=0,
    )
    plt.legend()

    plt.tight_layout()

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"evaluate-v2-bar-chart-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot heat map to compare overall exact and f1 with HasAns exact and f1
def plot_comparison_heat_map(df, dataset_name, figure_root):
    df_sorted = df[df["dataset"] == dataset_name].sort_values("overall_score")
    plt.figure(figsize=(14, 8))

    # Select and rename the columns for the heatmap
    heat_data_comparison = df_sorted[
        ["short_name", "f1", "exact", "has_ans_f1", "has_ans_exact"]
    ].set_index("short_name")
    heat_data_comparison.columns = [
        "Overall F1",
        "Overall Exact",
        "HasAns F1",
        "HasAns Exact",
    ]

    # Update index labels with manual model name map
    heat_data_comparison.index = [
        manual_model_name_map.get(name, name) for name in heat_data_comparison.index
    ]

    # Create the heatmap
    sns.heatmap(heat_data_comparison, annot=True, cmap="Blues")

    plt.title(
        f"Comparison of F1 & Exact Match Scores Overall and for Questions with Answers - {dataset_name}"
    )

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"evaluate-v2-heatmap-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot heat map for timing results
def plot_timing_heat_map(df, figure_root):
    # Sort the dataframe by overall_score
    df_sorted = df.sort_values("overall_score").drop_duplicates(subset=["short_name"])

    plt.figure(figsize=(8, 8))

    # Extract timing results and reshape data
    timing_data = df_sorted[["short_name", "SQuAD", "G"]].set_index("short_name")
    timing_data.columns = ["SQuAD v2.0", "GermanQuAD"]

    # Update index labels with manual model name map
    timing_data.index = [
        manual_model_name_map.get(name, name) for name in timing_data.index
    ]

    # Convert times to integers and round down
    timing_data = timing_data.fillna(0).astype(int)

    # Calculate the average time
    timing_data["Avg time"] = timing_data.mean(axis=1)

    # Create annotations with formatted times and scores
    annotations = timing_data.applymap(
        lambda x: f"{x:,}s" if isinstance(x, int) else f"{int(x):,}s"
    )

    # Create the heatmap
    sns.heatmap(timing_data, annot=annotations, fmt="", cmap="Reds")

    plt.title("Model Inference Time")

    # Rotate the axis descriptions by 90°
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, "timing-results-heatmap.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot combined heat map for average values of GermanQuAD and SQuAD
def plot_combined_heat_map(df, figure_root):
    # Select the relevant columns for both datasets
    relevant_columns = [
        "short_name",
        "has_ans_f1",
        "has_ans_exact",
        "bleu_score",
        "rouge_score",
        "bert_score",
        "overall_score",
    ]

    # Split the dataframe into SQuAD and GermanQuAD
    squad_df = df[df["dataset"] == "SQuAD"][relevant_columns]
    germanquad_df = df[df["dataset"] == "G"][relevant_columns]

    # Ensure 'short_name' is included in both dataframes
    squad_df = squad_df.set_index("short_name")
    germanquad_df = germanquad_df.set_index("short_name")

    # Compute the average
    combined_df = (squad_df + germanquad_df) / 2
    combined_df = combined_df.reset_index()

    # Select and rename the columns for the heatmap
    heat_data = combined_df.set_index("short_name")
    heat_data.columns = [
        (
            "F1"
            if "f1" in col
            else "EXACT" if "exact" in col else col.split("_")[0].upper()
        )
        for col in heat_data.columns
    ]
    heat_data.index = [
        manual_model_name_map.get(name, name) for name in heat_data.index
    ]

    # Plot the combined heat map
    heat_data_sorted = heat_data.sort_values(by="OVERALL")
    plt.figure(figsize=(12, 8))

    sns.heatmap(heat_data_sorted, annot=True, cmap="Greens")

    plt.title("Combined Model Metrics Heatmap")

    # SAVE ⬇️
    save_path = os.path.join(figure_root, "combined-heatmap.png")
    plt.savefig(save_path)

    plt.show()
