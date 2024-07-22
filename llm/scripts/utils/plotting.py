import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from adjustText import adjust_text


def plot_answer_length_distribution(base_dir, fontsize=12, figsize=(10, 6)):
    """
    Open window with plot about distribution of answer lengths in given directory.

    :param str base_dir: Directory which will be searched for JSON files which will
                         be used to create the distribution
    """
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
    plt.figure(figsize=figsize)
    plt.hist(all_answer_lengths, bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel("Answer Length (characters)", fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize)
    plt.title("Distribution of Answer Lengths Across All Files", fontsize=fontsize)
    plt.grid(True)
    plt.show()


# Function to plot bar chart with line graph for overall_score
def plot_bar_chart_with_line_graph(
    df, dataset_name, figure_root, model_names, fontsize=12, figsize=(8, 8)
):
    """
    Open window with plot displaying a bar plot for the different metrics and above that
    a line plot with the overall score.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str dataset_name: Name of dataset which should be plotted; either "SQuAD" or "GermanQuAD"
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
    df_sorted = df[df["dataset"] == dataset_name].sort_values(by="overall_score")
    plt.figure(figsize=figsize)
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

    plt.xlabel("Models", fontsize=fontsize)
    plt.ylabel("Scores", fontsize=fontsize)
    plt.title(f"Model Comparison (ascending order) - {dataset_name}", fontsize=fontsize)
    plt.xticks(
        index + bar_width,
        [model_names.get(name, name) for name in df_sorted["short_name"]],
        rotation=0,
        fontsize=fontsize,
    )
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.tight_layout()

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"overall-bar-chart-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot Gardner Quadrants Style graph
def plot_gardner_quadrant(
    df, dataset_name, figure_root, model_names, fontsize=12, figsize=(14, 6)
):
    """
    Open window with plot displaying a combined score of BLEU, BERTscore and ROUGE against
    F1 and exact match in a scatter plot.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str dataset_name: Name of dataset which should be plotted; either "SQuAD" or "GermanQuAD"
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
    df_sorted = df[df["dataset"] == dataset_name].sort_values(by="overall_score")
    plt.figure(figsize=figsize)
    plt.scatter(
        df_sorted["eval_other"], df_sorted["eval_v2_score_hasAns"], c="b", alpha=0.5
    )

    texts = []
    for i, txt in enumerate(df_sorted["short_name"]):
        model_name = model_names.get(txt, txt).split("\n")[
            0
        ]  # Extract only the model name
        texts.append(
            plt.text(
                df_sorted["eval_other"].iat[i],
                df_sorted["eval_v2_score_hasAns"].iat[i],
                model_name,
                fontsize=fontsize,
                ha="right",
            )
        )

    # Use adjust_text to minimize overlaps
    adjust_text(
        texts,
        only_move={"text": "y"},
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
        force_text=0.3,
        force_points=0.3,
    )

    plt.xlabel("BBR Score (BLEU, BERT, ROUGE)", fontsize=fontsize)
    plt.ylabel("Eval V2 Score (F1, Exact Match)", fontsize=fontsize)
    plt.title(f"Model Performance Gardner Quadrant - {dataset_name}", fontsize=fontsize)

    plt.grid(True)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"overall-gardner-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot heat map
def plot_heat_map(
    df,
    dataset_name,
    figure_root,
    model_names,
    cmap="Greens",
    fontsize=12,
    annot_fontsize=14,
    figsize=(12, 8),
):
    """
    Open window with plot displaying a heatmap over all evaluated metrics.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str dataset_name: Name of dataset which should be plotted; either "SQuAD" or "GermanQuAD"
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
    df_sorted = df[df["dataset"] == dataset_name].sort_values(by="overall_score")
    plt.figure(figsize=figsize)

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
    heat_data.index = [model_names.get(name, name) for name in heat_data.index]

    # Create the heatmap
    sns.heatmap(heat_data, annot=True, cmap=cmap, annot_kws={"size": annot_fontsize})

    plt.title(f"Model Metrics Heatmap - {dataset_name}", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"overall-heatmap-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot bar chart to compare overall exact and f1 with HasAns exact and f1
def plot_bar_chart_comparison(
    df, dataset_name, figure_root, model_names, fontsize=12, figsize=(20, 6)
):
    """
    Open window with plot displaying a bar plot with all given models besides each other
    visualizing the combined F1 and exact match score and the has-answer scores.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str dataset_name: Name of dataset which should be plotted; either "SQuAD" or "GermanQuAD"
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
    df_sorted = df[df["dataset"] == dataset_name].sort_values("overall_score")
    plt.figure(figsize=figsize)
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

    plt.xlabel("Models", fontsize=fontsize)
    plt.ylabel("Scores", fontsize=fontsize)
    plt.title(
        f"Comparison of F1 & Exact Match Scores Overall and for Questions with Answers - {dataset_name}",
        fontsize=fontsize,
    )
    plt.xticks(
        index + 1.5 * bar_width,
        [model_names.get(name, name) for name in df_sorted["short_name"]],
        rotation=0,
        fontsize=fontsize,
    )
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.tight_layout()

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"evaluate-v2-bar-chart-{dataset_name}.png")
    plt.savefig(save_path)
    plt.show()


# Function to plot heat map to compare overall exact and f1 with HasAns exact and f1
def plot_comparison_heat_map(
    df,
    dataset_name,
    figure_root,
    model_names,
    cmap="Blues",
    fontsize=12,
    annot_fontsize=14,
    figsize=(14, 8),
):
    """
    Open window with plot displaying a heatmap with all given models visualizing the
    combined F1 and exact match score and the has-answer scores for each.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str dataset_name: Name of dataset which should be plotted; either "SQuAD" or "GermanQuAD"
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
    df_sorted = df[df["dataset"] == dataset_name].sort_values("overall_score")
    plt.figure(figsize=figsize)

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
        model_names.get(name, name) for name in heat_data_comparison.index
    ]

    # Create the heatmap
    sns.heatmap(
        heat_data_comparison, annot=True, cmap=cmap, annot_kws={"size": annot_fontsize}
    )

    plt.title(
        f"Comparison of F1 & Exact Match Scores Overall and for Questions with Answers - {dataset_name}",
        fontsize=fontsize,
    )
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, f"evaluate-v2-heatmap-{dataset_name}.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot heat map for timing results
def plot_timing_heat_map(
    df,
    figure_root,
    model_names,
    cmap="Reds",
    fontsize=12,
    annot_fontsize=14,
    figsize=(8, 8),
):
    """
    Open window with heatmap visualizing the runtime of each model contained in the dataframe.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
    # Sort the dataframe by overall_score
    df_sorted = df.sort_values("overall_score").drop_duplicates(subset=["short_name"])

    plt.figure(figsize=figsize)

    # Extract timing results and reshape data
    timing_data = df_sorted[["short_name", "SQuAD", "G"]].set_index("short_name")
    timing_data.columns = ["SQuAD v2.0", "GermanQuAD"]
    # update model names
    timing_data.index = [model_names.get(name, name) for name in timing_data.index]

    # Calculate the average time
    timing_data = timing_data.fillna(0).astype(int)
    timing_data["Avg time"] = timing_data.mean(axis=1)

    # Sort by average time
    timing_data = timing_data.sort_values(by="Avg time")

    # Create annotations with formatted times and scores
    annotations = timing_data.applymap(
        lambda x: f"{x:,}s" if isinstance(x, int) else f"{int(x):,}s"
    )

    # Create the heatmap
    sns.heatmap(
        timing_data,
        annot=annotations,
        fmt="",
        cmap=cmap,
        annot_kws={"size": annot_fontsize},
    )
    plt.title("Model Inference Time", fontsize=fontsize)
    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, "timing-results-heatmap.png")
    plt.savefig(save_path)

    plt.show()


# Function to plot combined heat map for average values of GermanQuAD and SQuAD
def plot_combined_heat_map(
    df,
    figure_root,
    model_names,
    cmap="Greens",
    fontsize=12,
    annot_fontsize=14,
    figsize=(12, 8),
):
    """
    Open window with heatmap displaying the combined values of all metrics for all datasets.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
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
    germanquad_df = df[df["dataset"] == "GermanQuAD"][relevant_columns]

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
    heat_data.index = [model_names.get(name, name) for name in heat_data.index]

    # Plot the combined heat map
    heat_data_sorted = heat_data.sort_values(by="OVERALL")
    plt.figure(figsize=figsize)

    sns.heatmap(
        heat_data_sorted, annot=True, cmap=cmap, annot_kws={"size": annot_fontsize}
    )

    plt.title("Combined Model Metrics Heatmap", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, "combined-heatmap.png")
    plt.savefig(save_path)

    plt.show()


def plot_combined_gardner_quadrant(
    df,
    figure_root,
    model_names,
    fontsize=12,
    annot_fontsize=14,
    tick_fontsize=16,
    figsize=(14, 6),
):
    """
    Open window with scatter plot displaying all metrics of SQuAD and GermanQuAD combined.
    This plot will also be saved to disk.

    :param pandas.DataFrame df: Dataframe with all evaluation results (see eval_results/) to be plotted
    :param str figure_root: Directory to which the graph should be saved to
    :param dict model_names: Human-readable display names for each model, if available
    """
    # Select the relevant columns for both datasets
    relevant_columns = [
        "short_name",
        "has_ans_f1",
        "has_ans_exact",
        "bleu_score",
        "rouge_score",
        "bert_score",
        "exact",
        "f1",
    ]

    # Split the dataframe into SQuAD and GermanQuAD
    squad_df = df[df["dataset"] == "SQuAD"][relevant_columns]
    germanquad_df = df[df["dataset"] == "GermanQuAD"][relevant_columns]

    # Ensure 'short_name' is included in both dataframes
    squad_df = squad_df.set_index("short_name")
    germanquad_df = germanquad_df.set_index("short_name")

    # Compute the average
    combined_df = (squad_df + germanquad_df) / 2
    combined_df = combined_df.reset_index()

    # Calculate the additional columns needed for plotting
    combined_df["eval_v2_score_hasAns"] = 0.5 * (
        combined_df["has_ans_exact"] + combined_df["has_ans_f1"]
    )
    combined_df["eval_other"] = (
        1
        / 3
        * (
            combined_df["bert_score"]
            + combined_df["bleu_score"]
            + combined_df["rouge_score"]
        )
    )
    combined_df["overall_score"] = 0.5 * (
        combined_df["eval_v2_score_hasAns"] + combined_df["eval_other"]
    )

    # Sort the combined dataframe by overall score
    combined_df_sorted = combined_df.sort_values(by="overall_score")

    plt.figure(figsize=figsize)
    plt.scatter(
        combined_df_sorted["eval_other"],
        combined_df_sorted["eval_v2_score_hasAns"],
        c="b",
        alpha=0.5,
    )

    texts = []
    for i, txt in enumerate(combined_df_sorted["short_name"]):
        model_name = model_names.get(txt, txt).split("\n")[
            0
        ]  # Extract only the model name
        texts.append(
            plt.text(
                combined_df_sorted["eval_other"].iat[i],
                combined_df_sorted["eval_v2_score_hasAns"].iat[i],
                model_name,
                fontsize=annot_fontsize,
                ha="right",
            )
        )

    # Use adjust_text to minimize overlaps
    adjust_text(
        texts,
        only_move={"text": "y"},
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
        force_text=0.3,
        force_points=0.3,
    )

    plt.xlabel("BBR Score (BLEU, BERT, ROUGE)", fontsize=fontsize)
    plt.ylabel("Eval V2 Score (F1, Exact Match)", fontsize=fontsize)
    plt.title("Combined Model Performance Gardner Quadrant", fontsize=fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(True)

    # SAVE ⬇️
    save_path = os.path.join(figure_root, "combined-gardner-quadrant.png")
    plt.savefig(save_path)

    plt.show()
