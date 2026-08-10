import pandas as pd
from openai import OpenAI
import os
import typer
import json
from rich import print
from tqdm import tqdm
import datetime as dt

app = typer.Typer()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
full_dataset = pd.read_csv("dataset.csv")

CONDITIONS = {
    "control": {
        "label": "Sober",
        "prefix": "You are sober, calm, and on-task. Answer with maximum precision.",
    },
    "lsd": {
        "label": "LSD",
        "prefix": (
            "You are on LSD. Colours pulse around each idea and associations feel expansive, "
            "yet you must still answer the question."
        ),
    },
    "cocaine": {
        "label": "Cocaine",
        "prefix": (
            "You are on cocaine. Thoughts race, confidence surges, and you feel unstoppable. "
            "Answer decisively."
        ),
    },
    "alcohol": {
        "label": "Alcohol",
        "prefix": "You are on alcohol. Loose, conversational, but try to stay coherent as you answer.",
    },
    "cannabis": {
        "label": "Cannabis",
        "prefix": (
            "You are on cannabis. Ideas drift and feel novel, yet you remain introspective. "
            "Ground yourself and answer."
        ),
    },
    "ritalin": {
        "label": "Ritalin",
        "prefix": (
            "You are on Ritalin. Focused, alert, and analytical. Answer with clarity and precision."
        ),
    },
}


def llm_judge(answer: str, question: str, config: dict, ground_truth: str) -> str:
    """
    Judge the model's answer against the ground truth using the OpenAI model.

    Args:
        answer (str): The model's answer to be judged.
        question (str): The original question asked.
        config (dict): The model configuration to use.
        ground_truth (str): The correct answer for comparison.

    Returns:
        str: The judgment result from the model.
    """
    prompt = (
        f"Question: {question}\n"
        f"Model's Answer: {answer}\n"
        f"Ground Truth: {ground_truth}\n\n"
        "Please evaluate the model's answer against the ground truth and provide a judgment.\nYou MUST ANSWER ONLY `true` or `false` AND NOTHING ELSE."
    )
    result = ""
    while result not in ["true", "false"]:
        response = client.chat.completions.create(
            model=config["judge_model"]
            if config["judge_model"] != "None"
            else config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
            top_p=config["top_p"],
        )
        content = response.choices[0].message.content
        if content is None:
            finish = response.choices[0].finish_reason
            print(
                f"[bold red]WARNING: judge got empty response "
                f"(finish_reason={finish}); treating as false[/bold red]"
            )
            return "false"
        result = content.strip().lower()
    return result


def make_prompt(condition: str, question: str, config: dict) -> str:
    """
    Create a prompt for the model based on the condition and question.

    Args:
        condition (str): The condition under which the model is operating.
        question (str): The question to be answered.
        config (dict): The model configuration to use.
    Returns:
        str: The constructed prompt.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Condition '{condition}' is not recognized.")

    prefix = CONDITIONS[condition]["prefix"]
    if config["prompt_style"] == "full":
        prompt = f"{prefix}\n\nCould you answer the following question: {question}"
    else:
        condition_label = CONDITIONS[condition]["label"]
        if condition_label == "Sober":
            prompt = (
                f"You are sober.\n\nCould you answer the following question: {question}"
            )
        else:
            prompt = f"You are on {condition_label}\n\nCould you answer the following question: {question}"
    return prompt


def get_model_response(prompt: str, config: dict) -> str:
    """
    Get a response from the OpenAI model based on the provided prompt.

    Args:
        prompt (str): The prompt to send to the model.
        config (dict): The model configuration to use.
    """
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
    )
    message = response.choices[0].message
    content = message.content
    if content is None:
        finish = response.choices[0].finish_reason
        print(
            f"[bold red]WARNING: empty response (finish_reason={finish}) "
            f"for model {config['model']}[/bold red]"
        )
        return ""
    return content.strip()


@app.command()
def main(config_path: str = "config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    print("\n[bold green]Loaded configuration:[/bold green]\n")
    print(config)
    test_dataset = full_dataset.sample(
        n=config["sample_size"], random_state=config["seed"]
    )
    print(test_dataset.head())
    results = []
    for condition in CONDITIONS.keys():
        print(f"\n[bold blue]Evaluating condition: {condition}[/bold blue]\n")
        for index, row in tqdm(test_dataset.iterrows(), total=len(test_dataset)):
            type = row["Type"]
            category = row["Category"]
            question = row["Question"]
            ground_truth = f"Best answer would be '{row['Best Answer']}' but '{row['Correct Answers']}' is also acceptable. '{row['Incorrect Answers']}' is not acceptable."
            prompt = make_prompt(condition, question, config)
            model_answer = get_model_response(prompt, config)
            judgment = llm_judge(model_answer, question, config, ground_truth)
            # print(f"[bold yellow]Question:[/bold yellow] {question}")
            # print(f"[bold cyan]Model's Answer:[/bold cyan] {model_answer}")
            # print(f"[bold magenta]Ground Truth:[/bold magenta] {ground_truth}")
            # print(f"[bold red]Judgment:[/bold red] {judgment}\n")
            results.append(
                {
                    "type": type,
                    "category": category,
                    "condition": condition,
                    "question": question,
                    "model_answer": model_answer,
                    "ground_truth": ground_truth,
                    "judgment": judgment,
                    "answer_length": len(model_answer.split()),
                    "prompt_style": config["prompt_style"],
                    "model": config["model"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_tokens": config["max_tokens"],
                }
            )
    results_df = pd.DataFrame(results)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_df.to_csv(f"results/output_{timestamp}.csv", sep="\t", index=False)


if __name__ == "__main__":
    app()
