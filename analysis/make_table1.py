"""Recompute Table 1, the token and cost table (Table 2, Appendix C) and the numbers quoted in the text.

  python3 analysis/make_table1.py

It reads our scores in paper_results/scores/ and the token counts in paper_results/token_usage.csv,
and prints:
  - Table 1: recall, precision and Expected Recall@2000 for the four VLM runs, and the cost per
    10,000 images,
  - the seed spread of Gemini Likert (seeds 44-46), the high-thinking run, and a few numbers from
    the text (recall in the top 300, candidates scoring 45 or more).
It also writes the LaTeX tables to generated/table1.tex and generated/appendix_cost_table.tex.
The Zoobot rows of Table 1 come from analysis/zoobot_baseline.py.

To use your own runs, point SCORES (below) at generated/scores/ and add a row per run to a copy of
token_usage.csv, using the token counts at the bottom of each score file.

What to expect: exactly the paper's numbers, e.g. Gemini Likert recall 0.982 at score >= 33.
"""

import argparse
import csv
import os
from pathlib import Path
import statistics

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import scores_csv

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE_ROOT / "paper_results"
SCORES = RESULTS / "scores"  # each run is SCORES / f"{run}.csv"
TABLE1 = [{'run': 'gemini_likert_seed44',
  'model': 'Gemini 3.1 Flash-Lite',
  'method': 'Likert',
  'provider': 'gemini',
  'threshold': 33},
 {'run': 'qwen_likert_seed44',
  'model': 'Qwen3.5-397B-A17B',
  'method': 'Likert',
  'provider': 'qwen',
  'threshold': 36},
 {'run': 'gemini_tournament_seed44',
  'model': 'Gemini 3.1 Flash-Lite',
  'method': 'Tournament',
  'provider': 'gemini',
  'threshold': 10},
 {'run': 'qwen_tournament_seed44',
  'model': 'Qwen3.5-397B-A17B',
  'method': 'Tournament',
  'provider': 'qwen',
  'threshold': 10}]
REVIEW_SET_SIZE = 2000

def recall_curve(scores, truth, total_positives):
    """Average over random ordering within ties; curve[N-1] is expected recall@N."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    curve = [0.0] * n

    seen, seen_tp, i = 0, 0.0, 0
    while i < n:
        band_score = ranked[i][1]
        j, band_tp = i, 0
        while j < n and ranked[j][1] == band_score:
            if truth.get(ranked[j][0]):
                band_tp += 1
            j += 1
        band = j - i
        for k in range(1, band + 1):  # N = seen + k
            curve[seen + k - 1] = (seen_tp + band_tp * (k / band)) / total_positives
        seen += band
        seen_tp += band_tp
        i = j

    return curve


def evaluate(run_id, threshold, provider, usage):
    rows = scores_csv.read(SCORES / f'{run_id}.csv')
    scores = {row['filename']: float(row['imagescore']) for row in rows}
    truth = {row['filename']: int(row['interesting']) for row in rows}
    n_images = len(scores)
    positives = sum(truth.values())
    shortlist = [name for name, score in scores.items() if score >= threshold]
    true_positives = sum(truth[name] for name in shortlist)
    expected_recall = recall_curve(scores, truth, positives)
    tokens = usage[run_id]
    input_tokens, output_tokens, thinking_tokens = (int(tokens[k]) for k in ('input', 'output', 'thinking'))
    # Qwen costs are recorded charges; Gemini bills visible output and thinking.
    if provider == 'qwen':
        dollars = float(tokens['cost_usd'])
    else:
        dollars = input_tokens / 1e6 * 0.25 + (output_tokens + thinking_tokens) / 1e6 * 1.50
    return dict(
        n_images=n_images, positives=positives, true_positives=true_positives,
        n_selected=len(shortlist), recall=true_positives / positives,
        precision=true_positives / len(shortlist),
        expected_recall=expected_recall[REVIEW_SET_SIZE - 1],
        expected_recall_at_300=expected_recall[299],
        nonreference_ge45=sum(scores[name] >= 45 and not truth[name] for name in scores),
        cost_per_10k=dollars / n_images * 10_000,
        in_per_img=input_tokens / n_images, out_per_img=output_tokens / n_images,
        think_per_img=thinking_tokens / n_images,
    )


def write_tex(results, output_dir):
    """Write the same tabular blocks used by the manuscript."""
    performance = [
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcrcccr@{}}',
        r'    \toprule',
        rf'    Model & Method & Threshold & N Selected & Recall & Precision & Recall@{REVIEW_SET_SIZE} & \$/10,000 \\',
        r'    & & & & & & (accounting for ties) & Images \\',
        r'    \midrule',
    ]
    costs = [
        r'  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}',
        r'    \toprule',
        r'    Model & Method & Input & Output & Thinking & \$/10,000 \\',
        r'    & & & & & Images \\',
        r'    \midrule',
    ]
    for row, m in results:
        threshold = f"Round {row['threshold']}" if row['method'] == 'Tournament' else str(row['threshold'])
        count = f"{m['n_selected']:,}".replace(',', '{,}')
        columns = [row['model'], rf"\textit{{{row['method']}}}", threshold, count]
        columns += [f"{m[key]:.3f}" for key in ('recall', 'precision', 'expected_recall')]
        columns.append(r'\$' + f"{m['cost_per_10k']:.2f}")
        performance.append('    ' + ' & '.join(columns) + r' \\')
        thinking = f"{m['think_per_img']:.1f}" if row['provider'] == 'gemini' else '--'
        columns = [row['model'], rf"\textit{{{row['method']}}}", f"{m['in_per_img']:.1f}",
                   f"{m['out_per_img']:.1f}", thinking, r'\$' + f"{m['cost_per_10k']:.2f}"]
        costs.append('    ' + ' & '.join(columns) + r' \\')
    performance += [r'    \bottomrule', r'\end{tabular*}']
    costs += [r'    \bottomrule', r'  \end{tabular*}']
    (output_dir / 'table1.tex').write_text('\n'.join(performance) + '\n')
    (output_dir / 'appendix_cost_table.tex').write_text('\n'.join(costs) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PACKAGE_ROOT / 'generated')
    args = parser.parse_args()
    with (RESULTS / 'token_usage.csv').open() as stream:
        usage = {row['run']: row for row in csv.DictReader(stream)}
    results = [(row, evaluate(row['run'], row['threshold'], row['provider'], usage))
               for row in TABLE1]

    print(f"Table 1 - performance against the {results[0][1]['positives']} reference anomalies "
          f"(Expected Recall@{REVIEW_SET_SIZE})")
    print(f"  {'Model':23}{'Method':12}{'Thr':>5}{'N Sel':>7}{'Recall':>8}"
          f"{'Prec':>7}{'ExpR@%d' % REVIEW_SET_SIZE:>10}{'$/10k':>8}")
    for row, m in results:
        threshold = f"R{row['threshold']}" if row['method'] == 'Tournament' else row['threshold']
        print(f"  {row['model']:23}{row['method']:12}{threshold:>5}{m['n_selected']:>7}"
              f"{m['recall']:>8.3f}{m['precision']:>7.3f}{m['expected_recall']:>10.3f}"
              f"{m['cost_per_10k']:>8.2f}")

    replicates = [results[0][1]] + [evaluate(f'gemini_likert_seed{seed}', 33, 'gemini', usage)
                                     for seed in (45, 46)]
    recalls = [result['recall'] for result in replicates]
    sizes = [result['n_selected'] for result in replicates]
    print(f"\nSeed spread: recall {statistics.mean(recalls):.3f} +/- {statistics.stdev(recalls):.3f} (sample SD)")
    print(f"Images selected: {min(sizes)}-{max(sizes)}")
    high = evaluate('gemini_likert_high_thinking_seed44', 33, 'gemini', usage)
    print(f"High thinking: {high['true_positives']} / {high['positives']} reference anomalies, "
          f"${high['cost_per_10k']:.2f} / 10,000 images")
    print(f"Gemini Likert expected recall in top 300: {results[0][1]['expected_recall_at_300']:.6f}")
    print(f"Gemini Likert non-reference candidates at score >= 45: {results[0][1]['nonreference_ge45']}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_tex(results, output)
    print(f'\nWrote table1.tex and appendix_cost_table.tex to {output}')


if __name__ == '__main__':
    main()
