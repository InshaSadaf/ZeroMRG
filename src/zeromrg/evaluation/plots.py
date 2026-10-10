"""Dependency-free SVG graphs for experiment artifacts."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Mapping, Sequence


def _line_chart(title: str, series: Mapping[str, Sequence[float]], y_label: str) -> str:
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 55, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(value) for points in series.values() for value in points]
    maximum = max(values + [1.0])
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">{html.escape(title)}</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111"/>',
             f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111"/>',
             f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="Arial">{html.escape(y_label)}</text>',
             f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="Arial">Epoch</text>']
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * tick / 5
        items.extend([f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                      f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>'])
    for series_index, (name, points) in enumerate(series.items()):
        if not points:
            continue
        denominator = max(1, len(points) - 1)
        coords = [f"{left + plot_w * index / denominator:.1f},{top + plot_h - plot_h * float(value) / maximum:.1f}" for index, value in enumerate(points)]
        color = colors[series_index % len(colors)]
        items.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(coords)}"/>')
        items.append(f'<text x="{left+15+series_index*180}" y="{height-45}" font-family="Arial" font-size="13" fill="{color}">{html.escape(name)}</text>')
    items.append("</svg>")
    return "\n".join(items)


def _bar_chart(metrics: Mapping[str, Mapping[str, object]]) -> str:
    numeric_names = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]
    stages = list(metrics)
    width, height = 1000, 540
    left, top, plot_w, plot_h = 80, 55, 880, 390
    maximum = max([float(metrics[stage].get(name, 0.0)) for stage in stages for name in numeric_names] + [1.0])
    colors = ["#1d4ed8", "#f97316", "#16a34a"]
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>',
             '<text x="500" y="30" text-anchor="middle" font-family="Arial" font-size="22">COV-CTR held-out metrics</text>',
             f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111"/>']
    group_w = plot_w / len(numeric_names)
    bar_w = group_w / max(1, len(stages) + 1)
    for metric_index, name in enumerate(numeric_names):
        base_x = left + metric_index * group_w
        items.append(f'<text x="{base_x+group_w/2:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="13">{name}</text>')
        for stage_index, stage in enumerate(stages):
            value = float(metrics[stage].get(name, 0.0))
            bar_h = plot_h * value / maximum
            x = base_x + (stage_index + 0.5) * bar_w
            y = top + plot_h - bar_h
            items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" height="{bar_h:.1f}" fill="{colors[stage_index%len(colors)]}"/>')
            items.append(f'<text x="{x+bar_w*0.4:.1f}" y="{max(top+10,y-4):.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.2f}</text>')
    for index, stage in enumerate(stages):
        items.append(f'<text x="{left+20+index*210}" y="{height-25}" font-family="Arial" fill="{colors[index%len(colors)]}">{html.escape(stage)}</text>')
    items.append("</svg>")
    return "\n".join(items)


def write_experiment_graphs(output: Path, histories: Mapping[str, list[dict]], metrics: Mapping[str, Mapping[str, object]]) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    loss = {stage: [float(row["train_loss"]) for row in rows] for stage, rows in histories.items()}
    bleu = {stage: [float(row["validation_bleu4"]) for row in rows] for stage, rows in histories.items()}
    files = [output / "training_loss.svg", output / "validation_bleu4.svg", output / "test_metrics.svg"]
    files[0].write_text(_line_chart("COV-CTR training loss", loss, "Cross-entropy"), encoding="utf-8")
    files[1].write_text(_line_chart("COV-CTR validation BLEU-4", bleu, "BLEU-4 (%)"), encoding="utf-8")
    files[2].write_text(_bar_chart(metrics), encoding="utf-8")
    return files
