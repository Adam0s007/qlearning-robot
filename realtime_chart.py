import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from matplotlib import cm
import numpy as np

FILENAME = 'EpisodeRewards.txt'
WINDOW_SIZE = 100
MAX_EPISODES = 500
REFRESH_INTERVAL = 0.4  # sekundy między aktualizacjami

def read_last_episodes():
    rows = []
    with open(FILENAME, encoding='utf-8') as f:
        lines = f.readlines()[-MAX_EPISODES:]
        for ln in lines:
            ep, rew = ln.strip().split(';')
            rows.append((int(ep), float(rew.replace(',', '.'))))
    df = pd.DataFrame(rows, columns=['Episode', 'Reward'])
    df['MA'] = df['Reward'].rolling(WINDOW_SIZE, min_periods=1).mean()
    return df

plt.ion()  # tryb interaktywny
fig, ax = plt.subplots(figsize=(10, 6))

while True:
    if os.path.exists(FILENAME):
        df = read_last_episodes()
        ax.clear()
        ax.plot(df['Episode'], df['MA'], label=f'{WINDOW_SIZE}-episode moving avg')

        # Zakres Y
        y_min, y_max = df['MA'].min(), df['MA'].max()
        start_y = int(np.floor(y_min / 5)) * 5
        end_y = int(np.ceil(y_max / 5)) * 5
        levels = list(range(start_y, end_y + 1, 5))

        # Podział na ujemne i dodatnie
        neg_levels = [lvl for lvl in levels if lvl < 0]
        pos_levels = [lvl for lvl in levels if lvl > 0]

        # Kolory dla poziomych linii
        cmap_neg = cm.get_cmap('Blues', len(neg_levels))
        cmap_pos = cm.get_cmap('autumn', len(pos_levels))

        for i, level in enumerate(neg_levels):
            ax.axhline(y=level, color=cmap_neg(i), linestyle='--', linewidth=0.8, alpha=0.6)
        for i, level in enumerate(pos_levels):
            ax.axhline(y=level, color=cmap_pos(i), linestyle='--', linewidth=0.8, alpha=0.6)

        # Szara linia Y=0
        if y_min <= 0 <= y_max:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Live Moving Average of Rewards')
        ax.grid(True)
        ax.legend()
        plt.draw()
        plt.pause(0.01)
    time.sleep(REFRESH_INTERVAL)
