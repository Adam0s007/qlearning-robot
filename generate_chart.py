import pandas as pd
import matplotlib.pyplot as plt

FILENAME     = 'EpisodeRewards.txt'
WINDOW_SIZE  = 100

def read_df():
    rows = []
    with open(FILENAME, encoding='utf-8') as f:
        for ln in f:
            ep, rew = ln.strip().split(';')
            rows.append((int(ep), float(rew.replace(',', '.'))))
    df = pd.DataFrame(rows, columns=['Episode', 'Reward'])
    df['MA'] = df['Reward'].rolling(WINDOW_SIZE, min_periods=1).mean()
    return df

# Wczytanie danych tylko raz
df = read_df()

# Tworzenie wykresu
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Episode'], df['MA'], label=f'{WINDOW_SIZE}-episode moving avg')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Moving Average of Rewards')
ax.grid(True)
ax.legend()

# Zapis wykresu i pokazanie go
fig.savefig('chart.png', dpi=150)
print('Zapisano wykres do chart.png')
plt.show()
