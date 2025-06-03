import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo CSV
file_path = 'data/training/stats_(-8-16-4)_(-1-7-2)_5000_1000.csv'  # Substitua pelo caminho do seu arquivo CSV
data = pd.read_csv(file_path, on_bad_lines='skip')

# Plotar os dados
plt.figure(figsize=(12, 8))

# Plotar MSE e RMSE
plt.subplot(2, 2, 1)
plt.plot(data['Epoch'], data['MSE'], label='MSE', marker='o')
plt.plot(data['Epoch'], data['RMSE'], label='RMSE', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Erro')
plt.title('MSE e RMSE por Epoch')
plt.legend()
plt.grid()

# Plotar Accuracy
plt.subplot(2, 2, 2)
plt.plot(data['Epoch'], data['Accuracy'], label='Accuracy', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy por Epoch')
plt.legend()
plt.grid()

# Plotar Precision e Recall
plt.subplot(2, 2, 3)
plt.plot(data['Epoch'], data['Precision'], label='Precision', marker='o', color='orange')
plt.plot(data['Epoch'], data['Recall'], label='Recall', marker='o', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Valores')
plt.title('Precision e Recall por Epoch')
plt.legend()
plt.grid()

# Plotar F1Score
plt.subplot(2, 2, 4)
plt.plot(data['Epoch'], data['F1Score'], label='F1Score', marker='o', color='red')
plt.xlabel('Epoch')
plt.ylabel('F1Score')
plt.title('F1Score por Epoch')
plt.legend()
plt.grid()

plt.suptitle('Grafico de chegada a um min local, (tanh, none, softmax) - (8, 16, 4) - 0.05', fontsize=16)

# Ajustar layout e mostrar os gráficos
plt.tight_layout()
plt.show()