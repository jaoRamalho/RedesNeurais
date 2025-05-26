# Projeto de Regressão e Classificação de Gravidade de Vítimas

Este projeto foi desenvolvido como parte da disciplina **Sistemas Inteligentes 1** (UTFPR - 2023/1) e tem como objetivo aplicar técnicas de aprendizado de máquina para **reconstruir a fórmula médica de gravidade** e **classificar vítimas de desastres naturais** em diferentes níveis de gravidade, a partir de sinais vitais.

## 🧠 Objetivos

- Utilizar **duas técnicas de regressão** para estimar o valor de gravidade (`g`).
- Utilizar **duas técnicas de classificação** para prever a classe da vítima (`y`), que pode ser:
  - `1`: crítico
  - `2`: instável
  - `3`: potencialmente estável
  - `4`: estável
- Comparar os resultados com base em métricas padronizadas (RMSE, acurácia, precisão, recall, f1-score, matriz de confusão).

## 🗃️ Estrutura de Arquivos

- `sinaisvitais_hist.txt`: dados de treino com valores de sinais vitais, gravidade e classe.
- `sinaisvitais_teste.txt`: dados de teste (sem rótulo), para teste cego.
- `avaliacao.py`: avaliação dos modelos com métricas.
- `saida_teste.csv`: arquivo com os resultados da predição para o teste cego (formato: i, gravidade_predita, classe_predita).
- `README.md`: este arquivo.
- `artigo_final.pdf`: artigo conforme modelo da SBC (máx. 10 páginas).