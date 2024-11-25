# Projeto de Detecção de Diabetes com Machine Learning

## Descrição

Este projeto utiliza algoritmos de aprendizado de máquina para prever a presença de diabetes em pacientes com base em dados clínicos. O conjunto de dados contém informações como níveis de glicose, pressão arterial, IMC, histórico familiar de diabetes, entre outros fatores.

## Algoritmos Utilizados

1. **Regressão Logística**
2. **Random Forest**
3. **SVM (Support Vector Machine)**

## Estrutura do Projeto

- `diabetes_pt.csv`: Arquivo com o conjunto de dados utilizado.
- `diabetes_model_code.py`: Código fonte implementado em Python.
- `diabetes_model_report.docx`: Relatório detalhado do projeto.
- `algorithm_accuracies.png`: Gráfico de comparação das acurácias dos algoritmos.

## Pré-Requisitos

1. Python 3.7 ou superior
2. Bibliotecas necessárias (instale com `pip install -r requirements.txt`):
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - python-docx

## Como Executar

1. Certifique-se de que os arquivos mencionados na estrutura do projeto estão no mesmo diretório.
2. Execute o script Python:
   ```bash
   python diabetes_model_code.py
   ```
3. O script irá:
   - Carregar os dados, realizar o pré-processamento e treinar os modelos.
   - Gerar as acurácias dos algoritmos.
   - Criar um gráfico de comparação (salvo como `algorithm_accuracies.png`).

## Resultados

As acurácias dos modelos foram:

- **Regressão Logística**: 70,13%
- **Random Forest**: 78,57%
- **SVM**: 72,08%

O Random Forest apresentou a melhor performance.
