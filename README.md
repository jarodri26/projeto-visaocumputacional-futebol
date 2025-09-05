# Futebol Pipeline

Este projeto implementa um pipeline de visão computacional para análise de lances de futebol a partir de vídeos. O objetivo é automatizar a extração de frames, a geração de pseudo-rótulos, o treinamento de modelos e a inferência, facilitando a análise e indexação de jogadas importantes, como gols, faltas determinantes, precisão de pases, precisão de chutes ao gol, entre outros.

## Finalidade

O pipeline foi criado para processar vídeos de partidas de futebol, permitindo:
- Extrair frames de vídeos de lances (por exemplo, um gol).
- Gerar anotações automáticas (pseudo-rótulos) usando modelos pré-treinados.
- Treinar modelos de visão computacional para identificar eventos ou objetos (como gols, jogadores, bola).
- Avaliar e servir modelos para análise automática de novos vídeos.

Esse fluxo pode ser usado para criar datasets, automatizar a análise de partidas, gerar estatísticas ou highlights, e acelerar pesquisas em grandes volumes de vídeos esportivos.

---

## Como rodar no Databricks

O pipeline está preparado para execução no Databricks usando bundles. O notebook principal (`Runner.ipynb`) centraliza todas as etapas do fluxo.

### Passos para rodar no Databricks

1. **Configure os caminhos ou path dos dados**
   - Edite o arquivo `configs/config.yaml` e ajuste os seguintes campos:
     - `video_path`: caminho do vídeo a ser processado (ex: `/<Volumes>/<projeto_visaocomputacional>/<Sua-Pasta>/video-cut.mp4`)
     - `frames_dir`: diretório onde os frames extraídos serão salvos (ex: `/<Volumes>/<projeto_visaocomputacional>/<Sua-Pasta>/frames`)
     - `model_dir`: diretório para salvar o modelo treinado (ex: `/<Volumes>/<projeto_visaocomputacional>/<Sua-Pasta>/models`)
     - `sample_rate`: taxa de amostragem de frames do vídeo (ex: `2` para dois frames por segundo)
     - `num_classes`: número de classes para o modelo (ex: `2`)
   - Certifique-se de que esses diretórios existem no seu ambiente Databricks.
   - no arquivo train.py tem que colocar o caminho de seu expimento
      - ex: `experiment_name = "/Users/<seu-usuario>/<nome-do-experimento>"`


2. **Instale as dependências**
   - Execute no terminal ou notebook:
     ```bash
     pip install -r requirements.txt
     ```

3. **Execute o pipeline**
   - O bundle já está configurado para rodar o notebook `Runner.ipynb` como task principal.
   - Você pode rodar o pipeline via interface do Databricks ou usando o workflow de CI/CD já configurado.

---

## Como rodar localmente

Se preferir rodar fora do Databricks, basta seguir os mesmos passos de configuração e executar os scripts individualmente:

```bash
# Extrair frames do vídeo
python src/data/extract_frames.py

# Treinar o modelo
python -m src.models.train
```

---

## Estrutura do Projeto

- `configs/config.yaml`: configurações de caminhos e parâmetros do pipeline.
- `Runner.ipynb`: notebook principal que executa todas as etapas.
- `src/`: código-fonte do pipeline (extração de frames, dataloader, modelos, serving).
- `scripts/`: scripts auxiliares (pseudo-labeling, validação de caminhos).
- `.github/workflows/deploy.yml`: automação de deploy no Databricks via GitHub Actions.

---

## Variáveis de configuração (`configs/config.yaml`)

- `video_path`: caminho do vídeo de entrada.
- `frames_dir`: pasta onde os frames extraídos serão salvos.
- `sample_rate`: taxa de amostragem de frames do vídeo.
- `model_dir`: pasta onde o modelo treinado será salvo.
- `num_classes`: número de classes do modelo.

---

## Observações

- Certifique-se de que os caminhos configurados existem no Databricks ou no ambiente local.
- O pipeline pode ser expandido para outros tipos de eventos ou esportes, bastando ajustar os scripts e modelos.
- Para dúvidas ou sugestões, abra uma issue no repositório.



