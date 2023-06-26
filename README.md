# Leveraging Semantics for Style Representation Learning

This project is the implementation of my master's thesis in Artificial Intelligence, titled "Leveraging Semantically Similar Utterances to Enhance Writing Style Embedding Models". The project investigates the effectiveness of incorporating semantically similar utterances into transformer-based architectures for modeling linguistic style. 

A novel approach was adopted to improve the accuracy and robustness of the style representation model. Semantically similar pairs of utterances, identified using the `all-mpnet-base-v2` model from the `Sentence-Transformers` library, were employed to train the model. The use of these pairs allowed the model to discern and capture subtle stylistic nuances, enhancing its performance over versions of the same model that did not employ any content control. 

Data was drawn from numerous conversations on Reddit, enriching the complexity of writing styles represented in the model. This strategy bolstered the model's robustness and allowed for a more comprehensive evaluation of its ability to capture style. To assess the performance of the models, the STyle EvaLuation (STEL) framework was utilized. 

The results of the STEL evaluation helped ascertain the models' ability to accurately capture writing style and delineate the impact of introducing semantically similar pairings. The study found that although using semantically similar utterances resulted in substantial improvements over not using any form of content control, relying solely on such utterances as different-author examples was not the most optimal strategy. 

Instead, a combination of this approach with others, such as the conversation-based sampling of different-author examples, showed promise for future research. The investigation also highlighted effective techniques for preparing input data, underscoring the importance of author and topic diversity. These insights contribute to the advancement of style-content disentanglement tasks and pave the way for more nuanced and robust style representations.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project requires Python 3.10.6. You can download it [here](https://www.python.org/downloads/release/python-3106/). 

Additionally, you will need to install several Python packages which are listed in `requirements.txt`. You can install these packages using pip by running the following command in your terminal:

```
pip install -r requirements.txt
```

While using a GPU is optional, it is highly recommended for both mining the paraphrases and training/testing the model due to the computational intensity of these tasks. 

Please note that the `convokit` library, which is used in this project, stores everything in RAM. Therefore, if you're working with a large corpus, ensure that your machine has sufficient RAM.

If you want to use the STEL framework to evaluate the models, you will need to have access to the STEL data. You can find it [here](https://github.com/nlpsoc/stel).

## Usage

The main script of this project is `main.py` located in the `src` directory. You can run this script with several options:

```
usage: main.py [-h] [--data-source DATA_SOURCE] [-p PATH]
[--output-path OUTPUT_PATH] [--cache-path CACHE_PATH]
[-b BATCH_SIZE] [-e EPOCHS] [-m MODE] [--model-path MODEL_PATH]
```

### Options

- `-h, --help`: Show the help message and exit.
- `--data-source DATA_SOURCE`: The data source to use for the model. Default is 'reddit'.
- `-p PATH, --path PATH`: The path to the data directory. Default is 'data/'.
- `--output-path OUTPUT_PATH`: The path to the output directory. Default is 'output/'.
- `--cache-path CACHE_PATH`: The path to the cache directory. All temporary models and data will be saved here. Default is '.cache/'.
- `-b BATCH_SIZE, --batch-size BATCH_SIZE`: The batch size to use for training. Default is 8.
- `-e EPOCHS, --epochs EPOCHS`: The number of epochs to train the model for. Default is 3.
- `-m MODE, --mode MODE`: The mode to run the script in. Default is 'train'.
- `--model-path MODEL_PATH`: The path to the model to use in interactive mode. Default is None.

### Modes

The script can be run in three modes: 'train', 'interactive', and 'evaluate'. 

- In 'train' mode, the script trains a new model. If there is no `path`, `output_path`, or `cache_path`, the necessary folders will be created.
- In 'interactive' mode, the script uses an existing model (specified by `--model-path`) to interactively predict the similarity between two sentences.
- In 'evaluate' mode, the script evaluates an existing model (specified by `--model-path`) on a given dataset (specified by `--path`).

### Examples

Here are some examples of how to run the script:

- To run the script in 'train' mode:

```
python3 src/main.py -p data/ --output-path output/ --cache-path .cache/ -e 4 -b 8
```

- To run the script in 'interactive' mode:

```
python3 src/main.py -m interactive --model-path output/your-sentence-transformer/
```

- To run the script in 'evaluate' mode:

```
python3 src/main.py -m evaluate --model-path output/your-sentence-transformer/ --path data/special-eval-data/
```

## Contributing

Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. I appreciate your help in improving this project.

## Citation

If you use this project in your research, please cite this repository and the associated master's thesis. The BibTeX entry for the thesis is:

```bibtex
@mastersthesis{Koornstra2023,
  author  = {Tim Koornstra},
  title   = {Leveraging Semantically Similar Utterances to Enhance Writing Style Embedding Models},
  school  = {Utrecht University},
  year    = {2023},
  address = {Utrecht, The Netherlands},
  month   = {June},
  note    = {Available at: \url{https://github.com/TimKoornstra/style-embeddings}}
}
```

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
