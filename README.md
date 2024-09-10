# Temporal-causal-representation-for-multivariate-power-load-forecasting
 Temporal-causal-representation-for-multivariate-power-load-forecasting

```markdown
# Temporal Causal Representation for Multivariate Power Load Forecasting

This repository contains the implementation of temporal causal representation models for multivariate power load forecasting. The model is inspired by various open-source projects and focuses on improving the accuracy and efficiency of multivariate time series forecasting.

## 1. Prerequisites

Ensure that you have Python 3.7 or later installed on your system. It is recommended to use a virtual environment or Anaconda for managing dependencies.

## 2. Installation

To install the necessary packages, run the following commands. You can either use `pip` to install the dependencies or `requirements.txt` if provided.

### Using `pip`:

```bash
pip install torch
pip install einops
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install tqdm
```

Alternatively, you can install all dependencies from the `requirements.txt` file (if provided):

```bash
pip install -r requirements.txt
```

### Additional Dependencies

Some additional packages you may need:
- **`einops`**: This package is used for tensor operations, which simplifies reshaping in the model.

```bash
pip install einops
```

## 3. Dataset

You can obtain the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/). Download the dataset (e.g., HPC, Power Load dataset) and place it in the appropriate directory. Ensure that it is properly formatted for the model. If specific pre-processing steps are required, please refer to the dataset section in the repository.

## 4. Running the Model

You can run the model using the following command:

```bash
python -u main_informer.py --model informer --data HPC --seq_len 94 --pred_len 24
```

### Breakdown of the arguments:

- `--model`: The type of model to use. For this project, you should use `informer`.
- `--data`: The dataset to use. For example, `HPC` for the high-performance computing dataset.
- `--seq_len`: The length of the input sequence.
- `--pred_len`: The prediction length (number of steps to forecast).

### Example:

```bash
python -u main_informer.py --model informer --data HPC --seq_len 94 --pred_len 24
```

This will train the `informer` model on the `HPC` dataset using a sequence length of 94 and predicting 24 steps ahead.

## 5. Training Options

You can modify the training parameters by passing additional arguments. For example:

- **Training for a specific number of epochs**:
  
```bash
python -u main_informer.py --model informer --data HPC --seq_len 94 --pred_len 24 --epochs 50
```

- **Changing batch size**:
  
```bash
python -u main_informer.py --model informer --data HPC --seq_len 94 --pred_len 24 --batch_size 64
```

- **Using a different learning rate**:

```bash
python -u main_informer.py --model informer --data HPC --seq_len 94 --pred_len 24 --lr 0.001
```

## 6. Results

The results of the model (such as training/validation loss, accuracy, and predictions) will be logged during training. Checkpoints or final model weights will be saved at the end of training.

You can evaluate the model performance or visualize the predictions using the following command:

```bash
python -u main_informer.py --model informer --data HPC --seq_len 94 --pred_len 24 --evaluate
```

## 7. Visualization

To visualize the forecast results, you can generate plots of the predictions:

```bash
python -u main_informer.py --model informer --data HPC --seq_len 94 --pred_len 24 --plot
```

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 9. Acknowledgements

This work was inspired and supported by several open-source projects. Special thanks to the contributors of the following projects:

- [Informer2020](https://github.com/zhouhaoyi/Informer2020): A great foundation for efficient and scalable transformer-based forecasting models.
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear): An excellent resource for long-term time series forecasting with linear models.

## 10. Contributing

Feel free to open issues or contribute to the project by submitting pull requests. Please ensure your code follows the repository's coding standards.
```
