import re
import matplotlib.pyplot as plt

import re
import matplotlib.pyplot as plt


def read_log(file_path):
    with open(file_path, 'r') as file:
        log_data = file.readlines()
    return log_data


def extract_metrics(log_data):
    epochs = []
    accs = []
    recalls = []
    f1s = []

    current_epoch = None
    for line in log_data:
        # Check for epoch information
        epoch_match = re.search(r'(\d+)/30 - \d+\.?\d*%', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Check for [test] metrics
        test_match = re.search(r'\[test\] loss: .*?, acc: (\d+\.\d+), recall: (\d+\.\d+), f1: (\d+\.\d+)', line)
        if test_match and current_epoch is not None:
            acc = float(test_match.group(1))
            recall = float(test_match.group(2))
            f1 = float(test_match.group(3))

            # print(f"epoch:{current_epoch},acc:{acc},recall:{recall},f1:{f1}")
            epochs.append(current_epoch)
            accs.append(acc)
            recalls.append(recall)
            f1s.append(f1)

    return epochs, accs, recalls, f1s


def plot_metrics(epochs, accs, recalls, f1s):
    plt.figure(figsize=(12, 6))

    plt.plot(epochs, accs, marker='o', label='Accuracy')
    plt.plot(epochs, recalls, marker='o', label='Recall')
    plt.plot(epochs, f1s, marker='o', label='F1 Score')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Test Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)  # Ensure all epochs are marked on the x-axis
    plt.show()


# 主函数
def main():
    file_path = './logs/clcnn_24-09-12_09-33-28.log'
    log_data = read_log(file_path)
    epochs, accs, recalls, f1s = extract_metrics(log_data)
    plot_metrics(epochs, accs, recalls, f1s)


if __name__ == "__main__":
    main()
