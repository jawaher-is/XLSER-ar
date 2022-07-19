import argparse
import yaml
import os
import json
import matplotlib.pyplot as plt


def make_plot(xData, yData, chart_title, xlabel, ylabel):
    f, ax = plt.subplots()
    plt.plot(xData,yData)
    # plt.title(chart_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    ax.set_ylim(0)

    results_path = save_fig()
    plt.savefig(results_path + '/' + chart_title + '.png')
    # plt.show()
    plt.close()





def get_data(trainer_state):
    log_history = trainer_state['log_history']

    train_epoch = []
    train_step = []
    learning_rate = []
    # learning_rate = [log['learning_rate'] for i, log in enumerate(log_history) if i % 2 == 0]
    loss = []
    # step = []

    eval_epoch = []
    eval_step = []
    eval_accuracy = []
    eval_loss = []

    for i, log in enumerate(log_history):
        if i % 2 == 0:
            train_epoch.append(log['epoch'])
            train_step.append(log['step'])
            learning_rate.append(log['learning_rate'])
            loss.append(log['loss'])

        elif i % 2 == 1:
            eval_epoch.append(log['epoch'])
            eval_step.append(log['step'])
            eval_accuracy.append(log['eval_accuracy']*100)
            eval_loss.append(log['eval_loss'])

    return train_epoch, train_step, learning_rate, loss, eval_epoch, eval_step, eval_accuracy, eval_loss


# # Loss line chart
# xData = eval_step
# yData = eval_loss
# plt.plot(xData,yData)
# plt.title('Training Loss')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

def save_fig():
    results_path = configuration['output_dir'] + '/results'
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    return results_path


if __name__ == '__main__':
    # Get the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path') # EXAMPLE content/config/test.yaml
    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print('Loaded configuration file: ', config_file)
    print(configuration)


    trainer_state_path = configuration['output_dir'] + configuration['checkpoint'] + '/trainer_state.json'
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    model_name = configuration['output_dir'].split('/')[-1]
    train_epoch, train_step, learning_rate, loss, eval_epoch, eval_step, eval_accuracy, eval_loss = get_data(trainer_state)
    make_plot(train_step, loss, model_name + '-train-loss-steps', "Steps", "Loss")
    make_plot(train_epoch, loss, model_name + '-train-loss-epochs', "Epochs", "Loss")

    make_plot(eval_step, eval_loss, model_name + '-eval-loss-steps', "Steps", "Loss")
    make_plot(eval_epoch, eval_loss, model_name + '-eval-loss-epochs', "Epochs", "Loss")
    make_plot(eval_epoch, eval_accuracy, model_name + '-eval-acc', "Epochs", "Accuracy")
