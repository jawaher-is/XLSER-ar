# import argparse
import yaml
import os
import json
import matplotlib.pyplot as plt

'''
/Users/jia2025/Desktop/XLSER-ar/content/config/wav2vec2-ksuemotions-frozen.yaml
/Users/jia2025/Desktop/XLSER-ar/content/config/xlsr-53-arabic-ksuemotions-finetuned.yaml
/Users/jia2025/Desktop/XLSER-ar/content/config/xlsr-53-arabic-ksuemotions-frozen.yaml
/Users/jia2025/Desktop/XLSER-ar/content/config/xls-r-300m-arabic-ksuemotions-finetuned-noattn.yaml
/Users/jia2025/Desktop/XLSER-ar/content/config/xls-r-300m-arabic-ksuemotions-finetuned.yaml
/Users/jia2025/Desktop/XLSER-ar/content/config/xls-r-300m-arabic-ksuemotions-frozen-noattn.yaml
/Users/jia2025/Desktop/XLSER-ar/content/config/xls-r-300m-arabic-ksuemotions-frozen.yaml
/Users/jia2025/Desktop/XLSER-ar/content/config/xlsr-53-greek-ksuemotions-frozen.yaml
'''


def make_plot(dict_, step, xlabel, ylabel):

    f, ax = plt.subplots()
    chart_title = (step + ' ' + ylabel).capitalize()
    plt.title(chart_title)

    all_models = ''
    for model_name in dict_:
        xData = dict_[model_name][step + '_' + xlabel]
        yData = dict_[model_name][step + '_' + ylabel]
        plt.plot(xData, yData, label=model_name)

        all_models = all_models + '_' + model_name

    plt.xlabel(xlabel.capitalize())
    plt.ylabel(ylabel.capitalize())
    plt.grid(True)
    plt.legend()
    ax.set_ylim(0)

    plt.savefig('./plots/output/' + step + '_' + ylabel + '_' + xlabel + str(len(all_models)) + '.png', dpi=300)
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

    config_files_list = []

    while True:
        usr_input = input("Enter the configuration file paths separated by a new line (or press return to end): ").splitlines()
        if len(usr_input) < 1:
            break
        if isinstance(usr_input, list):
            for config_file in usr_input:
                config_files_list.append(config_file)
        else:
            config_files_list.append(usr_input)

    dict_ = {}
    for config_file in config_files_list:

        with open(config_file) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)

        model_name = configuration['output_dir'].split('/')[-1]
        label = input('Enter a label for ' + model_name + ' : ')

        trainer_state_path = configuration['output_dir'] + configuration['checkpoint'] + '/trainer_state.json'
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)

        train_epoch, train_step, learning_rate, loss, eval_epoch, eval_step, eval_accuracy, eval_loss = get_data(trainer_state)
        data = {}
        data = {'train_epoch': train_epoch, 'train_step': train_step,  'train_loss': loss, 'learning_rate': learning_rate, 'eval_epoch': eval_epoch, 'eval_step': eval_step, 'eval_accuracy': eval_accuracy, 'eval_loss': eval_loss}

        dict_[label] = data


    # make_plot(dict_, "train", "step", "loss")
    make_plot(dict_, "train", "epoch", "loss")

    # make_plot(dict_, "eval", "step", "loss")
    make_plot(dict_, "eval", "epoch", "loss")
    make_plot(dict_, "eval", "epoch", "accuracy")
